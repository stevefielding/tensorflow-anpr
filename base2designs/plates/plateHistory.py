# plate history is maintained in a dictionary that uses the plateText as the keys
# key = plateText
#   val = [ [plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, numberOfPlates],
#           [plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, numberOfPlates] ...]
# This works well if there is a single plate per image. If there are multiple plates, then
# there will be one dictionary entry for each plate, and the same image will be referenced multiple times.
# Thus images with multiple plates are output as multiple images with only one plate per image annotated.
# TODO: Detect when dictionary entries refer to the same image, and combine the plates in the same image

import copy
import numpy as np
import cv2
import re
from base2designs.plates.plateAnn import PlateAnn
class PlateHistory:

  def __init__(self, output_image_path, logFile, saveAnnotatedImage):
    self.rollingPlateDict = {}
    self.savedPlatesList = []
    self.fileCnt = 1
    self.output_image_path = output_image_path
    self.saveAnnotatedImage = saveAnnotatedImage
    self.logFile = logFile

  def scaleBB(self, box, W, H):
    # scale the bounding box from the range [0, 1] to [W, H]
    (startY, startX, endY, endX) = box
    startX = int(startX * W)
    startY = int(startY * H)
    endX = int(endX * W)
    endY = int(endY * H)
    return startX, startY, endX, endY

  # clear the dictionaries. Need to do this at the start of every new video clip
  def clearHistory(self):
    self.rollingPlateDict = {}
    self.savedPlatesList = []

  # plateList: list of plate texts
  # charBoxes: list of char boxes for each plate
  # plateBoxes: list of plate boxes
  # fullImage: Original image from which the plates were cropped
  # videoPath: path to the image
  # frameNumber: video clip frame number
  # rollingPlateDict has the following format:
  # key = plateText
  #   val = [ [plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, numberOfPlates, plateScore],
  #           [plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, numberOfPlates, plateScore] ...]
  def addPlatesToHistory(self, plateList, charBoxes, plateBoxes, fullImage, videoPath, frameNumber, plateScores):
    # add the new plates to rollingPlateDict
    for (plateText, chBoxes, plateBox, plateScore) in zip(plateList, charBoxes, plateBoxes, plateScores):
      if len(plateText) != 0:
        if plateText in self.rollingPlateDict:
          self.rollingPlateDict[plateText].append([plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, 1, plateScore])
        else:
          self.rollingPlateDict[plateText] = [[plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, 1, plateScore]]

  def removeOldPlatesFromHistory(self):
    self.rollingPlateDict = {}

  # Add new plates to the rollingPlateDict
  # Group plates that have similar plate text
  # Select the "best" plate from the group, and delete the others
  # Return the dictionary of best plates
  def selectTheBestPlates(self):

    # combine dictionary keys that match by at least 5 chars
    combinedSimilarPlateCnt = 0
    plateDictDeDuped = copy.deepcopy(self.rollingPlateDict)
    plateDict2 = copy.deepcopy(self.rollingPlateDict)
    for plateText1 in self.rollingPlateDict.keys():
      del plateDict2[plateText1]
      for plateText2 in plateDict2.keys():
        matchCnt = 0
        # compare each char in the plate text
        for i in np.arange(min(len(plateText1), len(plateText2))):
          if plateText1[i] == plateText2[i]:
            matchCnt += 1
        if matchCnt >= 3:
          # check if plateText2 is in the plateDictDuped. If not then it has already been claimed as a duplicate by
          # another plate, and is not available.
          # If it is available, then copy to dict values at plateText1 and remove plateText2 key
          if plateText2 in plateDictDeDuped.keys():
            entryAdds = self.rollingPlateDict[plateText2]
            for entryAdd in entryAdds:
              # if plateText1 key has already been deleted, then add it back again
              if plateText1 in plateDictDeDuped.keys():
                plateDictDeDuped[plateText1].append(entryAdd)
              else:
                plateDictDeDuped[plateText1] = [entryAdd]
                combinedSimilarPlateCnt -= 1
            del plateDictDeDuped[plateText2]
            combinedSimilarPlateCnt += 1

    # The previous code grouped plates that matched by at least 5 chars under the same dictionary key
    # However the dictionary key may not be the best plateText prediction for the group
    # Now we change the key so it represents the most popular combination of characters for each group
    plateDictPred = {}
    for plateText in plateDictDeDuped.keys():
      if len(plateDictDeDuped[plateText]) == 1:
        # if there is only one plate, then copy to predicted plates
        plateDictPred[plateText] = plateDictDeDuped[plateText]
      else:
        # For each char position build a histogram of character frequencies
        charDicts = [{}, {}, {}, {}, {}, {}, {}]
        for plateEntry in plateDictDeDuped[plateText]:
          for (i,char) in enumerate(plateEntry[0]):
            # check for plates that are too long
            if i > 6:
              continue
            if char in charDicts[i].keys():
              charDicts[i][char] += 1
            else:
              charDicts[i][char] = 1

        # for each histogram, sort and then select the most frequently occurring characters
        predPlateText =[]
        for (i,charDict) in enumerate(charDicts):
          # check for dictionary empty, ie no char at this plate position
          if bool(charDict) == True:
            plateTuples = charDict.items()
            plateChars = sorted(plateTuples, key=lambda x:x[1] )
            predPlateText.append(plateChars[-1][0])
        predPlateText = ''.join(predPlateText)

        # Copy dictionary entries from plateDictDeDuped, but use the new predicted plate text as the key
        plateDictPred[predPlateText] = plateDictDeDuped[plateText]

    # we only need one entry per plateText key, but let's try to pick the "best" entry
    # ie find the entry where the plateText is closest to the key
    bestMatchCnt = 0
    plateDictBest = {}
    for plateText in plateDictPred:
      for plateEntry in plateDictPred[plateText]:
        matchCnt = 0
        plateTextEntry = plateEntry[0]
        for i in np.arange(min(len(plateTextEntry), len(plateText))):
          if plateText[i] == plateTextEntry[i]:
            matchCnt += 1
        # if this entry has the best match to the key, then save it
        # We use >= so that we can catch the case where matchCnt = 0
        if matchCnt >= bestMatchCnt:
          bestMatchCnt = matchCnt
          bestPlateEntry = plateEntry
          plateDictBest[plateText] = bestPlateEntry
          numPlates = len(plateDictPred[plateText])
          # We are deleting the extra plate entries, but keep a count of how many there were
          plateDictBest[plateText][6] = numPlates

    return (plateDictBest)

  # write image file, annotation file, and log file
  def writeToFile(self, plateDict, destFolderRootName, imageWidth, imageHeight, imageDepth):
    plateAnn = PlateAnn()
    # for all the plates in plateDict, add to full log if the plate has not been previously seen
    # otherwise add to partial log
    plateDictForFullLog = {}
    plateDictForPartialLog = {}
    for plateText in plateDict:
      if plateText in self.savedPlatesList:
        plateDictForPartialLog[plateText] = plateDict[plateText]
      else:
        self.savedPlatesList.append(plateText)
        plateDictForFullLog[plateText] = plateDict[plateText]

    # prevent the savedPlateList from getting larger than 1000 entries
    if len(self.savedPlatesList) > 1000:
      del(self.savedPlatesList[0:len(self.savedPlatesList)-30])

    # Full log. Copy fullImage and plate Image to file and update log file
    for plateTextKey in plateDictForFullLog.keys():
      (plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, numberOfPlates, plateScore) = plateDictForFullLog[plateTextKey]

      # strip the video file name for use when saving still images
      videoFileName = videoPath.split("/")[-1]
      fileNamePrefix = videoPath.split("/")[-1].split("[")[0]

      # create unique file name for the full image file. Append plate text to the file name
      outputFullImageFileName = "{}_{}_{}_cnt{}_score_{:.2f}".format(fileNamePrefix, self.fileCnt, plateText, numberOfPlates, plateScore)
      outputFullImagePath = "{}/{}/{}.jpg".format(self.output_image_path, destFolderRootName, outputFullImageFileName)
      outputAnnPath = "{}/{}_ann/{}.xml".format(self.output_image_path, destFolderRootName, outputFullImageFileName)
      plateAnn.writeAnnFile(outputAnnPath, outputFullImagePath, plateBox, plateText, chBoxes, imageWidth, imageHeight, imageDepth)
      self.fileCnt += 1

      # optionally annotate the image
      if self.saveAnnotatedImage == True:
        startX, startY, endX, endY = self.scaleBB(plateBox, imageWidth, imageHeight)
        cv2.rectangle(fullImage, (startX, startY), (endX, endY), (0, 0, 255), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(fullImage, plateText, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for chBox in chBoxes:
          startX, startY, endX, endY = self.scaleBB(chBox, imageWidth, imageHeight)
          cv2.rectangle(fullImage, (startX, startY), (endX, endY), (0, 255, 0), 2)

      # print the plate text
      #print("[INFO] Found plate: {}".format(plateText))

      # save the image
      print("[INFO] logging image to file: {}".format(outputFullImagePath))
      cv2.imwrite(outputFullImagePath, fullImage)

      # update the log file
      # videoFileName, imageFileName, date, time, frameNumber, numberOfPlates, plateText, plateScore
      # lplate_toy_video4.mp4,2018_01_01/lplate_toy_video.mp4_51.jpg,2018_01_10,5:05,271,1,5HUY634,0.7
      imageFileName = "{}/{}.jpg".format(destFolderRootName, outputFullImageFileName)
      date = destFolderRootName
      m = re.search(r"^([0-9]{2}[.:][0-9]{2}[.:][0-9]{2})",videoFileName)
      if m:
        time = m.group(1)
      else:
        time = "HH.MM.SS"
      self.logFile.write("{},{},{},{},{},{},{},{:.2f}\n".format(videoFileName, imageFileName, date, time, frameNumber, numberOfPlates, plateText, plateScore))
      self.logFile.flush()

    # Partial log. Just update the log file
    for plateTextKey in plateDictForPartialLog.keys():
      (plateText, chBoxes, plateBox, fullImage, videoPath, frameNumber, numberOfPlates, plateScore) = plateDictForPartialLog[plateTextKey]

      # strip the video file name for use when saving still images
      videoFileName = videoPath.split("/")[-1]

      # update the log file
      # videoFileName, imageFileName, date, time, frameNumber, numberOfPlates, plateText, plateScore
      # eg: lplate_toy_video4.mp4,2018_01_01/lplate_toy_video.mp4_51.jpg,2018_01_10,5:05,271,1,5HUY634,0.7
      date = destFolderRootName
      m = re.search(r"^([0-9]{2}[.:][0-9]{2}[.:][0-9]{2})",videoFileName)
      if m:
        time = m.group(1)
      else:
        time = "HH.MM.SS"
      self.logFile.write("{},{},{},{},{},{},{},{:.2f}\n".format("NO_VIDEO", "NO_IMAGE", date, time, frameNumber, numberOfPlates, plateText, plateScore))
      self.logFile.flush()
