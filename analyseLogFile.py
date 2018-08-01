# -------------------- analyseLogFile.py -------------------------
# USAGE
# python analyzeLogFile.py --logFile lplateLogExample.txt --reportFile lplateReport.md
# python analyzeLogFile.py --logFile ../../datasets/lplates/train/images/lplateLog.txt --reportFile ../../datasets/lplates/train/lplateReport.md

# Process the log file
# Group license plates that match by at least 5 chars
# Discard duplicate license plates that are in the same video clip, and within 1000 frames
# Generate a Markdown compatible report file with links to image files
# Markdown extension for Chrome web browser can be used to view the file

import copy
import argparse
import numpy as np
import os
import sys
from operator import itemgetter

MIN_FRAME_GAP_BETWEEN_UNIQUE_PLATES = 1000
MIN_PLATE_REPS = 1

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--logFile", required=True,
	help="path to log file")
ap.add_argument("-r", "--reportFile", required=True,
	help="path to report file")
args = vars(ap.parse_args())

# check arguements
if os.path.exists(args["logFile"]) == False:
  print("[ERROR]: --logFile \"{}\" does not exist".format(args["logFile"]))
  sys.exit()
if os.path.exists(args["reportFile"]) == True:
  print("[ERROR]: --reportFile \"{}\" already exists. Delete first".format(args["reportFile"]))
  sys.exit()

def plateMatch(plateText1, plateText2, matchThres):
  # Try to match the plate texts without shifting
  matchCnt = 0
  minPlateLen = min(len(plateText1), len(plateText2))
  maxPlateLen = max(len(plateText1), len(plateText2))
  for i in np.arange(minPlateLen):
    if plateText1[i] == plateText2[i]:
      matchCnt += 1
  if matchCnt >= matchThres:
    return True

  # if the plate texts are equal in length, and they have not matched, then report bad match
  if minPlateLen == maxPlateLen:
    return False

  # The plate texts have not matched so far, and they are unequal length, so find the shortest
  if len(plateText1) < len(plateText2):
    plateShort, plateTextLong = plateText1, plateText2
  else:
    plateShort, plateTextLong = plateText2, plateText1

  # Assume that the shorter plate text has a missing character.
  # Try inserting a space in place of the missing character
  # Shift the insertion point from beginning to end of the string
  bestMatchCnt = 0
  for i in range(len(plateShort)):
    if i == 0:
      plateTextPad = " " + plateShort
    elif i == len(plateShort):
      plateTextPad = plateShort + " "
    else:
      plateTextPad = plateShort[0:i+1] + " " + plateShort[i+1:-1]
    matchCnt = 0
    for i in np.arange(len(plateTextPad)):
      if plateTextPad[i] == plateTextLong[i]:
        matchCnt += 1
    bestMatchCnt = max(matchCnt, bestMatchCnt)
  if bestMatchCnt >= matchThres:
    return True
  return False

# read the log file
logFile = open(args["logFile"], "r")
logs = logFile.read()
logFile.close()
logsSplit = [s.strip().split(",") for s in logs.splitlines()]
logsSplit = np.array(logsSplit)
logFileNumEntries = len(logsSplit)
plateDict = dict()

# Create a dictionary (plateDict) with the plateText as the keys
dates = []
for logLine in logsSplit:
  videoFileName = logLine[0]
  imageFileName = logLine[1]
  date = logLine[2]
  dates.append(date)
  time = logLine[3]
  frameNum = int(logLine[4])
  numberOfPlates = int(logLine[5])
  plateTexts = logLine[6:]
  plateTexts = plateTexts[0::2]
  # Create plateDict, and list of all dates
  for plateText in plateTexts:
    if plateText in plateDict:
      # plateText, videoFileName, imageFileName, date, time, frameNumber
      plateDict[plateText].append([plateText, videoFileName, imageFileName, date, time, frameNum, numberOfPlates])
    else:
      plateDict[plateText] = [[plateText, videoFileName, imageFileName, date, time, frameNum, numberOfPlates]]

# Find first and last date
dates = sorted(dates)
firstDate = dates[0]
lastDate = dates[-1]

# combine dictionary keys that match by at least 5 chars
combinedSimilarPlateCnt = 0
plateDictDeDuped = copy.deepcopy(plateDict)
#plateDictDeDuped = {}
plateDict2 = copy.deepcopy(plateDict)
for plateText1 in plateDict.keys():
  del plateDict2[plateText1]
  for plateText2 in plateDict2.keys():
    if plateMatch(plateText1, plateText2, 5) == True:
      # check if plateText2 is in the plateDictDuped. If not then it has already been claimed as a duplicate by
      # another plate, and is not available.
      # If it is available, then copy to dict values at plateText1 and remove plateText2 key
      if plateText2 in plateDictDeDuped.keys():
        entryAdds = plateDict[plateText2]
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
      numberOfPlates = plateEntry[6]
      plateTextInstance = plateEntry[0]
      for i in range(len(charDicts)):
        if i < len(plateTextInstance):
          char = plateTextInstance[i]
        else:
          # Use '*' to denote empty char, and then delete the empty chars when finished
          char = '*'
        if char in charDicts[i].keys():
          charDicts[i][char] += numberOfPlates
        else:
          charDicts[i][char] = numberOfPlates

    # for each histogram, sort and then select the most frequently occurring characters
    predPlateText =[]
    for (i,charDict) in enumerate(charDicts):
      if len(charDict) != 0:
        plateTuples = charDict.items()
        plateChars = sorted(plateTuples, key=lambda x:x[1] )
        predPlateText.append(plateChars[-1][0])
    predPlateText = ''.join(predPlateText)
    predPlateText = predPlateText.replace("*","")

    # Copy dictionary entries from plateDictDeDuped, but use the new predicted plate text as the key
    plateDictPred[predPlateText] = plateDictDeDuped[plateText]

# Remove duplicate license plates from a video clip
# Images that contain the same plateText and come from the
# same video clip are possibly duplicates
# Assume that similar plateText within 1000 frames of each other are
# duplicates, and remove the duplicates
removeDuplicatesFromVideoClip = True
if removeDuplicatesFromVideoClip == False:
  plateDictDeDuped2 = plateDictPred
  deletedVidSeqDupCnt = 0
else:
  plateDictDeDuped2 = {}
  plateDictPredCopy = copy.deepcopy(plateDictPred)
  for plateText in plateDictPredCopy.keys():
    plateDictSubGroup = {}
    if len(plateDictPredCopy[plateText]) == 1:
      # only one entry for this plateText, so simply copy, no need to de-dupe
      plateDictDeDuped2[plateText] = plateDictPredCopy[plateText]
    else:
      # more than one entry, so create a temporary videoFilename sub-group
      for plateEntry in plateDictPredCopy[plateText]:
        if plateEntry[1] in plateDictSubGroup.keys():
          plateDictSubGroup[plateEntry[1]].append(plateEntry)
        else:
          plateDictSubGroup[plateEntry[1]] = [plateEntry]

      # process the videoFilename sub-group
      for imageFileName in plateDictSubGroup:
        plateEntries = plateDictSubGroup[imageFileName]
        if len(plateEntries) == 1:
          # For this videoFile there is only one entry in the sub-group,
          # so need to sort, just add to the dictionary
          if plateText in plateDictDeDuped2.keys():
            plateDictDeDuped2[plateText].append(plateEntries[0])
          else:
            plateDictDeDuped2[plateText] = plateEntries
        else:

          # sort the multiple entries by frame number
          # Add the entry with the smallest frame number to the dict
          # and add subsequent entries only if they are separated by a
          # sufficient number of frames
          deletedVidSeqDupCnt = 0
          plateEntriesSorted = sorted(plateEntries, key=lambda x: x[5])
          for (i,plateEntry) in enumerate(plateEntriesSorted):
            if i == 0:
              frameNumBase = plateEntry[5]
              # add the first entry
              if plateText in plateDictDeDuped2.keys():
                plateDictDeDuped2[plateText].append(plateEntry)
              else:
                plateDictDeDuped2[plateText] = [plateEntry]
            elif plateEntry[5] > frameNumBase + MIN_FRAME_GAP_BETWEEN_UNIQUE_PLATES:
              frameNumBase = plateEntry[5]
              plateDictDeDuped2[plateText].append(plateEntry)
            else:
              deletedVidSeqDupCnt += 1


# generate the report file. Sort by plate, then by date and time
reportFile = open(args["reportFile"], "w")
reportFile.write("##### Report for {} to {}  \n".format(firstDate, lastDate))
sortedKeys = sorted(plateDictDeDuped2.keys())
lowRepPlateCnt = 0
for plateText in sortedKeys:
  plateEntries = plateDictDeDuped2[plateText]
  #plateEntries = sorted (plateEntries, key=lambda x:x[3])
  plateEntries = sorted (plateEntries, key = itemgetter(3,4))
  totalNumPlates = 0
  for plateEntry in plateEntries:
    totalNumPlates += int(plateEntry[6])
  if totalNumPlates > MIN_PLATE_REPS:
    reportFile.write("+ {} count: {}  \n".format(plateText, totalNumPlates))
    for plateEntry in plateEntries:
      # date time plateText videoFileName imageFileName frameNum
      if plateEntry[2] == "NO_IMAGE":
        reportFile.write("    + {} {} {} {} frameNum: {}  \n".format(plateEntry[3], plateEntry[4], plateEntry[0],
                                          plateEntry[1], plateEntry[5] ))
      else:
        reportFile.write("    + {} {} {} {} [imageFile](./{}) frameNum: {}  \n".format(plateEntry[3], plateEntry[4], plateEntry[0],
                                          plateEntry[1], plateEntry[2], plateEntry[5] ))
  else:
    lowRepPlateCnt += 1
reportFile.close()
print("[INFO] Deleted {} video sequence duplicates".format(deletedVidSeqDupCnt))
print("[INFO] Combined {} similar plates".format(combinedSimilarPlateCnt))
print("[INFO] Removed {} plates with only {} repetition".format(lowRepPlateCnt, MIN_PLATE_REPS))
print ("[INFO] Log file has {} entries, and {} unique plates".format(logFileNumEntries, len(plateDict.keys())))
print ("[INFO] After combining similar plates, report file has {} plates".format(len(sortedKeys)-lowRepPlateCnt))







