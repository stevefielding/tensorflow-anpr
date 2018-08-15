# -------------------- analyseLogFile.py -------------------------
# USAGE
# python analyzeLogFile.py --logFile video/C920/short/images/lplateLogSmall.txt --reportFile video/C920/short/images/lplateReport.md
# python analyzeLogFile.py --logFile video/Sunba/images/lplateLogSunba.txt --reportFile video/Sunba/images/lplateReport.md
# Process the log file
# Group license plates that are within MAX_PLATE_TEXT_DIST of each other
# Discard duplicate license plates that are in the same video clip, and within MIN_FRAME_GAP_BETWEEN_UNIQUE_PLATES frames
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
MAX_PLATE_TEXT_DIST = 1
REMOVE_VID_DUPLICATES = True

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

# find the number of characters that are the same in two strings
def matchScore(str1, str2):
  bestMatch = 0

  #ensure that shortest string is first
  if len(str1) > len(str2):
    str1, str2 = str2, str1

  # find best match string
  matchFound = False
  for i in range(len(str1)):
    match = 0
    j = 0
    while j < len(str2) and matchFound == False:
      if str1[i] == str2[j]:
        match += 1
        matchFound = True
        if i < len(str1) - 1 and j < len(str2) - 1:
          match += matchScore(str1[i+1:], str2[j+1:])
      j += 1
    if match > bestMatch:
      bestMatch = match

  return bestMatch

# Compare two plate texts, and return the number of chars that are different
# editDist = 0 indictaes a perfect match
def editDist(str1, str2):
  return max(len(str1), len(str2)) - matchScore(str1, str2)

def groupSimilarPlates(plateDict, matchThres):
  # combine dictionary keys that match by at least 'matchThres'.
  # Results in plateDictDeDuped
  # Plates that differ by no more than matchThres can end up in different groups
  # Say matchThres=1. Group 5pz433 is formed first, and contains plates 5pz433 and 5pzz433.
  # Group 5pzz433f is formed second, and contains 5pzz433f and 5pzz433l
  # Notice that 5pzz433 and 5pzz433l only differ by one char, but are in different groups
  combinedSimilarPlateCnt = 0
  # Create 2 copies of the plateDict. plateDictDeDuped will be the final output
  # plateDict2 is used for comparison
  plateDictDeDuped = copy.deepcopy(plateDict)
  plateDict2 = copy.deepcopy(plateDict)
  # Loop over all the keys in the original dictionary
  for plateText1 in plateDict.keys():
    # remove plateText1 from plateDict2, so we don't compare the exact same plate text
    del plateDict2[plateText1]
    # Loop over all the remaining keys in plateDict2, and look for close match to plateText1
    matchingPlateTexts = []
    for plateText2 in plateDict2.keys():
      score = editDist(plateText1, plateText2)
      # if match score is below threshold, then add the plate to the group
      if score <= matchThres:
        matchingPlateTexts.append(plateText2)


    # check if matchingPlateText is in the plateDictDuped. If not then it has already been claimed as a duplicate by
    # another plate, and is not available.
    # If it is available, then copy to dict values at plateText1 and remove plateText2 key
    for matchingPlateText in matchingPlateTexts:
      if matchingPlateText in plateDictDeDuped.keys():
        entryAdds = plateDict[matchingPlateText]
        for entryAdd in entryAdds:
          # if plateText1 key has already been deleted, then add it back again
          if plateText1 in plateDictDeDuped.keys():
            plateDictDeDuped[plateText1].append(entryAdd)
          else:
            plateDictDeDuped[plateText1] = [entryAdd]
            combinedSimilarPlateCnt -= 1
        del plateDictDeDuped[matchingPlateText]
        combinedSimilarPlateCnt += 1
  return plateDictDeDuped, combinedSimilarPlateCnt

# Change the key so it represents the most popular plate text for each group
def findBestKey_wordbased(plateDictDeDuped):
  plateDictPred = {}
  for plateText in plateDictDeDuped.keys():
    if len(plateDictDeDuped[plateText]) == 1:
      # if there is only one plate, then copy to predicted plates
      plateDictPred[plateText] = plateDictDeDuped[plateText]
    else:
      # For each word find word frequency
      plateDict = {}
      for plateEntry in plateDictDeDuped[plateText]:
        numberOfPlates = plateEntry[6]
        plateTextInstance = plateEntry[0]
        plateDict[plateTextInstance] = plateDict.get(plateTextInstance,0) + numberOfPlates

      # select the most frequently ocurring plate text
      predPlateText = ""
      bestScore = 0
      for plateText2 in plateDict.keys():
        if plateDict[plateText2] >= bestScore:
          predPlateText = plateText2
          bestScore = plateDict[plateText2]


      if predPlateText not in plateDictPred.keys():
        # predPlateText does not exist.
        # Simply copy dictionary entries from plateDictDeDuped, but use the new predicted plate text as the key
        plateDictPred[predPlateText] = plateDictDeDuped[plateText]
      else:
        # predPlateText already exists. Append plates to existing dict key
        entryAdds = plateDictDeDuped[plateText]
        for entryAdd in entryAdds:
          plateDictPred[predPlateText] = plateDictPred.get(predPlateText, [])
          plateDictPred[predPlateText].append(entryAdd)

  return plateDictPred

def removeDuplicates(plateDictPred, minGapBetweenUniqueFrames):
  # Remove duplicate license plates from a video clip
  # Images that contain the same plateText and come from the
  # same video clip are possibly duplicates
  # Assume that similar plateText within 1000 frames of each other are
  # duplicates, and remove the duplicates
  deletedVidSeqDupCnt = 0
  plateDictDeDuped = {}
  plateDictPredCopy = copy.deepcopy(plateDictPred)
  for plateText in plateDictPredCopy.keys():
    plateDictSubGroup = {}
    if len(plateDictPredCopy[plateText]) == 1:
      # only one entry for this plateText, so simply copy, no need to de-dupe
      plateDictDeDuped[plateText] = plateDictPredCopy[plateText]
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
          if plateText in plateDictDeDuped.keys():
            plateDictDeDuped[plateText].append(plateEntries[0])
          else:
            plateDictDeDuped[plateText] = plateEntries
        else:
          # sort the multiple entries by number of plates
          # Add the entry with the largest number of plates to the dict
          # and add subsequent entries only if they are separated by a
          # sufficient number of frames
          plateEntriesSorted = sorted(plateEntries, key=lambda x: x[6], reverse=True)
          for (i, plateEntry) in enumerate(plateEntriesSorted):
            if i == 0:
              frameNumBase = plateEntry[5]
              # add the first entry
              if plateText in plateDictDeDuped.keys():
                plateDictDeDuped[plateText].append(plateEntry)
              else:
                plateDictDeDuped[plateText] = [plateEntry]
            elif plateEntry[5] > frameNumBase + minGapBetweenUniqueFrames:
              frameNumBase = plateEntry[5]
              plateDictDeDuped[plateText].append(plateEntry)
            else:
              deletedVidSeqDupCnt += 1

  return plateDictDeDuped, deletedVidSeqDupCnt

def genReportFile(plateDict, reportFileName, minPlateReps, firstDate, lastDate):
  # generate the report file. Sort by plate, then by date and time
  reportFilePlateCnt = 0
  reportFile = open(reportFileName, "w")
  reportFile.write("##### Report for {} to {}  \n".format(firstDate, lastDate))
  sortedKeys = sorted(plateDict.keys())
  lowRepPlateCnt = 0
  for plateText in sortedKeys:
    plateEntries = plateDict[plateText]
    # plateEntries = sorted (plateEntries, key=lambda x:x[3])
    plateEntries = sorted(plateEntries, key=itemgetter(3, 4))
    totalNumPlates = 0
    for plateEntry in plateEntries:
      totalNumPlates += int(plateEntry[6])
    if totalNumPlates > minPlateReps:
      reportFile.write("+ {} count: {}  \n".format(plateText, totalNumPlates))
      for plateEntry in plateEntries:
        reportFilePlateCnt += 1
        # date time plateText videoFileName imageFileName frameNum
        if plateEntry[2] == "NO_IMAGE":
          reportFile.write("    + {} {} {} {} frameNum: {}  \n".format(plateEntry[3], plateEntry[4], plateEntry[0],
                                                                       plateEntry[1], plateEntry[5]))
        else:
          reportFile.write(
            "    + {} {} {} {} [imageFile](./{}) frameNum: {}  \n".format(plateEntry[3], plateEntry[4], plateEntry[0],
                                                                          plateEntry[1], plateEntry[2], plateEntry[5]))
    else:
      lowRepPlateCnt += 1
  reportFile.close()
  return reportFilePlateCnt, lowRepPlateCnt

def readLogFile(logFileName):
  # read the log file, and group plates with identical text
  logFile = open(logFileName, "r")
  print("[INFO] Processing: \"{}\"".format(args["logFile"]))
  logs = logFile.read()
  logFile.close()
  logsSplit = [s.strip().split(",") for s in logs.splitlines()]
  logsSplit = np.array(logsSplit)
  logFileNumEntries = len(logsSplit)
  plateDict = dict()

  # Create a dictionary (plateDict) with the plateText as the keys
  dates = []
  for logLine in logsSplit:
    if len(logLine) < 7:
      continue
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
        # plateText, videoFileName, imageFileName, date, time, frameNumber, numberOfPlates
        plateDict[plateText].append([plateText, videoFileName, imageFileName, date, time, frameNum, numberOfPlates])
      else:
        plateDict[plateText] = [[plateText, videoFileName, imageFileName, date, time, frameNum, numberOfPlates]]
  numberOfPlateGroupsOriginal = len(plateDict.keys())

  # Find first and last date
  dates = sorted(dates)
  firstDate = dates[0]
  lastDate = dates[-1]
  return plateDict, logFileNumEntries, numberOfPlateGroupsOriginal, firstDate, lastDate

# read the log file
plateDict, logFileNumEntries, numberOfPlateGroupsOriginal, firstDate, lastDate = readLogFile(args["logFile"])

# Group similar plates
plateDict, combinedSimilarPlateCnt = groupSimilarPlates(plateDict, MAX_PLATE_TEXT_DIST)
numberOfPlateGroupsAfterDeDupe = len(plateDict.keys())

# The previous code grouped plates.
# However the dictionary key may not be the best plateText prediction for the group
# Now we change the key so it represents the most popular plate text for each group
plateDict = findBestKey_wordbased(plateDict)

# After changing the plate text keys, perform plate grouping a second time, and fix the dict keys a second time
# This effectively doubles the max plate distance.
# ie if the MAX_PLATE_TEXT_DIST= 1, then it will become 2 after performing this process a second time
plateDict, combinedSimilarPlateCnt2 = groupSimilarPlates(plateDict, MAX_PLATE_TEXT_DIST)
combinedSimilarPlateCnt += combinedSimilarPlateCnt2
plateDict = findBestKey_wordbased(plateDict)

# Remove duplicate license plates
deletedVidSeqDupCnt = 0
if REMOVE_VID_DUPLICATES == True:
  plateDict, deletedVidSeqDupCnt = removeDuplicates(plateDict, MIN_FRAME_GAP_BETWEEN_UNIQUE_PLATES)


# generate the report file. Remove plates with low rep count.
# Sort by plate, then by date and time
reportFilePlateCnt, lowRepPlateCnt = genReportFile(plateDict, args["reportFile"], MIN_PLATE_REPS, firstDate, lastDate)

print ("[INFO] Original log file has {} entries, and {} unique plate groups".format(logFileNumEntries, numberOfPlateGroupsOriginal))
print("[INFO] Combined {} plates, differing by {} chars or less".format(combinedSimilarPlateCnt, MAX_PLATE_TEXT_DIST))
print("[INFO] After combining similar plates, left with {} plate groups".format(numberOfPlateGroupsAfterDeDupe))
print("[INFO] Deleted {} video sequence duplicates".format(deletedVidSeqDupCnt))
print ("[INFO] After removing video sequence duplicates, left with {} plate groups".format(len(plateDict.keys())))
print("[INFO] Removed {} plates with only {} repetition".format(lowRepPlateCnt, MIN_PLATE_REPS))
print ("[INFO] After removing low repetition groups, report file has {} plates, and {} plate groups".format(reportFilePlateCnt, len(plateDict.keys())-lowRepPlateCnt))







