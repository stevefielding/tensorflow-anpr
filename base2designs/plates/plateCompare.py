import numpy as np

class PlateCompare():
  def __init__(self):
    self.cum_plateWithCharMatchCnt = 0
    self.cum_plateFrameMatchCnt = 0
    self.cum_plateCnt_gt = 0
    self.cum_plateCnt_pred = 0
    self.cum_charMatchCnt = 0
    self.cum_charCnt_gt = 0
    self.cum_charCnt_pred = 0

  # generate some stats for the images analysed
  # platesWithCharCorrect - Plates detected in correct location and containing correct characters in the correct locations
  # platesCorrect - Plates in correct location, but no checking of characters
  # platesIncorrect - Plates detected outside of correct location. Calculated as a percentage, but bear in mind that
  #                   the plates outside the correct location is unbounded, so plateCorrect+platesIncorrect may not add
  #                   up to 100%
  # charsCorrect - Characters detected in the correct place with the correct contents
  # charsIncorrect - Chars detected outside of the correct location, or the location is correct,
  #                  but the contents are wrong. Calculated as a percentage, but bear in mind that
  #                  the number of characters outside the correct location is unbounded, so charsCorrect+charsIncorrect
  #                  may not add up to 100%
  def calcStats(self):
    platesWithCharCorrect = self.cum_plateWithCharMatchCnt / self.cum_plateCnt_gt
    platesCorrect = self.cum_plateFrameMatchCnt / self.cum_plateCnt_gt
    platesIncorrect = (self.cum_plateCnt_pred - self.cum_plateFrameMatchCnt) / self.cum_plateCnt_gt
    charsCorrect = self.cum_charMatchCnt / self.cum_charCnt_gt
    charsIncorrect = (self.cum_charCnt_pred - self.cum_charMatchCnt) / self.cum_charCnt_gt
    return platesWithCharCorrect, platesCorrect, platesIncorrect, charsCorrect, charsIncorrect

  # calculate the intersection over union of two boxes
  def intersectionOverUnion(self, box1, box2):
    (box1StartY, box1StartX, box1EndY, box1EndX) = box1
    (box2StartY, box2StartX, box2EndY, box2EndX) = box2
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box2StartX, box1StartX)
    yA = max(box2StartY, box1StartY)
    xB = min(box2EndX, box1EndX)
    yB = min(box2EndY, box1EndY)

    # if the boxes are intersecting, then compute the area of intersection rectangle
    if xB > xA and yB > yA:
      interArea = (xB - xA) * (yB - yA)
    else:
      interArea = 0.0

    # compute the area of the box1 and box2
    box1Area = (box1EndY - box1StartY) * (box1EndX - box1StartX)
    box2Area = (box2EndY - box2StartY) * (box2EndX - box2StartX)

    # compute the intersection area / box1 area
    iou = interArea / float(box1Area + box2Area - interArea)

    # return the intersection over area value
    return iou

  # compare ground truth and predicted results
  # return stats for this compare and keep cumulative stats in instance variables that can be retrieved
  # using calcStats
  def comparePlates(self, plateBoxes_gt, charBoxes_gt, charTexts_gt, plateBoxes_pred, charBoxes_pred, charTexts_pred):
    # set mask to all true
    maskPlate_pred = np.ones(len(plateBoxes_pred), dtype=bool)

    matchIndices = []
    plateFrameMatchCnt = 0
    charCntTotal_gt = 0
    # loop over the ground truth and predicted plate boxes
    for (i, plateBox_gt) in enumerate(plateBoxes_gt):
      # keep a count of the total number of gt chars
      charCntTotal_gt += len(charBoxes_gt[i])
      for (j, plateBox_pred) in enumerate(plateBoxes_pred):
        iou = self.intersectionOverUnion(plateBox_gt, plateBox_pred)
        if iou >= 0.5 and maskPlate_pred[j] == True:
          # maintain a list of indices for matching gt and pred plate boxes
          matchIndices.append((i,j))
          maskPlate_pred[j] = False
          plateFrameMatchCnt += 1
          # we have found a match for this plate gt, so move onto the next one
          continue

    # init the counters prior to loop
    plateWithCharMatchCnt = 0
    charCntTotal_pred = 0
    charMatchCntTotal = 0

    # loop over all the matching plate boxes found in the previous code block
    for (i,j) in matchIndices:
      # set mask to all true
      maskChar_pred = np.ones(len(charBoxes_pred[j]), dtype=bool)

      # grab all the char boxes and text for the current plate
      chBoxes_gt = charBoxes_gt[i]
      chBoxes_pred = charBoxes_pred[j]
      chTexts_gt = charTexts_gt[i]
      chTexts_pred = charTexts_pred[j]

      charMatchCnt = 0
      # loop over the ground truth and predicted char boxes
      for (k, chBox_gt) in enumerate(chBoxes_gt):
        for (l, chBox_pred) in enumerate(chBoxes_pred):
          # If the iou is greater than 0.5 and predChar is not already matched, then check
          # if the chars match and increment the match count
          iou = self.intersectionOverUnion(chBox_gt, chBox_pred)
          if iou >= 0.5 and maskChar_pred[l] == True:
            maskChar_pred[l] = False
            if chTexts_gt[k] == chTexts_pred[l]:
              charMatchCnt += 1
            # we have found a match for this gt char, so move onto the next gt char
            continue

      # check for perfect plate match. ie plate boxes match, and char boxes and labels match
      if charMatchCnt == len(chBoxes_gt) and charMatchCnt == len(chBoxes_pred):
        plateWithCharMatchCnt += 1

      # update the counters
      charMatchCntTotal += charMatchCnt
      charCntTotal_pred += len(chBoxes_pred)

    plateCntTotal_gt = len(plateBoxes_gt)
    plateCntTotal_pred = len(plateBoxes_pred)

    # maintain the cumulative scores
    self.cum_plateWithCharMatchCnt += plateWithCharMatchCnt
    self.cum_plateFrameMatchCnt += plateFrameMatchCnt
    self.cum_plateCnt_gt += plateCntTotal_gt
    self.cum_plateCnt_pred += plateCntTotal_pred
    self.cum_charMatchCnt += charMatchCntTotal
    self.cum_charCnt_gt += charCntTotal_gt
    self.cum_charCnt_pred += charCntTotal_pred

    return plateWithCharMatchCnt, plateFrameMatchCnt, plateCntTotal_gt, plateCntTotal_pred, charMatchCntTotal, charCntTotal_gt, charCntTotal_pred