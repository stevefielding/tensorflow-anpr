
import numpy as np

class PlateFinder:

  def __init__(self, minConfidence, categoryIdx, charIOUMax=0.3, charPlateIOAMin=0.5, rejectPlates=False, minScore=0.6, minChars=2):
    # boxes below minConfidence are rejected
    self.minConfidence = minConfidence
    # character boxes that do not overlap plate box by at least 'charPlateIOAMin' are rejected
    self.charPlateIOAMin = charPlateIOAMin
    # character boxes overlapping other char boxes by more than 'charIOUMax' are rejected
    self.charIOUMax = charIOUMax
    # If 'rejectPlates' is True, then plates with a "complete score" less than minScore,
    # or less than 'minChars' characters, or plate boxes that touch the edge of the frame, will be rejected
    self.rejectPlates = rejectPlates
    self.minScore = minScore
    self.minChars = minChars
    self.categoryIdx = categoryIdx

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

  # calculate the intersection of the charBox with the plateBox over
  # the area of the charBox
  def intersectionOverArea(self, charBox, plateBox):
    (plateStartY, plateStartX, plateEndY, plateEndX) = plateBox
    (charStartY, charStartX, charEndY, charEndX) = charBox
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(plateStartX, charStartX)
    yA = max(plateStartY, charStartY)
    xB = min(plateEndX, charEndX)
    yB = min(plateEndY, charEndY)

    # if the boxes are intersecting, then compute the area of intersection rectangle
    if xB > xA and yB > yA:
      interArea = (xB - xA) * (yB - yA)
    else:
      interArea = 0.0

    # compute the area of the char box
    charBoxArea = (charEndY - charStartY) * (charEndX - charStartX)

    # compute the intersection area / charBox area
    ioa = interArea / float(charBoxArea)

    # return the intersection over area value
    return ioa

  # Scrub the chars within the plates. Once cleaned, generate
  # lists of char text, char boxes, and and char scores for every plate
  def processPlates(self, plates, plateBoxes, plateScores):
    # Working from left to right, discard any charBox that has an iou > 'charIOUMax'
    # with the box immediately to the left.
    # Loop over the chars, adding chars to charsNoOverLap, if there is no overlap
    licensePlateFound = False
    platesFinal = []
    for plate in plates:
      charsNoOverlap = []
      prevChar = None
      if plate != None:
        for plateChar in plate:
          # First plateChar has no plateChar to left, so add to the list
          if prevChar == None:
            prevChar = plateChar
            charsNoOverlap.append(plateChar)
          # else check for overlap
          else:
            iou = self.intersectionOverUnion(plateChar[1], prevChar[1])
            #print(iou)
            if iou < self.charIOUMax:
              charsNoOverlap.append(plateChar)
              prevChar = plateChar
      #else:
      #  print("Empty plate detected")
      platesFinal.append(charsNoOverlap)

    # Extract the character text, boxes and scores and append to lists
    # Final result is 3 lists containing char text, char boxes and char scores
    # These lists should be the same size as the plateBoxes list
    charTexts = []
    charBoxes = []
    charScores = []
    for plate in platesFinal:
      if len(plate) != 0:
        licensePlateFound = True
        plateArray = np.array(plate, object)
        chars = plateArray[:,2]
        chars = ''.join(chars)
        charTexts.append(chars)
        charBoxes.append(plateArray[:,1])
        charScores.append(plateArray[:,3])
      # else there are no characters in this plate, so append empty lists at this location
      else:
        charTexts.append([])
        charBoxes.append([])
        charScores.append([])

    # Calculate the average score per plate.
    # Generate mask to reject plates with; low number of chars, plates on the edge of the frame
    # and plates with a low average score.
    plateAverageScores = []
    mask = np.ones(len(plateScores), dtype=bool)
    for (i, (plateBox, plateScore, chScores)) in enumerate(zip(plateBoxes, plateScores, charScores)):
      # Calc the average score for plate plus characters inside the plate, and save to plateAverageScores
      averageScore = (plateScore + sum(chScores)) / (len(chScores) + 1)
      plateAverageScores.append(averageScore)
      # set mask to reject bad plates
      if averageScore < self.minScore or len(chScores) <= self.minChars or max(plateBox) >= 0.998 or min(plateBox) <= 0.002:
        mask[i] = False

    # optionally remove bad plates
    if self.rejectPlates == True:
      # update the lists to remove discarded plates
      plateAverageScores = list(np.array(plateAverageScores)[mask,...])
      plateBoxes = list(np.array(plateBoxes)[mask,...])
      charScores = list(np.array(charScores)[mask,...])
      charTexts = list(np.array(charTexts,object)[mask,...])
      charBoxes = list(np.array(charBoxes)[mask,...])

    if (len(plateBoxes) != len(plateAverageScores) or len(plateBoxes) != len(charTexts)):
      print("[ERROR]: len(platesBoxes):{} != len(plateAverageScores):{} or len(platesBoxes):{} != len(charTexts):{}"
            .format(len(plateBoxes), len(plateAverageScores), len(plateBoxes), len(charTexts)))
    if (len(plateBoxes) != len(charBoxes)):
      print("[ERROR]: len(platesBoxes):{} != len(charBoxes):{}"
            .format(len(plateBoxes), len(charBoxes)))
    if licensePlateFound == True and len(plateAverageScores) == 0:
      print("[INFO] license plate found but now rejected")

    return plateBoxes, charTexts, charBoxes, charScores, plateAverageScores

  # Find plate boxes and the text associated with each plate
  def findPlates(self, boxes, scores, labels):
    licensePlateFound = False
    # set mask to all true
    mask = np.ones(len(scores), dtype=bool)

    # Start by discarding all boxes below min score, and moving plate boxes to separate list
    plateBoxes = []
    plateScores = []
    for (i, (box, score, label)) in enumerate(zip(boxes, scores, labels)):
      if score < self.minConfidence:
        mask[i] = False
        continue
      label = self.categoryIdx[label]
      label = "{}".format(label["name"])
      # if label is plate, then append box to plateBoxes list and discard from original lists
      if label == "plate":
        mask[i] = False
        plateBoxes.append(box)
        plateScores.append(score)

    # update the lists to remove discarded boxes
    boxes = boxes[mask,...]
    scores = scores[mask,...]
    labels = labels[mask,...]

    # For each plate box, discard char boxes that are less than 0.5 ioa with plateBox.
    # re-order the remaining boxes by startX
    plates = []
    for plateBox in plateBoxes:
      chars = []
      # loop over the lists: boxes, scores and labels
      # and discard chars that have low ioa with plateBox
      # The boxes scores and labels associated with plates have already been removed
      # so the lists only reference characters
      for (charBox, score, label) in zip(boxes, scores, labels):
        ioa = self.intersectionOverArea(charBox, plateBox)
        if ioa > 0.5:
          label = self.categoryIdx[label]
          label = "{}".format(label["name"])
          char = [charBox[1], charBox, label, score]
          chars.append(char)
      # sort the remaining chars by horizontal location
      chars = sorted(chars, key=lambda x: x[0])
      if len(chars) > 0:
        plates.append(chars)
      else:
        plates.append(None)

    plateBoxes, charTexts, charBoxes, charScores, plateAverageScores = self.processPlates(plates, plateBoxes, plateScores)
    if len(plateBoxes) != 0:
      licensePlateFound = True

    return licensePlateFound, plateBoxes, charTexts, charBoxes, charScores, plateAverageScores

  # Find only plates, and ignore chars
  # return the plateBoxes and plateScores
  def findPlatesOnly(self, boxes, scores, labels):
    licensePlateFound = False

    # Discard all boxes below min score, and move plate boxes to separate list
    plateBoxes = []
    plateScores = []
    # loop over all the boxes and associated scores and labels
    for (i, (box, score, label)) in enumerate(zip(boxes, scores, labels)):
      if score < self.minConfidence:
        continue
      label = self.categoryIdx[label]
      label = "{}".format(label["name"])
      # if label is plate, then append data to new lists
      if label == "plate":
        licensePlateFound = True
        plateBoxes.append(box)
        plateScores.append(score)

    return licensePlateFound, plateBoxes, plateScores

  # Find ground truth plate boxes and the text associated with each plate
  def findGroundTruthPlates(self, boxes, labels):
    labels = [x.decode("ASCII") for x in labels]
    labels = np.array(labels)
    licensePlateFound = False
    # set mask to all true
    mask = np.ones(len(labels), dtype=bool)

    # move plate boxes to separate list
    plateBoxes = []
    for (i, (box, label)) in enumerate(zip(boxes, labels)):
      # if label is plate, then append box to plateBoxes list and discard from original lists
      if label == "plate":
        mask[i] = False
        plateBoxes.append(box)

    # update the lists to remove plate boxes
    boxes = boxes[mask,...]
    labels = labels[mask,...]

    # For each plate box, discard char boxes that are less than 'charPlateIOAMin' ioa with plateBox.
    # re-order the remaining boxes by startX
    plates = []
    for plateBox in plateBoxes:
      chars = []
      for (charBox, label) in zip(boxes, labels):
        ioa = self.intersectionOverArea(charBox, plateBox)
        if ioa > self.charPlateIOAMin:
          char = [charBox[1], charBox, label]
          chars.append(char)
      chars = sorted(chars, key=lambda x: x[0])
      if len(chars) > 0:
        plates.append(chars)
      else:
        plates.append([])


    # Extract the plate text and append to list
    charTexts = []
    charBoxes = []
    for plate in plates:
      if len(plate) != 0:
        licensePlateFound = True
        plateArray = np.array(plate, object)
        chars = plateArray[:,2]
        chars = ''.join(chars)
        charTexts.append(chars)
        charBoxes.append(plateArray[:,1])
      else:
        charTexts.append([])
        charBoxes.append([])

    if (len(plateBoxes) != len(plates) or len(plateBoxes) != len(charTexts)):
      print("[ERROR]: len(platesBoxes):{} !plates= len(plates):{} or len(platesBoxes):{} != len(charText):{}"
            .format(len(plateBoxes), len(plates), len(plateBoxes), len(charTexts)))

    return licensePlateFound, plateBoxes, charTexts, charBoxes

  def findCharsOnly(self, boxes, scores, labels, plateBox, fullImageHeight, fullImageWidth):
    # loop over the boxes, scores and labels
    # Remove the plate boxes and
    # with the remaining char boxes, build a 'chars' list
    # chars entries contain (charBox left co-ordinate, all  4 charBox co-ordinates relative to original image, label, score)
    # Order chars from left to right
    chars = []
    for (charBox, score, label) in zip(boxes, scores, labels):
      if score < self.minConfidence:
        continue
      label = self.categoryIdx[label]
      label = "{}".format(label["name"])
      if label == "plate":
        continue
      # prediction returns a charBox relative to the cropped plateImage
      # convert co-ordinates from fractional to pixels, add the plate box offset
      # and then convert back to fractional
      (pbStartY, pbStartX, pbEndY, pbEndX) = (int(plateBox[0] * fullImageHeight),
                                              int(plateBox[1] * fullImageWidth),
                                              int(plateBox[2] * fullImageHeight),
                                              int(plateBox[3] * fullImageWidth))
      pbHeight = pbEndY - pbStartY
      pbWidth = pbEndX - pbStartX
      charBox = ((charBox[0] * pbHeight + pbStartY) / fullImageHeight,
                 (charBox[1] * pbWidth + pbStartX) / fullImageWidth,
                 (charBox[2] * pbHeight + pbStartY) / fullImageHeight,
                 (charBox[3] * pbWidth + pbStartX) / fullImageWidth)
      char = [charBox[1], charBox, label, score]
      chars.append(char)
    # sort the remaining chars by horizontal location
    chars = sorted(chars, key=lambda x: x[0])
    return chars