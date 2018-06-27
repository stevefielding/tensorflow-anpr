
import numpy as np

class PlateFinder:

  def __init__(self, minConfidence):
    self.minConfidence = minConfidence

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

  # Find plate boxes and the text associated with each plate
  def findPlates(self, boxes, scores, labels, categoryIdx):
    licensePlateFound = False
    # set mask to all true
    mask = np.ones(len(scores), dtype=bool)

    # Start by discarding all boxes below min score, and moving plate boxes to separate list
    plateBoxes = []
    for (i, (box, score, label)) in enumerate(zip(boxes, scores, labels)):
      if score < self.minConfidence:
        mask[i] = False
        continue
      label = categoryIdx[label]
      label = "{}".format(label["name"])
      # if label is plate, then append box to plateBoxes list and discard from original lists
      if label == "plate":
        mask[i] = False
        plateBoxes.append(box)

    # update the lists to remove discarded boxes
    boxes = boxes[mask,...]
    scores = scores[mask,...]
    labels = labels[mask,...]

    # For each plate box, discard char boxes that are less than 0.5 ioa with plateBox.
    # re-order the remaining boxes by startX
    plates = []
    for plateBox in plateBoxes:
      chars = []
      for (charBox, score, label) in zip(boxes, scores, labels):
        ioa = self.intersectionOverArea(charBox, plateBox)
        if ioa > 0.5:
          label = categoryIdx[label]
          label = "{}".format(label["name"])
          char = [charBox[1], charBox, label, score]
          chars.append(char)
      #chars = np.array(sorted(chars, key=lambda x: x[0]))
      chars = sorted(chars, key=lambda x: x[0])
      #chars = chars[:,0]
      #chars = ''.join(chars)
      if len(chars) > 0:
        plates.append(chars)
      else:
        plates.append(None)

    # Working from left to right, discard any charBox that has an iou > 0.5 with the box immediatley to the left
    # Loop over the chars, adding chars to charsNoOverLap, if there is no overlap
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
            if iou < 0.3:
              charsNoOverlap.append(plateChar)
              prevChar = plateChar
      #else:
      #  print("Empty plate detected")
      platesFinal.append(charsNoOverlap)

    # Extract the plate text and append to list
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
      else:
        charTexts.append([])
        charBoxes.append([])
        charScores.append([])

    if (len(plateBoxes) != len(platesFinal) or len(plateBoxes) != len(charTexts)):
      print("[ERROR]: len(platesBoxes):{} != len(platesFinal):{} or len(platesBoxes):{} != len(charText):{}"
            .format(len(plateBoxes), len(platesFinal), len(plateBoxes), len(charTexts)))
    #if licensePlateFound == False:
    #  print("[INFO] No license plate found")

    return licensePlateFound, plateBoxes, charTexts, charBoxes, charScores

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

    # For each plate box, discard char boxes that are less than 0.5 ioa with plateBox.
    # re-order the remaining boxes by startX
    plates = []
    for plateBox in plateBoxes:
      chars = []
      for (charBox, label) in zip(boxes, labels):
        ioa = self.intersectionOverArea(charBox, plateBox)
        if ioa > 0.5:
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
      print("[ERROR]: len(platesBoxes):{} != len(plates):{} or len(platesBoxes):{} != len(charText):{}"
            .format(len(plateBoxes), len(plates), len(plateBoxes), len(charTexts)))

    return licensePlateFound, plateBoxes, charTexts, charBoxes
