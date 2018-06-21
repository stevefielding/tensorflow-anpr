
import cv2

class VideoWriter():


  def __init__(self, outputPath, W, H):
    self.W = W
    self.H = H
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # save annotated video to "output_video_path". Use the same file prefix, but change the suffix to mp4
    self.writer = cv2.VideoWriter(outputPath, fourcc, 20, (self.W, self.H), True)

  def scaleBB(self, box):
    # scale the bounding box from the range [0, 1] to [W, H]
    (startY, startX, endY, endX) = box
    startX = int(startX * self.W)
    startY = int(startY * self.H)
    endX = int(endX * self.W)
    endY = int(endY * self.H)
    return startX, startY, endX, endY

  def writeFrame(self, frame, plateBoxes, charTexts, charBoxes, charScores):
    for (plateBox, chText, chBoxes, chScores) in zip(plateBoxes, charTexts, charBoxes, charScores):
      # display the plate box and plate text
      if len(plateBox) != 0:
        startX, startY, endX, endY = self.scaleBB(plateBox)
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        if len(chText) != 0:
          cv2.putText(frame, chText, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

      # if there is char boxes inside the plate box
      # if chBoxes != None:
      # display the char boxes
      for chBox in chBoxes:
        if len(chBox) != 0:
          startX, startY, endX, endY = self.scaleBB(chBox)
          cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # write the frame to the outputImage file
    self.writer.write(frame)

  def closeWriter(self):
    self.writer.release()