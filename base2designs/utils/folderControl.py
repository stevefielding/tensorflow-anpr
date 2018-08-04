
import os

class FolderControl:

  def __init__(self):
    self.currentFileNamePrefix = None

  def createDestFolders(self, fileNamePrefix, saveVideoPath, outputImagePath, outputVideoPath):
    # if new day then create new directory
    if (fileNamePrefix != self.currentFileNamePrefix):
      saveVideoDir = "{}/{}".format(saveVideoPath, fileNamePrefix)
      print ("New day: {}".format(fileNamePrefix))

      # directory to save original video clips
      if (os.path.isdir(saveVideoDir) != True):
        os.makedirs(saveVideoDir)

      # directory for full size images extracted from the video
      outputImageDir = "{}/{}" .format(outputImagePath, fileNamePrefix)
      if (os.path.isdir(outputImageDir) != True):
        os.makedirs(outputImageDir)

      # directory to store image annotation files
      annDir = "{}_ann" .format(outputImageDir)
      if (os.path.isdir(annDir) != True):
        os.makedirs(annDir)

      # dir for annotated video
      outputVideoDir = "{}/{}" .format(outputVideoPath, fileNamePrefix)
      if (os.path.isdir(outputVideoDir) != True):
        os.makedirs(outputVideoDir)

      self.currentFileNamePrefix = fileNamePrefix


