
import os

class FolderControl:

	def __init__(self):
		self.currentFileNamePrefix = None

	def createDestFolders(self, fileNamePrefix, saveImagePath, outputImagePath, croppedImagePath, outputVideoPath):
		# if new day then create new directory
		if (fileNamePrefix != self.currentFileNamePrefix):
			saveImageDir = "{}/{}".format(saveImagePath, fileNamePrefix)
			print ("New day: {}".format(fileNamePrefix))
			if (os.path.isdir(saveImageDir) != True):
				os.makedirs(saveImageDir)
			detectedImageDir = "{}/{}" .format(outputImagePath, fileNamePrefix)
			if (os.path.isdir(detectedImageDir) != True):
				os.makedirs(detectedImageDir)
			croppedImageDir = "{}/{}" .format(croppedImagePath, fileNamePrefix)
			if (os.path.isdir(croppedImageDir) != True):
				os.makedirs(croppedImageDir)
			outputVideoDir = "{}/{}" .format(outputVideoPath, fileNamePrefix)
			if (os.path.isdir(outputVideoDir) != True):
				os.makedirs(outputVideoDir)
			self.currentFileNamePrefix = fileNamePrefix


