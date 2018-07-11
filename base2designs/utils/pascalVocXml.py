import cv2
import os

class PascalVocXml():

  def xmlStart(self, imagePath):
    imageFolder = imagePath.split(os.sep)[-2]
    imageFilename = imagePath.split(os.sep)[-1]
    image = cv2.imread(imagePath)
    (imageHeight, imageWidth, imageDepth) = image.shape

    pascal_voc_start = ("<annotation verified=\"yes\">\n"
                        "	<folder>" + imageFolder + "</folder>\n"
                        "	<filename>" + imageFilename + "</filename>\n"
                        "	<path>" + imagePath + "</path>\n"
                        "	<source>\n"
                        "	  <database>Unknown</database>\n"
                        "	</source>\n"
                        "	<size>\n"
                        "		<width>" + str(imageWidth) + "</width>\n"
                        "		<height>" + str(imageHeight) + "</height>\n"
                        "		<depth>" + str(imageDepth) + "</depth>\n"
                        "	</size>\n"
                        "	<segmented>0</segmented>\n")
    return pascal_voc_start

  def xmlBox(self, objName, xmin, ymin, xmax, ymax):
    pascal_voc_object = ("	<object>\n"
                         "		<name>" + objName + "</name>\n"
                         "		<pose>Unspecified</pose>\n"
                         "		<truncated>0</truncated>\n"
                         "		<difficult>0</difficult>\n"
                         "		<bndbox>\n"
                         "			<xmin>" + str(xmin) + "</xmin>\n"
                         "			<ymin>" + str(ymin) + "</ymin>\n"
                         "			<xmax>" + str(xmax) + "</xmax>\n"
                         "			<ymax>" + str(ymax) + "</ymax>\n"
                         "		</bndbox>\n"
                         "	</object>\n")
    return pascal_voc_object

  def xmlEnd(self):
    pascal_voc_end = "</annotation>\n"
    return pascal_voc_end
