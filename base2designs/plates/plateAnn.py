
import os

class PlateAnn():

  def scaleBB(self, box, W, H):
    # scale the bounding box from the range [0, 1] to [W, H]
    (startY, startX, endY, endX) = box
    startX = int(startX * W)
    startY = int(startY * H)
    endX = int(endX * W)
    endY = int(endY * H)
    return startX, startY, endX, endY

  def xmlStart(self, imagePath, imageHeight, imageWidth, imageDepth):
    imageFolder = imagePath.split(os.sep)[-2]
    imageFilename = imagePath.split(os.sep)[-1]

    pascal_voc_start = ("<annotation>\n"
      "	<folder>" + imageFolder + "</folder>\n"
      "	<filename>" + imageFilename + "</filename>\n"
      "	<path>" + imagePath + "</path>\n"
      "	<source>\n"
      "		<database>Unknown</database>\n"
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

  def writeAnnFile(self, xmlPath, imagePath, plateBox, plateText, charBoxes, imageWidth, imageHeight, imageDepth):

    # create the xml file, and write the preamble
    xmlFile = open(xmlPath, "w")
    xmlFile.write(self.xmlStart(imagePath, imageWidth, imageHeight, imageDepth))

    # add the plateBox to the xml file
    left, top, right, bottom = self.scaleBB(plateBox, imageWidth, imageHeight)
    xmlFile.write(self.xmlBox("plate", left, top, right, bottom))

    # add the plate char boxes to the xml file
    for (char, charBox) in zip(plateText, charBoxes):
      # write the box info to file
      left, top, right, bottom = self.scaleBB(charBox, imageWidth, imageHeight)
      xmlFile.write(self.xmlBox(char.lower(), left, top, right, bottom))

    # write final text to xml file and close
    xmlFile.write(self.xmlEnd())
    xmlFile.close()
    #print("[INFO] Created: \"{}\"".format(xmlPath))








