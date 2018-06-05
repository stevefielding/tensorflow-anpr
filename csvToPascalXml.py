# -------------------- csvToPascalXml.py -------------------------
# python csvToPascalXml.py --csvFileIn Batch_3213647_batch_results_second_submission_checked.csv \
# --xmlPath SJ7STAR_images/2018_03_02_ann

from imutils import paths
from PIL import Image, ImageDraw, ImageFont
import argparse
import numpy as np
import os
import sys
import csv
import re
import cv2

def xmlStart(imagePath):
  imageFolder = imagePath.split(os.sep)[-2]
  imageFilename = imagePath.split(os.sep)[-1]
  image = cv2.imread(imagePath)
  (imageHeight, imageWidth, imageDepth) = image.shape

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

def xmlBox(objName, xmin, ymin, xmax, ymax):
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

def xmlEnd():
  pascal_voc_end = "</annotation>\n"
  return pascal_voc_end

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--csvFileIn", required=True,
  help="path to labels input file")
ap.add_argument("-o", "--xmlPath", required=True,
  help="path to labels output file")
args = vars(ap.parse_args())

# check arguements
if os.path.exists(args["csvFileIn"]) == False:
  print("[ERROR]: --csvFileIn \"{}\" does not exist".format(args["csvFileIn"]))
  sys.exit()
if os.path.exists(args["xmlPath"]) == False:
  print("[ERROR]: --xmlPath \"{}\" does not exist".format(args["xmlPath"]))
  sys.exit()

# read the csv file and copy to dictionary
csvFileIn = open(args["csvFileIn"], "r")
csvReader = csv.DictReader(csvFileIn)

# Loop over the lines in the csv file. Each line read as a dictionary
rejectCnt = 0
acceptCnt = 0
for hitDict in csvReader:
  imagePath = os.path.join("SJ7STAR_images" , hitDict["Input.image_url"])
  imageFilename = os.path.split(imagePath) [-1]
  # if annotation not approved then skip to the next file
  if hitDict["AssignmentStatus"] == "Rejected" or (hitDict["AssignmentStatus"] == "Submitted" and hitDict["Approve"] == ""):
    print("Skipping: \"{}\"".format(imageFilename))
    rejectCnt += 1
    continue

  acceptCnt += 1
  # open the xml output file for writing
  fnMatch = re.search(r"(.*)\..*$", imageFilename)
  xmlPath = args["xmlPath"] + os.sep + fnMatch.group(1) + ".xml"
  xmlFile = open(xmlPath, "w")
  xmlFile.write(xmlStart(imagePath))

  #find the boxes
  annotations = hitDict["Answer.annotation_data"]
  matches = re.finditer(r"\{(.*?)\}", annotations)
  # loop over all the boxes
  for matchNum, match in enumerate(matches):
    locDict = dict()
    match = match.group()
    annSplit = match.split(",")
    # loop over the box location data
    for loc in annSplit:
      locSplit = loc.split(":")
      locDict[locSplit[0].strip("\"{}")] = locSplit[1].strip("\"{}")
    top = int(locDict["top"])
    left = int(locDict["left"])
    bottom = top + int(locDict["height"])
    right = left + int(locDict["width"])

    # write the box info to file
    xmlFile.write(xmlBox(locDict["label"].lower(), left, top, right, bottom))

  # write final text to xml file and close
  xmlFile.write(xmlEnd())
  xmlFile.close()
  print("Created: \"{}\"".format(imageFilename))

print ("Accepted: {} annotations, rejected: {} annotations".format(acceptCnt, rejectCnt))







