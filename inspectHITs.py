# -------------------- inspectHITs.py -------------------------
# python inspectHITs.py --csvFileIn ../../datasets/lplates_smallset/hand_labelled/images_labelled/labels.txt  \
# --csvFileOut ../../datasets/lplates_smallset/hand_labelled/images_labelled/labelsNew.txt \
# --imagePath ../../datasets/lplates_smallset/hand_labelled/images_labelled/img

# Read a label.txt file and extract the contents to a dictionary where the keys are the filenames
# and the values are the plate text.
# Now display the images with an embedded text box containing the plate text that corresponds with the
# filename key. The user has the opportunity to view the plate text in the image and compare to the plate text
# retrieved from the lables.txt file. The text can be left unmodified or edited. Either way the plate text and
# filename are written to the new output label file.

from imutils import paths
from PIL import Image, ImageDraw, ImageFont
import tkinter
from PIL import ImageTk
import argparse
import numpy as np
import os
import sys
import csv
import re

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--csvFileIn", required=True,
  help="path to labels input file")
ap.add_argument("-o", "--csvFileOut", required=True,
  help="path to labels output file")
ap.add_argument("-i", "--imagePath", required=True,
  help="path to image files")
args = vars(ap.parse_args())

# check arguements
if os.path.exists(args["csvFileIn"]) == False:
  print("[ERROR]: --csvFileIn \"{}\" does not exist".format(args["csvFileIn"]))
  sys.exit()
if os.path.exists(args["imagePath"]) == False:
  print("[ERROR]: --imagePath \"{}\" does not exist".format(args["imagePath"]))
  sys.exit()
if os.path.exists(args["csvFileOut"]) == True:
  print("[ERROR]: --csvFileOut \"{}\" already exists. Delete first".format(args["csvFileOut"]))
  #sys.exit()

# read the csv file and copy to dictionary
csvFileIn = open(args["csvFileIn"], "r")
csvReader = csv.DictReader(csvFileIn)

#print("[INFO] Number of HITs in the file: {}".format(len(csvSplit) - 1))

# get list of all input image files,
# and open the labels output file in write mode.
myPaths = paths.list_files(args["imagePath"], validExts=(".jpg"))

# return key event handler
def return_key_exit_mainloop (event):
  event.widget.quit() # this will cause mainloop to unblock.

# configure the main window, and bind the return key
root = tkinter.Tk()
root.geometry('+%d+%d' % (100,100))
root.bind('<Return>', return_key_exit_mainloop)

# open the csv output file as a dictionary
csvWriteFile = open(args["csvFileOut"], 'w', newline='')
csvWriter = csv.DictWriter(csvWriteFile, fieldnames=csvReader.fieldnames)
csvWriter.writeheader()


# For every file, add a new line of text to the comma delimited label file
# Each line will contain the old fileName, new file name, and license plate characters
for hitDict in csvReader:
  imagePath = os.path.join("SJ7STAR_images" , hitDict["Input.image_url"])

  # Read the image
  image = Image.open(imagePath)
  draw = ImageDraw.Draw(image)

  #find the boxes, and draw them on the image
  annotations = hitDict["Answer.annotation_data"]
  matches = re.finditer(r"\{(.*?)\}", annotations)
  font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 30)
  # if BDV part number found in line, then convert to Steris part number
  for matchNum, match in enumerate(matches):
    locDict = dict()
    match = match.group()
    annSplit = match.split(",")
    for loc in annSplit:
      locSplit = loc.split(":")
      locDict[locSplit[0].strip("\"{}")] = locSplit[1].strip("\"{}")
    top = int(locDict["top"])
    left = int(locDict["left"])
    bottom = top + int(locDict["height"])
    right = left + int(locDict["width"])
    #draw.rectangle((top,right,bottom,left), outline=128, fill=None)
    draw.rectangle((left, top, right, bottom), outline=128, fill=None)
    draw.text((left, max(0,top-30)), locDict["label"].upper(), font=font, fill="red")
  del draw

  # resize image
  basewidth = 1000
  wpercent = (basewidth / float(image.size[0]))
  hsize = int((float(image.size[1]) * float(wpercent)))
  image = image.resize((basewidth, hsize), Image.ANTIALIAS)

  # Add image, prompt text, and text Entry box to window, default text is from labels file
  root.geometry('%dx%d' % (image.size[0],image.size[1]))
  tkpi = ImageTk.PhotoImage(image)
  label_image = tkinter.Label(root, image=tkpi)
  label_image.place(x=0,y=0,width=image.size[0],height=image.size[1])
  label = tkinter.Label(root, text="Enter x to accept, or a reason to reject")
  label.pack()
  e = tkinter.Entry(root)
  fileName = imagePath.split('/')[-1]
  e.insert(0, "x")
  e.pack()
  e.focus_set()
  root.title(imagePath)
  root.mainloop() # wait until user presses 'return'

  # get the accept/reject response, and write to the output csv file
  acceptReject = e.get()
  if acceptReject == 'x' or acceptReject == 'X':
    hitDict["Approve"] = "x"
  else:
    hitDict["Reject"] = acceptReject
  csvWriter.writerow(hitDict)

  # Finished with the window, destroy
  label_image.destroy()
  e.destroy()
  label.destroy()
  print(acceptReject)

csvWriteFile.close()



