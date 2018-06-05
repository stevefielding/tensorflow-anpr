# -------------------- inspectHITs.py -------------------------
# python inspectHITs.py --csvFileIn Batch_3211133_batch_results.csv \
# --imagePathRoot SJ7STAR_images --csvFileOut batch_results_checked.csv

# Use this application to review a downloaded MTurk csv results file
# and generate a a modified csv results file with the accept and reject entries
# filled in. An 'x' in the accept column accepts the HIT, or any text in the  reject
# column rejects the HIT
# Loops over all the HITs, diplaying the images, bounding boxes and box labels
# Displays a text box where the response can be entered. If the entry is 'x'
# this is placed in the accept column, any other text is placed in the reject column
# Once complete, upload the new csv file to MTurk

# ---- Format of downloaded csv file ---
# The csv file that is retrieved from the first submission has "AssignmentStatus"
# assigned "Submitted"
# Every entry will have to be checked and "Approve" and "Reject" fields assgned appropriately
# When the csv file is returned from the second submission, it gets a more complicated
# The "Reject" assignment is moved to "RequesterFeedback"
# There will be two entries for each "Input.image_url". One with "AssignmentStatus" equal
# to "Rejected" or "Accepted", and one with "AssignmentStatus" equal to "Submitted"

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
ap.add_argument("-i", "--imagePathRoot", required=True,
  help="path to image files")
args = vars(ap.parse_args())

# check arguements
if os.path.exists(args["csvFileIn"]) == False:
  print("[ERROR]: --csvFileIn \"{}\" does not exist".format(args["csvFileIn"]))
  sys.exit()
if os.path.exists(args["imagePathRoot"]) == False:
  print("[ERROR]: --imagePathRoot \"{}\" does not exist".format(args["imagePathRoot"]))
  sys.exit()

# read the csv file and copy to dictionary
csvFileIn = open(args["csvFileIn"], "r")
csvReader = csv.DictReader(csvFileIn)

#print("[INFO] Number of HITs in the file: {}".format(len(csvSplit) - 1))

# return key event handler
def return_key_exit_mainloop (event):
  event.widget.quit() # this will cause mainloop to unblock.

# configure the main window, and bind the return key
root = tkinter.Tk()
root.geometry('+%d+%d' % (100,100))
root.bind('<Return>', return_key_exit_mainloop)

# Process the csv output file
csvImagesChecked = dict()
# If the csv output file already exists, then open the file and read all the image file names,
# add them to a set, and then close the csv file
if os.path.exists(args["csvFileOut"]) == True:
  print("[INFO] csv output file already exists. Appending data")
  csvFileOutExists = True
  csvFileOut = open(args["csvFileOut"], "r")
  csvOutReader = csv.DictReader(csvFileOut)
  # Loop over all entries in the csv output, and for each "Input.image_url"
  # add the "AssignmentStatus"
  for hitDict in csvOutReader:
    if hitDict["Input.image_url"] in csvImagesChecked:
      csvImagesChecked[hitDict["Input.image_url"]].append(hitDict["AssignmentStatus"])
    else:
      csvImagesChecked[hitDict["Input.image_url"]] = [hitDict["AssignmentStatus"]]
  csvFileOut.close()
# else the csv output file does not exist. Set a flag 'csvFileOutExists'
else:
  print("[INFO] csv output file not found. Creating new file")
  csvFileOutExists = False
# open the csv output file in append mode. If it does not exist, create it, and add header
csvWriteFile = open(args["csvFileOut"], 'a+', newline='')
csvWriter = csv.DictWriter(csvWriteFile, fieldnames=csvReader.fieldnames)
if csvFileOutExists == False:
  csvWriter.writeheader()

# Loop over the lines in the csv file. Each line read as a dictionary
hitProcessedAlready = False
for hitDict in csvReader:
  # if HIT has already been copied to csv output file, then break from the loop
  localImagePath = hitDict["Input.image_url"]
  hitAssignmentStatus = csvImagesChecked.get(localImagePath)
  for status in hitAssignmentStatus:
    if status == hitDict["AssignmentStatus"]:
      hitProcessedAlready = True
      continue
  if hitProcessedAlready == True:
    continue
  # if HIT "AssignmentStatus" is not "Submitted", then break from loop
  if hitDict["AssignmentStatus"] != "Submitted" or hitProcessedAlready == True:
    csvWriter.writerow(hitDict)
    continue

  # Read the image
  imagePath = os.path.join(args["imagePathRoot"], hitDict["Input.image_url"])
  image = Image.open(imagePath)
  draw = ImageDraw.Draw(image)

  #find the boxes, and draw them on the image
  annotations = hitDict["Answer.annotation_data"]
  matches = re.finditer(r"\{(.*?)\}", annotations)
  font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 30)
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
  csvWriteFile.flush()

  # Finished with the window, destroy
  label_image.destroy()
  e.destroy()
  label.destroy()
  print(acceptReject)

csvWriteFile.close()



