# USAGE
# $ python predict_video.py --conf conf/lplates_smallset.json

import argparse
import os
import re
import shutil
import sys
import time

import cv2
import imutils
import numpy as np
import tensorflow as tf
from imutils import paths
# import the necessary packages
from object_detection.utils import label_map_util

from base2designs.plates.plateFinder import PlateFinder
from base2designs.plates.plateHistory import PlateHistory
from base2designs.utils.conf import Conf
from base2designs.utils.folderControl import FolderControl
from base2designs.utils.videoWriter import VideoWriter
from base2designs.plates.predicter import Predicter

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the json config file")
args = vars(ap.parse_args())

# Check if config file exists
if os.path.exists(args["conf"]) == False:
  print("[ERROR]: --conf \"{}\" does not exist".format(args["conf"]))
  sys.exit()

# Read the json config
conf = Conf(args["conf"])

reject_poor_quality_plates=conf["reject_poor_quality_plates"]=="true"
print("Detecting objects using model: {}, and {} stage(s) of prediction".format(conf["model"], conf["pred_stages"]))
print("reject_poor_quality_plates: {}".format(reject_poor_quality_plates))

# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
  # initialize the graph definition
  graphDef = tf.GraphDef()

  # load the graph from disk
  with tf.gfile.GFile(conf["model"], "rb") as f:
    serializedGraph = f.read()
    graphDef.ParseFromString(serializedGraph)
    tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(conf["labels"])
categories = label_map_util.convert_label_map_to_categories(
  labelMap, max_num_classes=conf["num_classes"],
  use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# open the logFile, create if it does not exist, otherwise open in append mode
logFilePath = "{}/{}" .format(conf["output_image_path"], conf["log_file_name"])
if (os.path.isdir(conf["output_image_path"]) != True):
  os.makedirs(conf["output_image_path"])
if os.path.exists(logFilePath) == False:
  logFile = open(logFilePath, "w")
else:
  logFile = open(logFilePath, "a")

# create a plateFinder and load the plate history utility
plateFinder = PlateFinder(conf["min_confidence"], categoryIdx,
                          rejectPlates=reject_poor_quality_plates, charIOUMax=conf["charIOUMax"])
folderController = FolderControl()
plateHistory = PlateHistory(conf["output_image_path"], logFile,
                            saveAnnotatedImage=conf["saveAnnotatedImage"] == "true")
quit = False
plateLogLatency = conf["plateLogLatency"]* conf["videoFrameRate"]
platesReadyForLog = False
myPaths = paths.list_files(conf["input_video_path"], validExts=(".h264", "mp4", "mov", "ts"))


# loop over all the detected video files
for videoPath in sorted(myPaths):
  print("[INFO] reading video file {}...".format(videoPath))
  start_time = time.time()
  frameCount = 0
  oldFrameCount = 0
  validImages = 0
  loggedPlateCount = 0
  frameCntForPlateLog = 0
  frameDecCnt = 1
  m = re.search(r"([0-9]{4}[-_][0-9]{2}[-_][0-9]{2})", videoPath)
  if m:
    destFolderRootName = m.group(1)  # assumes video stored in sub-directory
  else:
    destFolderRootName = "YYYY-MM-DD"
  folderController.createDestFolders(destFolderRootName, conf["save_video_path"],
                                     conf["output_image_path"], conf["output_video_path"])

  # create a session to perform inference
  with model.as_default():
    with tf.Session(graph=model) as sess:
      # initialize the points to the video files
      stream = cv2.VideoCapture(videoPath)
      videoWriter = None

      # Prepare findFrameWithPlate for a new video sequence
      plateLogFlag = False
      firstPlateFound = False

      # create a predicter, used to predict plates and chars
      predicter = Predicter(model, sess, categoryIdx)

      # loop over frames from the video file stream
      while True:
        # grab the next frame
        #(grabbed, image) = stream.read()
        # read the next frame from the video stream
        grabbed = stream.grab()  # grab frame but do not decode

        # We have reached the end of the video clip. Save any residual plates to log
        # Remove all the plate history
        if not grabbed:
          if platesReadyForLog == True:
            plateDictBest = plateHistory.selectTheBestPlates()
            # generate output files, ie cropped Images, full image and log file
            plateHistory.writeToFile(plateDictBest, destFolderRootName, W, H, D)
            loggedPlateCount += len(plateDictBest)
          plateHistory.clearHistory()
          firstPlateFound = False

          # move video clip from input directory to saveVideoDir
          outputPathSaveOriginalImage = conf["save_video_path"] + "/" + destFolderRootName + videoPath[videoPath.rfind("/") + 0:]
          if (conf["move_video_file"] == "true"):
            try:
              # os.rename(videoPath, outputPathSaveOriginalImage) #does not work between two different file systems
              shutil.move(videoPath, outputPathSaveOriginalImage)
            except OSError as e:
              print("OS error({0}): {1}".format(e.errno, e.strerror))
              sys.exit(1)
          break

        # Decimate the frames
        frameCount += 1
        if firstPlateFound == True:
          frameCntForPlateLog += 1
        if frameCntForPlateLog > plateLogLatency:
          plateLogFlag = True
          frameCntForPlateLog = 0
        if (frameDecCnt == 1):
          # retrieve the already grabbed frame, and get the dimensions
          grabbed, image = stream.retrieve()
          (H, W, D) = image.shape[:3]

          # if the video writer is None, initialize it
          # initializing in the middle of the loop, because we don't know
          # W, H, and D until the first video frame is read
          if conf["saveAnnotatedVideo"] == "true":
            if videoWriter is None:
              # save annotated video to "output_video_path". Use the same file prefix, but change the suffix to mp4
              outputPathVideo = conf["output_video_path"] + "/" + destFolderRootName + videoPath[
                                                                                       videoPath.rfind("/") + 0:]
              outputPathVideo = outputPathVideo[0: outputPathVideo.rfind(".")] + ".mp4"
              videoWriter = VideoWriter(outputPathVideo, W, H)
            # create a copy of image
            videoImage = image.copy()

          # If prediction stages == 2, then perform prediction on full image, find the plates, crop the plates from the image,
          # and then perform prediction on the plate images
          if conf["pred_stages"] == 2:
            # Perform inference on the full image, and then select only the plate boxes
            boxes, scores, labels = predicter.predictPlates(image)
            licensePlateFound, plateBoxes, plateScores = plateFinder.findPlatesOnly(boxes, scores, labels)

            # loop over the plate boxes, find the chars inside the plate boxes,
            # and then scrub the chars with 'processPlates', resulting in a list of final plateBoxes, char texts, char boxes, char scores and complete plate scores
            plates = []
            for plateBox in plateBoxes:
              boxes, scores, labels = predicter.predictChars(image, plateBox, conf["min_confidence"])
              chars = plateFinder.findCharsOnly(boxes, scores, labels, plateBox, image.shape[0], image.shape[1])
              if len(chars) > 0:
                plates.append(chars)
              else:
                plates.append(None)
            plateBoxes, charTexts, charBoxes, charScores, plateAverageScores = plateFinder.processPlates(
              plates, plateBoxes, plateScores)
            if len(plateAverageScores) == 0:
              licensePlateFound = False
            else:
              licensePlateFound = True

          # If prediction stages == 1, then predict the plates and characters in one pass
          elif conf["pred_stages"] == 1:
            # Perform inference on the full image, and then find the plate text associated with each plate
            boxes, scores, labels = predicter.predictPlates(image, preprocess=False)
            licensePlateFound, plateBoxes, charTexts, charBoxes, charScores, plateAverageScores = plateFinder.findPlates(
              boxes, scores, labels)
          else:
            print("[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(
              conf["pred_stages"]))
            quit()

          # write annotated video if option is selected
          if conf["saveAnnotatedVideo"] == "true":
            # write the frame plus annotation to the video stream
            videoWriter.writeFrame(videoImage, plateBoxes, charTexts, charBoxes, charScores)

          # if license plates have been found, then predict the plate text, and add to the history
          if licensePlateFound == True:
            plateHistory.addPlatesToHistory(charTexts, charBoxes, plateBoxes, image, videoPath, frameCount, plateAverageScores)
            validImages += 1
            firstPlateFound = True
            platesReadyForLog = True

          # if sufficient time has passed since the last log, then
          # get a dictionary of the best de-duplicated plates,
          # and remove old plates from history, then save images and update the log
          if plateLogFlag == True:
            platesReadyForLog = False
            plateLogFlag = False
            plateDictBest = plateHistory.selectTheBestPlates()
            # generate output files, ie cropped Images, full image and log file
            plateHistory.writeToFile(plateDictBest, destFolderRootName, W, H, D)
            plateHistory.removeOldPlatesFromHistory()
            loggedPlateCount += len(plateDictBest)

        if (frameDecCnt == conf["frameDecimationFactor"]):
          frameDecCnt = 1
        else:
          frameDecCnt += 1

      # close the video writer and release the stream
      if conf["saveAnnotatedVideo"] == "true":
        videoWriter.closeWriter()
        stream.release()

      # print some performance statistics
      curTime = time.time()
      processingTime = curTime - start_time
      frameCountDelta = (frameCount - oldFrameCount) / conf["frameDecimationFactor"]
      fps = frameCountDelta / processingTime
      oldFrameCount = frameCount
      print(
        "[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCountDelta, processingTime,
                                                                                     fps))
      print("[INFO] validImages: {}, frameCount: {}, loggedPlateCount: {}".format(validImages, frameCount, loggedPlateCount))
