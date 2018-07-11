
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

# initialize the colors list and the model
#COLORS = [(0, 255, 0), (0, 0, 255)]
COLORS = np.random.randint(0,256, size=37*3)
COLORS = COLORS.reshape((37,3))
#COLORS = [tuple(x) for x in COLORS]
COLORS2 = []
for oldTuple in COLORS:
  newTuple = (int(oldTuple[0]), int(oldTuple[1]), int(oldTuple[2]))
  COLORS2.append(newTuple)
COLORS = COLORS2

# if the video frames are larger than this dimension they
# will be resized, whilst retaining the same aspect ratio
MAX_VID_DIM = 2000

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
plateFinder = PlateFinder(conf["min_confidence"])
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
                                     conf["output_image_path"], conf["output_cropped_image_path"], conf["output_video_path"])

  # create a session to perform inference
  with model.as_default():
    with tf.Session(graph=model) as sess:
      # initialize the points to the video files
      stream = cv2.VideoCapture(videoPath)
      videoWriter = None

      # Prepare findFrameWithPlate for a new video sequence
      plateLogFlag = False
      firstPlateFound = False

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
          grabbed, image = stream.retrieve()  # retrieve the already grabbed frame

          # grab a reference to the input image tensor and the
          # boxes
          imageTensor = model.get_tensor_by_name("image_tensor:0")
          boxesTensor = model.get_tensor_by_name("detection_boxes:0")

          # for each bounding box we would like to know the score
          # (i.e., probability) and class label
          scoresTensor = model.get_tensor_by_name("detection_scores:0")
          classesTensor = model.get_tensor_by_name("detection_classes:0")
          numDetections = model.get_tensor_by_name("num_detections:0")

          # grab the image dimensions
          (H, W) = image.shape[:2]

          # check to see if we should resize along the width
          if W > H and W > MAX_VID_DIM:
            image = imutils.resize(image, width=MAX_VID_DIM)

          # otherwise, check to see if we should resize along the
          # height
          elif H > W and H > MAX_VID_DIM:
            image = imutils.resize(image, height=MAX_VID_DIM)

          # get the new dimensions for the resized image
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

          # Convert image into format expected by tensorflow, ie RGB and extra dimension to represent the batch axis
          tfImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
          tfImage = np.expand_dims(tfImage, axis=0)

          # perform inference and compute the bounding boxes,
          # probabilities, and class labels
          (boxes, scores, labels, N) = sess.run(
            [boxesTensor, scoresTensor, classesTensor, numDetections],
            feed_dict={imageTensor: tfImage})

          # squeeze the lists into a single dimension
          boxes = np.squeeze(boxes)
          scores = np.squeeze(scores)
          labels = np.squeeze(labels)

          # write annotated video if option is selected
          if conf["saveAnnotatedVideo"] == "true":
            # write the frame plus annotation to the video stream
            videoWriter.writeFrame(videoImage, plateBoxes, charTexts, charBoxes, charScores)

          # find the plates, and find the chars within the plates
          licensePlateFound, plateBoxes, charTexts, charBoxes, charScores = plateFinder.findPlates(boxes, scores, labels, categoryIdx)

          # if license plates have been found, then predict the plate text, and add to the history
          if licensePlateFound == True:
            plateHistory.addPlatesToHistory(charTexts, charBoxes, plateBoxes, image, videoPath, frameCount)
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
      frameCountDelta = frameCount - oldFrameCount
      fps = frameCountDelta / processingTime
      oldFrameCount = frameCount
      print(
        "[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCountDelta, processingTime,
                                                                                     fps))
      print("[INFO] validImages: {}, frameCount: {}, loggedPlateCount: {}".format(validImages, frameCount, loggedPlateCount))
