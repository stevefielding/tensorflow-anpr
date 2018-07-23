# USAGE
# python predict.py --model datasets/experiment_faster_rcnn/2018_06_12/exported_model/frozen_inference_graph.pb \
# --labels datasets/records/classes.pbtxt --annotations_dir images/C920_images/2018_06_14_ann --num-classes 37 \
# --image_display False
# Scans the annotations_dir and looks for PASCAL_VOC annotations with verified=yes, reads all the associated image
# files, and performs inference. Predicted results are filtered by plateFinder, which discards char boxes that are
# outside a platebox, discards char boxes that overlap each other, and then orders the chars within each plate box
# from left to right.
# Predicted labels and bounding boxes are compared to the ground truths, and the
# labelled image is optionally displayed.
# Scores are displayed for plates and chars.
# platesWithCharCorrect - Plates detected in correct location and containing correct characters in the correct locations
# platesCorrect - Plates in correct location, but no checking of characters
# platesIncorrect - Plates detected outside of correct location. Calculated as a percentage, but bear in mind that
#                   the plates outside the correct location is unbounded, so plateCorrect+platesIncorrect may not add
#                   up to 100%
# charsCorrect - Characters detected in the correct place with the correct contents
# charsIncorrect - Chars detected outside of the correct location, or the location is correct,
#                  but the contents are wrong. Calculated as a percentage, but bear in mind that
#                  the number of characters outside the correct location is unbounded, so charsCorrect+charsIncorrect
#                  may not add up to 100%

# import the necessary packages
import argparse

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from object_detection.utils import label_map_util
from base2designs.plates.plateFinder import PlateFinder
from base2designs.plates.plateXmlExtract import PlateXmlExtract
from base2designs.plates.plateDisplay import PlateDisplay
from base2designs.plates.plateCompare import PlateCompare
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
  help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
  help="labels file")
ap.add_argument("-a", "--annotations_dir", required=True,
  help="path to annotations dir")
ap.add_argument("-n", "--num-classes", type=int, required=True,
  help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
  help="minimum probability used to filter weak detections")
ap.add_argument("-d", "--image_display", type=bool, default="false",
  help="Enable display of ground truth and predicted annotated images")

args = vars(ap.parse_args())


# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
  # initialize the graph definition
  graphDef = tf.GraphDef()

  # load the graph from disk
  with tf.gfile.GFile(args["model"], "rb") as f:
    serializedGraph = f.read()
    graphDef.ParseFromString(serializedGraph)
    tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(args["labels"])
categories = label_map_util.convert_label_map_to_categories(
  labelMap, max_num_classes=args["num_classes"],
  use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# create a plateFinder
plateFinder = PlateFinder(args["min_confidence"], rejectPlates=True)

# create a plate xml extractor
plateXmlExtract = PlateXmlExtract(args["labels"])

# create a plate comparator
plateCompare = PlateCompare()

# create plate displayer
plateDisplay = PlateDisplay()

# create a session to perform inference
with model.as_default():
  with tf.Session(graph=model) as sess:
    # grab a reference to the input image tensor and the boxes
    # tensor
    imageTensor = model.get_tensor_by_name("image_tensor:0")
    boxesTensor = model.get_tensor_by_name("detection_boxes:0")

    # get the list of verified xml files
    xmlFileCnt, xmlFiles = plateXmlExtract.getXmlVerifiedFileList(args["annotations_dir"])
    print("[INFO] Processing {} xml annotation files ...".format(xmlFileCnt))
    frameCnt = 0
    start_time = time.time()
    # loop over the xml files
    for xmlFile in xmlFiles:
      frameCnt += 1
      # grab the image, and get the ground truth boxes and labels
      image, boxes, labels = plateXmlExtract.getXmlData(xmlFile)
      # get the image shape
      (H, W) = image.shape[:2]

      # check to see if we should resize along the width
      #if W > H and W > 1500:
      #  image = imutils.resize(image, width=1500)

      # otherwise, check to see if we should resize along the
      # height
      #elif H > W and H > 1500:
      #  image = imutils.resize(image, height=1500)

      # make 2 copies of the image, one for ground truth labelling and the other for prediction labelling
      imageGT = image.copy()
      imagePreds = image.copy()

      # prepare the image for inference input
      (H, W) = image.shape[:2]
      image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
      image = np.expand_dims(image, axis=0)

      # find the ground truth plates and display
      licensePlateFound_gt, plateBoxes_gt, charTexts_gt, charBoxes_gt = plateFinder.findGroundTruthPlates(boxes, labels)
      if args["image_display"] == "true":
        imageLabelled = plateDisplay.labelImage(imageGT, plateBoxes_gt, charBoxes_gt, charTexts_gt)
        cv2.imshow("Ground truth plates", imageLabelled)

      # for each bounding box we would like to know the score
      # (i.e., probability) and class label
      scoresTensor = model.get_tensor_by_name("detection_scores:0")
      classesTensor = model.get_tensor_by_name("detection_classes:0")
      numDetections = model.get_tensor_by_name("num_detections:0")

      # perform inference and compute the bounding boxes,
      # probabilities, and class labels
      (boxes, scores, labels, N) = sess.run(
        [boxesTensor, scoresTensor, classesTensor, numDetections],
        feed_dict={imageTensor: image})

      # squeeze the lists into a single dimension
      boxes = np.squeeze(boxes)
      scores = np.squeeze(scores)
      labels = np.squeeze(labels)

      # find the plate text associated with each plate, and display
      licensePlateFound_pred, plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateScores_pred = plateFinder.findPlates(boxes, scores, labels, categoryIdx)
      if args["image_display"] == "true":
        imageLabelled = plateDisplay.labelImage(imagePreds, plateBoxes_pred, charBoxes_pred, charTexts_pred)
        cv2.imshow("Predicted plates", imageLabelled)

      # compare ground truth boxes and labels with predicted boxes and labels
      plateWithCharMatchCnt, plateFrameMatchCnt, plateCntTotal_gt, plateCntTotal_pred, charMatchCntTotal, charCntTotal_gt, charCntTotal_pred = \
          plateCompare.comparePlates(plateBoxes_gt, charBoxes_gt, charTexts_gt, plateBoxes_pred, charBoxes_pred, charTexts_pred)

      # warn if no 100% correct plates found
      if plateWithCharMatchCnt == 0:
        print("[INFO] No perfect match plates found in \"{}\". plateFrameMatchCnt: {}, charMatchCntTotal: {}".format(xmlFile, plateFrameMatchCnt, charMatchCntTotal))
        for charText_pred in charTexts_pred:
          print("     Found: {}".format(charText_pred))

      if args["image_display"] == "true":
        # wait for key, so that image windows are displayed
        cv2.waitKey(0)

# print some performance statistics
curTime = time.time()
processingTime = curTime - start_time
fps = frameCnt / processingTime
print("[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCnt, processingTime, fps))
	
# get some stats and print
platesWithCharCorrect, platesCorrect, platesIncorrect, charsCorrect, charsIncorrect = plateCompare.calcStats()
print("[INFO] platesWithCharCorrect: {:.1f}%, platesCorrect: {:.1f}%, platesIncorrect: {:.1f}%, charsCorrect: {:.1f}%, charsIncorrect: {:.1f}%" \
       .format(platesWithCharCorrect * 100, platesCorrect * 100, platesIncorrect * 100, charsCorrect * 100, charsIncorrect * 100))

print("[INFO] Processed {} xml annotation files".format(xmlFileCnt))
