# USAGE
# Two stage SSD usage
# python predict_images_and_score.py --model datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb \
# --labels datasets/records/classes.pbtxt \
# --annotations_dir images_verification \
# --num-classes 37 \
# --min-confidence 0.1 \
# --pred_stages 2
# Single stage faster R-CNN usage
# python predict_images_and_score.py --model datasets/experiment_faster_rcnn/2018_07_15/exported_model/frozen_inference_graph.pb \
# --labels datasets/records/classes.pbtxt \
# --annotations_dir images_verification \
# --num-classes 37 \
# --min-confidence 0.5 \
# --pred_stages 1
# Use boolean arg "--image_display True" if you wish to see the annotated images.
# Scans the annotations_dir and looks for PASCAL_VOC annotations with verified=yes, reads all the associated image
# files, and performs inference. Predicted results are filtered by plateFinder, which discards char boxes that are
# outside a platebox (if pred_stages = 1), discards char boxes that overlap each other,
# and then orders the chars within each plate box from left to right.
# Predicted labels and bounding boxes are compared to the ground truths, and the
# labelled image is optionally displayed.
# Scores are displayed for plates and chars.

# Example using PyCallGraph. Unfortunately calls to tensorflow session are not visible
# See https://pycallgraph.readthedocs.io/en/master/index.html
# pycallgraph --max-depth=3 graphviz -- predict_images_and_score.py
# --model datasets/experiment_ssd_mobilenet_v1/2018_08_09/exported_model/frozen_inference_graph.pb
# --labels datasets/records/classes.pbtxt --annotations_dir images_verification --num-classes 37
# --min-confidence 0.1 --pred_stages 2


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
from base2designs.plates.predicter import Predicter
import time


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

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
ap.add_argument("-d", "--image_display", type=str2bool, default=False,
  help="Enable display of ground truth and predicted annotated images")
ap.add_argument("-p", "--pred_stages", type=int, required=True,
  help="number of prediction stages")

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
plateFinder = PlateFinder(args["min_confidence"], categoryIdx, rejectPlates=True)

# create a plate xml extractor
plateXmlExtract = PlateXmlExtract(args["labels"])

# create a plate comparator
plateCompare = PlateCompare()

# create plate displayer
plateDisplay = PlateDisplay()

# create a session to perform inference
with model.as_default():
  with tf.Session(graph=model) as sess:
    # create a predicter, used to predict plates and chars
    predicter = Predicter(model, sess, categoryIdx)

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

      # make 1 copy of the image for prediction labelling
      image_gt = image
      image_pred = image.copy()

      # find the ground truth plates and chars, and display
      licensePlateFound_gt, plateBoxes_gt, charTexts_gt, charBoxes_gt = plateFinder.findGroundTruthPlates(boxes, labels)
      if args["image_display"] == True:
        imageLabelled = plateDisplay.labelImage(image_gt, plateBoxes_gt, charBoxes_gt, charTexts_gt)
        cv2.imshow("Ground truth plates", imageLabelled)

      # If prediction stages == 2, then perform prediction on full image, find the plates, crop the plates from the image,
      # and then perform prediction on the plate images
      if args["pred_stages"] == 2:
        # Perform inference on the full image, and then select only the plate boxes
        boxes, scores, labels = predicter.predictPlates(image_pred, preprocess=True)
        licensePlateFound_pred, plateBoxes_pred, plateScores_pred = plateFinder.findPlatesOnly(boxes, scores, labels)
        # loop over the plate boxes, find the chars inside the plate boxes,
        # and then scrub the chars with 'processPlates', resulting in a list of final plateBoxes, char texts, char boxes, char scores and complete plate scores
        plates = []
        for plateBox in plateBoxes_pred:
          boxes, scores, labels = predicter.predictChars(image_pred, plateBox)
          chars = plateFinder.findCharsOnly(boxes, scores, labels, plateBox, image_pred.shape[0], image_pred.shape[1])
          if len(chars) > 0:
            plates.append(chars)
          else:
            plates.append(None)
        plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateAverageScores_pred = plateFinder.processPlates(plates, plateBoxes_pred, plateScores_pred)

      # If prediction stages == 1, then predict the plates and characters in one pass
      elif args["pred_stages"] == 1:
        # Perform inference on the full image, and then find the plate text associated with each plate
        boxes, scores, labels = predicter.predictPlates(image_pred, preprocess=False)
        licensePlateFound_pred, plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateScores_pred = plateFinder.findPlates(
          boxes, scores, labels)
      else:
        print("[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(args["pred_stages"]))
        quit()

      # Display the full image with predicted plates and chars
      if args["image_display"] == True:
        imageLabelled = plateDisplay.labelImage(image_pred, plateBoxes_pred, charBoxes_pred, charTexts_pred)
        cv2.imshow("Predicted plates", imageLabelled)

      # compare ground truth boxes and labels with predicted boxes and labels
      plateWithCharMatchCnt, plateFrameMatchCnt, plateCntTotal_gt, plateCntTotal_pred, charMatchCntTotal, charCntTotal_gt, charCntTotal_pred = \
          plateCompare.comparePlates(plateBoxes_gt, charBoxes_gt, charTexts_gt, plateBoxes_pred, charBoxes_pred, charTexts_pred)

      # warn if no 100% correct plates found
      if plateWithCharMatchCnt == 0:
        print("[INFO] No perfect match plates found in \"{}\". Plate frame matches: {}/{}, Char matches: {}/{}"
              .format(xmlFile, plateFrameMatchCnt, plateCntTotal_gt, charMatchCntTotal, charCntTotal_gt))
        for charText_pred in charTexts_pred:
          print("     Found: {}".format(charText_pred))

      if args["image_display"] == True:
        # wait for key, so that image windows are displayed
        cv2.waitKey(0)

# print some performance statistics
curTime = time.time()
processingTime = curTime - start_time
fps = frameCnt / processingTime
print("[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCnt, processingTime, fps))
	
# get some stats and print
platesWithCharCorrect_recall, platesWithCharCorrect_precision, plateFrames_recall, plateFrames_precision, chars_recall, chars_precision = plateCompare.calcStats()
print("[INFO] platesWithCharCorrect_recall: {:.1f}%, platesWithCharCorrect_precision: {:.1f}%, "
      "\n       plateFrames_recall: {:.1f}%, plateFrames_precision: {:.1f}%, "
      "\n       chars_recall: {:.1f}%, chars_precision: {:.1f}%" \
       .format(platesWithCharCorrect_recall * 100, platesWithCharCorrect_precision * 100, plateFrames_recall * 100,
               plateFrames_precision * 100, chars_recall * 100, chars_precision * 100))
print("[INFO] Definitions. Precision: Percentage of all the objects detected that are correct. Recall: Percentage of ground truth objects that are detected")
print("[INFO] Processed {} xml annotation files".format(xmlFileCnt))
