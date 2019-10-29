# Predict license plate boxes and license plate characters in images. Display the labelled images
# USAGE
# Example using single stage Faster RCNN
# python predict_images.py --model datasets/experiment_faster_rcnn/2018_07_25_14-00/exported_model/frozen_inference_graph.pb \
# --pred_stages 1 \
# --labels datasets/records/classes.pbtxt \
# --imagePath images/SJ7STAR_images/2018_02_24_9-00 \
# --num-classes 37 \
# --image_display True

# python predict_images.py --model datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb \
# --pred_stages 2 \
# --labels datasets/records/classes.pbtxt \
# --imagePath images/SJ7STAR_images/2018_02_24_9-00 \
# --num-classes 37 \
# --image_display True

# import the necessary packages
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from object_detection.utils import label_map_util
from base2designs.plates.plateFinder import PlateFinder
from base2designs.plates.predicter import Predicter
from base2designs.plates.plateDisplay import PlateDisplay

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def predictImages(modelArg, labelsArg, imagePathArg, num_classesArg, min_confidenceArg, image_displayArg, pred_stagesArg):


  # initialize the model
  model = tf.Graph()

  # create a context manager that makes this model the default one for
  # execution
  with model.as_default():
    # initialize the graph definition
    graphDef = tf.GraphDef()

    # load the graph from disk
    with tf.gfile.GFile(modelArg, "rb") as f:
      serializedGraph = f.read()
      graphDef.ParseFromString(serializedGraph)
      tf.import_graph_def(graphDef, name="")

  # load the class labels from disk
  labelMap = label_map_util.load_labelmap(labelsArg)
  categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes=num_classesArg,
    use_display_name=True)
  categoryIdx = label_map_util.create_category_index(categories)

  # create a plateFinder
  plateFinder = PlateFinder(min_confidenceArg, categoryIdx,
                            rejectPlates=False, charIOUMax=0.3)

  # create plate displayer
  plateDisplay = PlateDisplay()

  # create a session to perform inference
  with model.as_default():
    with tf.Session(graph=model) as sess:
      # create a predicter, used to predict plates and chars
      predicter = Predicter(model, sess, categoryIdx)

      imagePaths = paths.list_images(imagePathArg)
      frameCnt = 0
      start_time = time.time()
      platesReturn = []
      numPlates = 0
      # Loop over all the images
      for imagePath in imagePaths:
        frameCnt += 1

        # load the image from disk
        print("[INFO] Loading image \"{}\"".format(imagePath))
        image = cv2.imread(imagePath)
        (H, W) = image.shape[:2]

        # If prediction stages == 2, then perform prediction on full image, find the plates, crop the plates from the image,
        # and then perform prediction on the plate images
        if pred_stagesArg == 2:
          # Perform inference on the full image, and then select only the plate boxes
          boxes, scores, labels = predicter.predictPlates(image, preprocess=True)
          licensePlateFound_pred, plateBoxes_pred, plateScores_pred = plateFinder.findPlatesOnly(boxes, scores, labels)
          # loop over the plate boxes, find the chars inside the plate boxes,
          # and then scrub the chars with 'processPlates', resulting in a list of final plateBoxes, char texts, char boxes, char scores and complete plate scores
          plates = []
          for plateBox in plateBoxes_pred:
            boxes, scores, labels = predicter.predictChars(image, plateBox)
            chars = plateFinder.findCharsOnly(boxes, scores, labels, plateBox, image.shape[0], image.shape[1])
            if len(chars) > 0:
              plates.append(chars)
            else:
              plates.append(None)
          plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateAverageScores_pred = plateFinder.processPlates(plates, plateBoxes_pred, plateScores_pred)

        # If prediction stages == 1, then predict the plates and characters in one pass
        elif pred_stagesArg == 1:
          # Perform inference on the full image, and then find the plate text associated with each plate
          boxes, scores, labels = predicter.predictPlates(image, preprocess=False)
          licensePlateFound_pred, plateBoxes_pred, charTexts_pred, charBoxes_pred, charScores_pred, plateScores_pred = plateFinder.findPlates(
            boxes, scores, labels)
        else:
          print("[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(pred_stagesArg))
          quit()

        # Print plate text
        for charText in charTexts_pred:
          print("    Found: ", charText)

        # Display the full image with predicted plates and chars
        if image_displayArg == True:
          imageLabelled = plateDisplay.labelImage(image, plateBoxes_pred, charBoxes_pred, charTexts_pred)
          cv2.imshow("Labelled Image", imageLabelled)
          cv2.waitKey(0)



        imageResults = []
        for i, plateBox in enumerate(plateBoxes_pred):
          imageResults.append({ 'plateText': charTexts_pred[i], 'plateBoxLoc': list(plateBox), 'charBoxLocs': list([list(x) for x in charBoxes_pred[i]])})
          numPlates += 1

        platesReturn.append({'imagePath': imagePath.split("/")[-1], 'imageResults': imageResults})

      # print some performance statistics
      curTime = time.time()
      processingTime = curTime - start_time
      fps = frameCnt / processingTime
      print("[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCnt, processingTime, fps))

      #results = results.encode('utf-8')
      return {"processingTime": processingTime,  "numPlates": numPlates, "numImages": len(platesReturn), "images": platesReturn}

if __name__ == '__main__':
  #tf.app.run()
  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-m", "--model", required=True,
                  help="base path for frozen checkpoint detection graph")
  ap.add_argument("-l", "--labels", required=True,
                  help="labels file")
  ap.add_argument("-i", "--imagePath", required=True,
                  help="path to input image path")
  ap.add_argument("-n", "--num-classes", type=int, required=True,
                  help="# of class labels")
  ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                  help="minimum probability used to filter weak detections")
  ap.add_argument("-d", "--image_display", type=str2bool, default=False,
                  help="Enable display of annotated images")
  ap.add_argument("-p", "--pred_stages", type=int, required=True,
                  help="number of prediction stages")

  args = vars(ap.parse_args())
  results = predictImages(args["model"], args["labels"], args["imagePath"], args["num_classes"], args["min_confidence"],
                 args["image_display"], args["pred_stages"])
