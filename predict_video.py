# USAGE
# $ python predict_video.py \
# 	--model dlib_front_and_rear_vehicles_v1/experiments/exported_model/frozen_inference_graph.pb \
# 	--labels dlib_front_and_rear_vehicles_v1/records/classes.pbtxt \
# 	--input example_input.avi --output output.avi --num-classes 2

# import the necessary packages
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
	help="labels file")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-n", "--num-classes", type=int, required=True,
	help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# initialize the colors list and the model
COLORS = [(0, 255, 0), (0, 0, 255)]
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

# create a session to perform inference
with model.as_default():
	with tf.Session(graph=model) as sess:
		# initialize the points to the video files
		stream = cv2.VideoCapture(args["input"])
		writer = None

		# loop over frames from the video file stream
		while True:
			# grab the next frame
			(grabbed, image) = stream.read()

			# if the frame was not grabbed, then we have reached the
			# end of the stream
			if not grabbed:
				break

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
			if W > H and W > 1000:
				image = imutils.resize(image, width=1000)

			# otherwise, check to see if we should resize along the
			# height
			elif H > W and H > 1000:
				image = imutils.resize(image, height=1000)

			# prepare the image for detection
			(H, W) = image.shape[:2]
			output = image.copy()
			image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
			image = np.expand_dims(image, axis=0)

			# if the video writer is None, initialize if
			if writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 20,
					(W, H), True)

			# perform inference and compute the bounding boxes,
			# probabilities, and class labels
			(boxes, scores, labels, N) = sess.run(
				[boxesTensor, scoresTensor, classesTensor, numDetections],
				feed_dict={imageTensor: image})

			# squeeze the lists into a single dimension
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			labels = np.squeeze(labels)

			# loop over the bounding box predictions
			for (box, score, label) in zip(boxes, scores, labels):
				# if the predicted probability is less than the minimum
				# confidence, ignore it
				if score < args["min_confidence"]:
					continue

				# scale the bounding box from the range [0, 1] to [W, H]
				(startY, startX, endY, endX) = box
				startX = int(startX * W)
				startY = int(startY * H)
				endX = int(endX * W)
				endY = int(endY * H)

				# draw the prediction on the output image
				label = categoryIdx[label]
				idx = int(label["id"]) - 1
				label = "{}: {:.2f}".format(label["name"], score)
				cv2.rectangle(output, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.putText(output, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS[idx], 1)

			# write the frame to the output file
			writer.write(output)

		# close the video file pointers
		writer.release()
		stream.release()