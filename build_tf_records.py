r"""Convert raw PASCAL like dataset to TFRecord for object_detection.
Enable view_mode if you wish to check your annotations
If view mode is enabled, then the record file will not be written
If plate_split_label != 'none', then the TFRecord will contain separate annotations for the top level boxes,
and the 'bottom level boxes.

Example usage, with plate char split:
    python build_tf_records.py \
    --record_dir=datasets/records \
    --annotations_dir=images \
    --label_map_file=datasets/records/classes.pbtxt \
    --view_mode=False \
    --image_scale_factor=0.4 \
    --test_record_file=testing_plate_char_split.record \
    --train_record_file=training_plate_char_split.record
    --split_label='plate'

Example usage, with no split:
    python build_tf_records.py \
    --record_dir=datasets/records \
    --annotations_dir=images \
    --label_map_file=datasets/records/classes.pbtxt \
    --view_mode=False \
    --test_record_file=testing_plate_char_combined.record \
    --train_record_file=training_plate_char_combined.record \
    --split_label=none
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import tempfile
import imutils
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow as tf
import cv2
import re
from sklearn.model_selection import train_test_split

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('record_dir', '', 'Path to output TFRecord')
flags.DEFINE_string('test_record_file', 'testing.record', 'Testing record filename')
flags.DEFINE_string('train_record_file', 'training.record', 'Training record filename')
flags.DEFINE_string('label_map_file', '',
                    'label map proto file')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_boolean('view_mode', False, 'View mode enable')
flags.DEFINE_float('image_scale_factor',1.0,'image scale factor')
flags.DEFINE_string('split_label', 'none', 'Name of the label to use for split. \'none\' defines no split')
FLAGS = flags.FLAGS

logging.basicConfig(filename='build_tf_records.log', level=logging.DEBUG)


def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
  # loop over the directory structure
  for (rootDir, dirNames, filenames) in os.walk(basePath, followlinks=True):
    # loop over the filenames in the current directory
    for filename in filenames:
      # if the contains string is not none and the filename does not contain
      # the supplied string, then ignore the file
      if contains is not None and filename.find(contains) == -1:
        continue

      # determine the file extension of the current file
      ext = filename[filename.rfind("."):].lower()

      # check to see if the file is an image and should be processed
      if ext.endswith(validExts):
        # construct the path to the image and yield it
        imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
        yield imagePath

def create_train_test_split(annotations_dir):
  xmlVerifiedPaths = []
  xmlPaths = list_files(annotations_dir, validExts=(".xml"))

  # Read each xml file and check if the annotation has been verified.
  # If it has, then add to the verified list
  for xmlPath in xmlPaths:
    xmlFile = open(xmlPath, "r")
    xmlString = xmlFile.read()
    xmlFile.close()
    m = re.search(r"<annotation.*?verified=\"yes\"", xmlString)
    if m != None:
      xmlVerifiedPaths.append(xmlPath)

  # create training and testing splits from our data dictionary
  (trainFiles, testFiles) = train_test_split(list(xmlVerifiedPaths),
    test_size=0.15, random_state=42)
  return (trainFiles, testFiles)

# generate new bounding box for cropped image
# Return the bottomBox box co-ordinates relative to the cropped topBox box
# warn if the bottomBox box is outside the cropped topBox box
def getBox(topBoxXMin, topBoxYMin, topBoxXMax, topBoxYMax, filePath, xmin, ymin, xmax, ymax):
  if topBoxXMin > xmin:
    xmin = topBoxXMin
    print("[WARNING] topBoxXMin > xmin in \"{}\"".format(filePath))
  if topBoxYMin > ymin:
    ymin = topBoxYMin
    print("[WARNING] topBoxYMin > ymin in \"{}\"".format(filePath))
  if topBoxYMax < ymax:
    ymax = topBoxYMax
    print("[WARNING] topBoxYMax < ymax in \"{}\"".format(filePath))
  if topBoxXMax < xmax:
    xmax = topBoxXMax
    print("[WARNING] topBoxXMax < xmax in \"{}\"".format(filePath))

  return xmin - topBoxXMin, ymin - topBoxYMin, xmax - topBoxXMin, ymax - topBoxYMin

def genSquareImage(image):
  debugFlag = False
  blue = image[...,0]
  green = image[...,1]
  red = image[...,2]
  blueAv = np.sum(blue)/ blue.size
  greenAv = np.sum(green)/ green.size
  redAv = np.sum(red)/ red.size
  dimMax = max(image.shape[0], image.shape[1])
  imageOut = np.empty((dimMax,dimMax,3), dtype=np.uint8)
  imageOut[...,0] = np.uint8(blueAv)
  imageOut[...,1] = np.uint8(greenAv)
  imageOut[...,2] = np.uint8(redAv)
  if image.shape[0] <= image.shape[1]:
    imageOut[0:image.shape[0],...] = image
  else:
    imageOut[:,0:image.shape[1],...] = image
    #debugFlag = True
  return imageOut, debugFlag

# Convert XML derived dict to tf.Example proto.
# Generate either an TF example of a labelled topBox, or the labelled bottomBoxes within the topBox.
# The labelled topBox will use the entire image at 'image_path', but the labelled bottomBoxes will
# use a cropped topBox from the image at 'image_path'
#  Args:
#    split_top: Split out the top boxes (True), bottom boxes (False), or all boxes (None)
#    split_label: Top level label to split on
#    view_mode: Enable image display
#    imageScaleFactor: Factor to scale down the images. Typically only used to scale down the
#      images containing topBoxes. Mainly required for records that will be used for SSD training. SSD training
#      can easily run out of memory, and crash, if the images in the record file are too large. imageScaleFactor = 0.4
#      works for desktop with 16GB of RAM and TitanX with 12GB of RAM
#    dataObjects: dict holding PASCAL XML fields for image labels
#    ignore_difficult_instances: Whether to skip difficult instances in the
#      dataset  (default: False).
#    dataFileName: filename extracted from xml file. I think this is the image file name
#    label_map_dict: A map from string label names to integers ids.
#  Returns:
#    example: The converted tf.Example.
def dict_to_tf_example(split_top, split_label, view_mode, imageScaleFactor, dataObjects,
                 ignore_difficult_instances, image_path,
                 dataFileName, label_map_dict):
  debugFlag = False

  # create temporary files for storing images
  imageTempFile = tempfile.NamedTemporaryFile(suffix='.jpg')
  imageTempFileName = imageTempFile.name

  if split_top != True and split_top != False:
    # open the image, get the height and width, and copy the image to temporary file
    imageFull = cv2.imread(image_path)
    imageHeight, imageWidth = imageFull.shape [:2]
    height_scaled, width_scaled = imageHeight, imageWidth
    cv2.imwrite(imageTempFileName, imageFull)
  else:
    # Load as opencv image, square, resize and write to temporary file
    imageFull = cv2.imread(image_path)
    imageSquare, debugFullImageFlag = genSquareImage(imageFull)
    imageHeight, imageWidth = imageSquare.shape [:2]
    width_scaled = int(imageWidth * imageScaleFactor)
    height_scaled = int(imageHeight * imageScaleFactor)
    imageResized = imutils.resize(imageSquare, width=int(imageWidth * imageScaleFactor))
    cv2.imwrite(imageTempFileName, imageResized)

  # if bottomBoxes selected, then crop the topBox from the image, and overwrite the image, image width, and image height
  # with the topBox image, PIW, and PIH
  topBoxXMin = 0
  topBoxYMin = 0
  topBoxXMax = 0
  topBoxYMax = 0
  if split_top == False:
    topBoxFound = False
    for obj in dataObjects:
      if obj['name'] == split_label:
        topBoxXMin = int(obj['bndbox']['xmin'])
        topBoxYMin = int(obj['bndbox']['ymin'])
        topBoxXMax = int(obj['bndbox']['xmax'])
        topBoxYMax = int(obj['bndbox']['ymax'])
        topBoxImage = imageSquare[topBoxYMin: topBoxYMax,
                     topBoxXMin: topBoxXMax,...]
        topBoxImage, debugFlag = genSquareImage(topBoxImage)
        cv2.imwrite(imageTempFileName, topBoxImage)
        imageHeight, imageWidth = topBoxImage.shape[:2]
        width_scaled = int(imageWidth)
        height_scaled = int(imageHeight)
        topBoxFound = True
        continue
    if topBoxFound == False:
      print("[ERROR] No top level box found in \"{}\"".format(image_path))
      quit()

  # open image file as tensorflow compatible. Check if image file is jpeg format
  if view_mode == True or debugFlag == True:
    cvImage = cv2.imread(imageTempFileName)
  with tf.gfile.GFile(imageTempFileName, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # init the attribute lists
  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  # loop over all the labels in the annotation
  for obj in dataObjects:
    difficult = bool(int(obj['difficult']))
    # discard difficult objects
    if ignore_difficult_instances and difficult:
      continue
    # if split_top labels selected, then only generate bounding boxes for labels in split_label
    if split_top == True and obj['name'] != split_label:
      continue
    # if split_top labels not selected, then only generate bounding boxes for labels not in split_label
    if split_top == False and obj['name'] == split_label:
      continue

    # if split bottom level boxes selected, then convert bottom level box co-ordinates to be relative to cropped top box
    # else maintain the original box co-ordinates
    if split_top == False:
      xmin, ymin, xmax, ymax = getBox(topBoxXMin, topBoxYMin, topBoxXMax, topBoxYMax, image_path,
                                    int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                                    int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax']))
    else:
      xmin, ymin, xmax, ymax = (int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                                int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax']))

    # Normalize box co-ordinates
    xmins.append(float(xmin) / imageWidth)
    ymins.append(float(ymin) / imageHeight)
    xmaxs.append(float(xmax) / imageWidth)
    ymaxs.append(float(ymax) / imageHeight)

    # Add info to object attribute lists
    difficult_obj.append(int(difficult))
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

    # if view mode is enabled, then add top bounding boxes, bottom bounding boxes bottom box labels to the image
    if view_mode == True or debugFlag == True:
      # denormalize the bounding box coordinates, and add bbox rect to image
      startX = int(xmins[-1] * width_scaled)
      startY = int(ymins[-1] * height_scaled)
      endX = int(xmaxs[-1] * width_scaled)
      endY = int(ymaxs[-1] * height_scaled)
      # if topBox box then display the bbox in red
      if classes[-1] == 1:
        color = (0,0,255)
      # else display the bottom box in green and display the bottom labels
      else:
        color = (0,255,0)
        text = str(classes_text[-1])
        m = re.match(r"b.*?(\w+)", text)
        cv2.putText(cvImage, m.group(1), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
      # draw the bounding box on the image
      cv2.rectangle(cvImage, (startX, startY), (endX, endY),
                    color, 1)

  # display the labelled image
  if view_mode == True or debugFlag == True:
    cv2.imshow("Image", cvImage)
    cv2.imwrite("myImage.jpg",cvImage)
    cv2.waitKey(0)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height_scaled),
      'image/width': dataset_util.int64_feature(width_scaled),
      'image/filename': dataset_util.bytes_feature(
        dataFileName.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
        dataFileName.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def getImagePath(xmlFilePath, imageFolder, imageFileName):
  # generate path to image file associated with the annotation
  # Assumes that annotations and images are in seperate directories, and they are both
  # sub-directories of the same root directory.
  # eg 'dataset/2018_02_24' and 'dataset/2018_02_24_ann'
  # The annotation file defines data['folder'], which should be the name of a sub-directory of datset_directory
  # If you inspect an xml annotation file, 'folder' specifies a single folder containing the
  # corresponding image. 'path' specifies a full absolute path
  filePathRoot = xmlFilePath.split(os.sep)
  filePathRoot = filePathRoot [:-2]
  filePathRoot = (os.sep).join(filePathRoot)
  filePathRoot = os.path.join(filePathRoot, imageFolder)
  image_path = os.path.join(filePathRoot, imageFileName)
  return image_path

# Create TF records. imageScaleFactor not used if split_label == 'none'
def create_record(xmlList, label_map_file, recordFilePath, imageScaleFactor, split_enable, split_label, view_mode=False,
                  ignore_difficult_instances=False):
  # for every xml file, read the annotation and the image file, and add to the record file

  if view_mode == False:
    writer = tf.python_io.TFRecordWriter(recordFilePath)

  label_map_dict = label_map_util.get_label_map_dict(label_map_file)

  logging.info('Reading from dataset.')
  for idx, xmlFilePath in enumerate(xmlList):
    if idx % 100 == 0:
      logging.info('On annotation %d of %d', idx, len(xmlList))
    with tf.gfile.GFile(xmlFilePath, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    image_path = getImagePath(xmlFilePath, data['folder'], data['filename'])
    if split_enable == True:
      tf_example_top = dict_to_tf_example(True, split_label, view_mode,
                                      imageScaleFactor, data['object'], ignore_difficult_instances, image_path,
                                      data['filename'], label_map_dict)

      tf_example_bottom = dict_to_tf_example(False, split_label, view_mode,
                                      1.0, data['object'], ignore_difficult_instances, image_path,
                                      data['filename'], label_map_dict)
    else:
      tf_example_all = dict_to_tf_example(None, split_label, view_mode,
                                      1.0, data['object'], ignore_difficult_instances, image_path,
                                      data['filename'], label_map_dict)
    if view_mode == False:
      if split_enable == True:
        writer.write(tf_example_top.SerializeToString())
        writer.write(tf_example_bottom.SerializeToString())
      else:
        writer.write(tf_example_all.SerializeToString())

  if view_mode == False:
    writer.close()


def main(_):

  annotations_dir = FLAGS.annotations_dir
  label_map_file = FLAGS.label_map_file
  view_mode = FLAGS.view_mode
  imageScaleFactor = FLAGS.image_scale_factor
  split_label = FLAGS.split_label
  if split_label.lower() == 'none':
    split_enable = False
    fileNumMult = 1
  else:
    split_enable = True
    fileNumMult = 2

  # split the dataset into training data and testing data
  # Note that we are splitting the xml annotation files
  # If an image does not have a corresponding annotation file
  # it will not be used
  (trainList,testList) = create_train_test_split(annotations_dir)
  print("[INFO] Found {} annotation files".format(len(trainList) + len(testList)))
  print("[INFO] Splitting into {} training files, and {} testing files".format(len(trainList), len(testList)))
  print("[INFO] Splitting labels based on top level label: \'{}\'".format(split_label))
  print("[INFO] If split_label != \'none\', then the TF Records will contain double the number of annotation files")
  print("[INFO] For license plate example, every annotation file results in a separate plate and 'plate chars' pair")

  # create the training record
  trainingFilePath = os.path.join(FLAGS.record_dir , FLAGS.train_record_file)
  print("[INFO] Writing \"{}\", containing {} images".format(trainingFilePath, len(trainList)*fileNumMult))
  create_record(trainList, label_map_file,
                trainingFilePath, imageScaleFactor, split_enable, split_label, view_mode=view_mode,
                ignore_difficult_instances=FLAGS.ignore_difficult_instances)

  # create the testing record
  testingFilePath = os.path.join(FLAGS.record_dir , FLAGS.test_record_file)
  print("[INFO] Writing \"{}\", containing {} images".format(testingFilePath, len(testList)*fileNumMult))
  create_record(testList, label_map_file,
                testingFilePath, imageScaleFactor, split_enable, split_label, view_mode=view_mode,
                ignore_difficult_instances=FLAGS.ignore_difficult_instances)

if __name__ == '__main__':
  tf.app.run()
