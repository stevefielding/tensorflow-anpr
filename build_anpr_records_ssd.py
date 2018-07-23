r"""Convert raw PASCAL like dataset to TFRecord for object_detection.
Creates separate annotations for the plate box, and the 'plate char' boxes
Enable view_mode if you wish to check your annotations
If view mode is enabled, then the record file will not be written

Example usage:
    python build_anpr_records_ssd.py \
    --record_dir=datasets/records \
    --annotations_dir=images \
    --label_map_file=datasets/records/classes.pbtxt \
    --view_mode=False \
    --image_scale_factor=0.4 \
    --test_record_file=testing_scaled_mixed2.record \
    --train_record_file=training_scaled_mixed2.record
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

FLAGS = flags.FLAGS

logging.basicConfig(filename='build_anpr_records.log', level=logging.DEBUG)


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
# xOffset = 0, yOffset = 0, corresponds to full image. ie no cropping
# warn if the box is outside the crop area
def getBox(plateXMin, plateYMin, plateXMax, plateYMax, filePath, xmin, ymin, xmax, ymax):
  if plateXMin > xmin:
    xmin = plateXMin
    print("[WARNING] plateXMin > xmin in \"{}\"".format(filePath))
  if plateYMin > ymin:
    ymin = plateYMin
    print("[WARNING] plateYMin > ymin in \"{}\"".format(filePath))
  if plateYMax < ymax:
    ymax = plateYMax
    print("[WARNING] plateYMax < ymax in \"{}\"".format(filePath))
  if plateXMax < xmax:
    xmax = plateXMax
    print("[WARNING] plateXMax < xmax in \"{}\"".format(filePath))

  return xmin - plateXMin, ymin - plateYMin, xmax - plateXMin, ymax - plateYMin

# Generate either an TF example of a labelled plate, or the labelled chars within the plate.
# The labelled plate will use the entire image at 'image_path', but the labelled chars will
# use a cropped plate from the image at 'image_path'
def processImage(classSelect, view_mode, imageScaleFactor, dataObjects,
                 ignore_difficult_instances, image_path,
                 dataFileName, label_map_dict):

  # create temporary files for storing images
  imageTempFile = tempfile.NamedTemporaryFile(suffix='.jpg')
  imageTempFileName = imageTempFile.name

  # Load as opencv image, resize and write to temporary file
  imageFull = cv2.imread(image_path)
  imageHeight, imageWidth = imageFull.shape [:2]
  imageResized = imutils.resize(imageFull, width=int(imageWidth * imageScaleFactor))
  cv2.imwrite(imageTempFileName, imageResized)

  # if 'chars' class only, then crop the plate image, and overwrite the image, image width, and image height
  # with the plate image, PIW, and PIH
  plateXMin = 0
  plateYMin = 0
  plateXMax = 0
  plateYMax = 0
  if classSelect == 'chars':
    plateFound = False
    for obj in dataObjects:
      if obj['name'] == 'plate':
        plateXMin = int(obj['bndbox']['xmin'])
        plateYMin = int(obj['bndbox']['ymin'])
        plateXMax = int(obj['bndbox']['xmax'])
        plateYMax = int(obj['bndbox']['ymax'])
        plateImage = imageResized[plateYMin: plateYMax,
                     plateXMin: plateXMax,...]
        cv2.imwrite(imageTempFileName, plateImage)
        imageHeight, imageWidth = plateImage.shape[:2]
        plateFound = True
        continue
    if plateFound == False:
      print("[ERROR] No plate box found in \"{}\"".format(image_path))
      quit()

  if view_mode == True:
    cvImage = cv2.imread(imageTempFileName)
  with tf.gfile.GFile(imageTempFileName, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # get the image dims and scale by imageScaleFactor
  width_original = imageWidth
  height_original = imageHeight
  width_scaled = int(width_original * imageScaleFactor)
  height_scaled = int(height_original * imageScaleFactor)

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in dataObjects:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    # if plates only class selected, then only generate bounding boxes for plates
    if classSelect == 'plates' and obj['name'] != 'plate':
      continue

    # if chars only class selected, then only generate bounding boxes for chars
    if classSelect == 'chars' and obj['name'] == 'plate':
      continue

    difficult_obj.append(int(difficult))
    if classSelect == 'chars':
      xmin, ymin, xmax, ymax = getBox(plateXMin, plateYMin, plateXMax, plateYMax, image_path,
                                    int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                                    int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax']))
    else:
      xmin, ymin, xmax, ymax = (int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                                int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax']))
    xmins.append(float(xmin) / width_original)
    ymins.append(float(ymin) / height_original)
    xmaxs.append(float(xmax) / width_original)
    ymaxs.append(float(ymax) / height_original)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

    if view_mode == True:
      # denormalize the bounding box coordinates, and add bbox rect to image
      startX = int(xmins[-1] * width_scaled)
      startY = int(ymins[-1] * height_scaled)
      endX = int(xmaxs[-1] * width_scaled)
      endY = int(ymaxs[-1] * height_scaled)
      # if plate box then display the bbox in red
      if classes[-1] == 1:
        color = (0,0,255)
      # else display the char box in green and display the char
      else:
        color = (0,255,0)
        text = str(classes_text[-1])
        m = re.match(r"b.*?(\w+)", text)
        cv2.putText(cvImage, m.group(1), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
      # draw the bounding box on the image
      cv2.rectangle(cvImage, (startX, startY), (endX, endY),
                    color, 1)

  if view_mode == True:
    # show the output image
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




def dict_to_tf_example(data,
                       label_map_dict, xmlFilePath, imageScaleFactor,
                       ignore_difficult_instances=False,
                       view_mode=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    xmlFilePath: Path to the xml annotation files
    imageScaleFactor: Factor to scale down the images. Typically only used to scale down the
      full images, not the plate images
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  # generate path to image file associated with the annotation
  # Assumes that annotations and images are in seperate directories, and they are both
  # sub-directories of the same root directory.
  # eg images at 'dataset/2018_02_24' and annotations at dataset/2018_02_24_ann
  # data['folder'] should be the name of a sub-directory of datset_directory
  # If you inspect an xml annotation file, 'folder' specifies a single folder containing the
  # corresponding image. 'path' specifies a full absolute path
  filePathRoot = xmlFilePath.split(os.sep)
  filePathRoot = filePathRoot [:-2]
  filePathRoot = (os.sep).join(filePathRoot)
  filePathRoot = os.path.join(filePathRoot, data['folder'])
  image_path = os.path.join(filePathRoot, data['filename'])

  exampleFindPlate = processImage('plates', view_mode,
                                  imageScaleFactor, data['object'], ignore_difficult_instances, image_path,
                                  data['filename'], label_map_dict)

  exampleFindChars = processImage('chars', view_mode,
                                  1.0, data['object'], ignore_difficult_instances, image_path,
                                  data['filename'], label_map_dict)

  return (exampleFindPlate, exampleFindChars)

def create_record(xmlList, label_map_file, recordFilePath, imageScaleFactor, view_mode=False,
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
    tf_example_plate, tf_example_chars = dict_to_tf_example(data, label_map_dict, xmlFilePath, imageScaleFactor,
                                    ignore_difficult_instances, view_mode=view_mode)
    if view_mode == False:
      writer.write(tf_example_plate.SerializeToString())
      writer.write(tf_example_chars.SerializeToString())

  if view_mode == False:
    writer.close()


def main(_):

  annotations_dir = FLAGS.annotations_dir
  label_map_file = FLAGS.label_map_file
  view_mode = FLAGS.view_mode
  imageScaleFactor = FLAGS.image_scale_factor

  # split the dataset into training data and testing data
  # Note that we are splitting the xml annotation files
  # If an image does not have a corresponding annotation file
  # it will not be used
  (trainList,testList) = create_train_test_split(annotations_dir)
  print("[INFO] Found {} annotation files".format(len(trainList) + len(testList)))
  print("[INFO] Splitting into {} training files, and {} testing files".format(len(trainList), len(testList)))
  print("[INFO] Note that the TF Records contain double the number of annotation files")
  print("[INFO] ie for every annotation file, a separate plate and 'plate char' pair will be created")

  # create the training record
  trainingFilePath = os.path.join(FLAGS.record_dir , FLAGS.train_record_file)
  print("[INFO] Writing \"{}\", containing {} images".format(trainingFilePath, len(trainList)*2))
  create_record(trainList, label_map_file,
                trainingFilePath, imageScaleFactor, view_mode=view_mode,
                ignore_difficult_instances=FLAGS.ignore_difficult_instances)

  # create the testing record
  testingFilePath = os.path.join(FLAGS.record_dir , FLAGS.test_record_file)
  print("[INFO] Writing \"{}\", containing {} images".format(testingFilePath, len(testList)*2))
  create_record(testList, label_map_file,
                testingFilePath, imageScaleFactor, view_mode=view_mode,
                ignore_difficult_instances=FLAGS.ignore_difficult_instances)

if __name__ == '__main__':
  tf.app.run()
