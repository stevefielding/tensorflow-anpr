import os
import re
import cv2
from lxml import etree
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import numpy as np

class PlateXmlExtract():

  def __init__(self, label_map_file):
    # label_map_dict: A map from string label names to integers ids.
    self.label_map_dict = label_map_util.get_label_map_dict(label_map_file)

  # yield a list of files
  def list_files(self, basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
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

  # return a list of verified xml files, and the the number of files found
  def getXmlVerifiedFileList(self, annotations_dir):
    xmlVerifiedPaths = []
    xmlPaths = self.list_files(annotations_dir, validExts=(".xml"))

    # Read each xml file and check if the annotation has been verified.
    # If it has, then add to the verified list
    fileCnt = 0
    for xmlPath in xmlPaths:
      xmlFile = open(xmlPath, "r")
      xmlString = xmlFile.read()
      xmlFile.close()
      m = re.search(r"<annotation.*?verified=\"yes\"", xmlString)
      if m != None:
        xmlVerifiedPaths.append(xmlPath)
        fileCnt += 1

    return fileCnt, xmlVerifiedPaths

  # From xml dictionary, extract box co-ordinates and labels.
  # Read the referenced image as a CV BGR image
  # Return image, boxes and labels
  def dictToBoxData(self, data, xmlFilePath):
    """ Read xml file, and return the referenced image, and bounding box info
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      xmlFilePath: Path to xml files
    Returns:
      cvImage: full image in cv format
      xmin: list of bounding box xmin
      ymin: list of bounding box ymin
      xmax: list of bounding box xmax
      ymax: list of bounding box ymax
      classes_text: list of box classes
    Raises:
    """
    # data['folder'] should be the name of a sub-directory of datset_directory
    # If you inspect an xml annotation file, 'folder' specifies a single folder containing the
    # corresponding image. 'path' specifies a full absolute path
    filePathRoot = xmlFilePath.split(os.sep)
    filePathRoot = filePathRoot[:-2]
    filePathRoot = (os.sep).join(filePathRoot)
    filePathRoot = os.path.join(filePathRoot, data['folder'])
    full_path = os.path.join(filePathRoot, data['filename'])
    cvImage = cv2.imread(full_path)
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    for obj in data['object']:
      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      #classes_text.append(obj['name'].encode('utf8'))
      classes_text.append(obj['name'].encode('UTF8'))
      classes.append(self.label_map_dict[obj['name']])
    return cvImage, xmin, ymin, xmax, ymax, classes_text

  # convert xml file to dict, and then extract the image, box and label info
  def getXmlData(self, xmlFilePath):
    # for xml file, read the annotation and the image file
    with tf.gfile.GFile(xmlFilePath, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    cvImage, xmin, ymin, xmax, ymax, classes_text =  self.dictToBoxData(data, xmlFilePath)
    boxes = np.stack((ymin,xmin,ymax, xmax), axis=-1)

    return cvImage, boxes, classes_text