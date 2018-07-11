r"""Convert raw PASCAL like dataset to TFRecord for object_detection.
Enable view_mode if you wish to check your annotations
If view mode is enabled, then the record file will not be written

Example usage:
  python delete_unverified_ann.py \
  --image_dir=images --record_dir=datasets/records --annotations_dir=images
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lxml import etree
import tensorflow as tf
import re
import glob

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Root directory to images.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
FLAGS = flags.FLAGS

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


def list_not_verified_ann(annotations_dir):
  xmlNotVerifiedPaths = []
  xmlPaths = list_files(annotations_dir, validExts=(".xml"))

  # Read each xml file and check if the annotation has been verified.
  # If it has, then add to the verified list
  for xmlPath in xmlPaths:
    xmlFile = open(xmlPath, "r")
    xmlString = xmlFile.read()
    xmlFile.close()
    m = re.search(r"<annotation.*?verified=\"yes\"", xmlString)
    if m == None:
      xmlNotVerifiedPaths.append(xmlPath)

  return xmlNotVerifiedPaths

# remove non matching files
# if a file appears in file1Paths, but not in file2Paths,
# then remove file1
def remove_unmatched_files(file1Paths, file2Paths):
  filesRemoveCnt = 0
  for file1Path in file1Paths:
    file1NameStripped = (".").join((file1Path.split(os.sep)[-1]).split(".")[:-1])
    foundMatch = False
    for file2Path in file2Paths:
      file2NameStripped = (".").join((file2Path.split(os.sep)[-1]).split(".")[:-1])
      if file2NameStripped == file1NameStripped:
        foundMatch = True
        continue
    if foundMatch == False:
      #print("Removing file \"{}\"".format(file1Path))
      os.remove(file1Path)
      filesRemoveCnt += 1
  return filesRemoveCnt


def remove_unverified_files(data, xmlFilePath):
  # Remove unverified XML and associated image file
  # Args:
  #  data: dict holding PASCAL XML fields for a single image (obtained by
  #    running dataset_util.recursive_parse_xml_to_dict)
  #  dataset_directory: Path to root directory holding image sub-directories
  #  label_map_dict: A map from string label names to integers ids.
  #  ignore_difficult_instances: Whether to skip difficult instances in the
  #    dataset  (default: False).
  # data['folder'] should be the name of a sub-directory of datset_directory
  # If you inspect an xml annotation file, 'folder' specifies a single folder containing the
  # corresponding image.
  # 'path' specifies a full absolute path
  filePathRoot = xmlFilePath.split(os.sep)
  filePathRoot = filePathRoot [:-2]
  filePathRoot = (os.sep).join(filePathRoot)
  filePathRoot = os.path.join(filePathRoot, data['folder'])
  full_path = os.path.join(filePathRoot, data['filename'])
  print("Removing \"{}\" and \"{}\"".format(full_path.split(os.sep)[-1], xmlFilePath.split(os.sep)[-1]))
  if os.path.isfile(full_path):
    os.remove(full_path)
  os.remove(xmlFilePath)

def main(_):

  image_dir = FLAGS.image_dir
  annotations_dir = FLAGS.annotations_dir

  xmlPaths = list_not_verified_ann(annotations_dir)
  for xmlPath in xmlPaths:
    with tf.gfile.GFile(xmlPath, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    remove_unverified_files(data, xmlPath)
  print("Removed {} unverified annotation and image files".format(len(xmlPaths)))

  # remove image files that are not referenced by annotation
  xmlPaths = list_files(annotations_dir, validExts=(".xml"))
  xmlPaths = [i for i in xmlPaths]
  imagePaths = list_files(image_dir)
  imagePaths = [i for i in imagePaths]
  imageFileRemoveCnt = remove_unmatched_files(imagePaths, xmlPaths)
  print("Removed {} unannotated image files".format(imageFileRemoveCnt))
  annFileRemoveCnt = remove_unmatched_files(xmlPaths, imagePaths)
  print("Removed {} annotation files with no matching image file".format(annFileRemoveCnt))

if __name__ == '__main__':
  tf.app.run()
