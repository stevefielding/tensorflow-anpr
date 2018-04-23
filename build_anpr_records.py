r"""Convert raw PASCAL like dataset to TFRecord for object_detection.
Enable view_mode if you wish to check your annotations
If view mode is enabled, then the record file will not be written

Example usage:
     python build_anpr_records.py \
    --dataset_file=SJ7STAR_images/imageSets/train.txt \
    --image_dir=SJ7STAR_images \
    --output_path=SJ7STAR_images/records/training.record \
    --annotations_dir=SJ7STAR_images/2018_02_24_9-00_ann \
    --label_map_file=SJ7STAR_images/records/classes.pbtxt \
    --view_mode=False
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
import cv2
import re

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Root directory to images.')
flags.DEFINE_string('dataset_file', '', 'File defining the set of images to use.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_file', '',
                    'label map proto file')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_boolean('view_mode', False, 'View mode enable')

FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       view_mode=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding image sub-directories
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  # data['folder'] should be the name of a sub-directory of datset_directory
  # If you inspect an xml annotation file, 'folder' specifies a single folder containing the
  # corresponding image. 'path' specifies a full absolute path
  img_path = os.path.join(data['folder'], data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  if view_mode == True:
    cvImage = cv2.imread(full_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

    if view_mode == True:
      # denormalize the bounding box coordinates, and add bbox rect to image
      startX = int(xmin[-1] * width)
      startY = int(ymin[-1] * height)
      endX = int(xmax[-1] * width)
      endY = int(ymax[-1] * height)
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
    cv2.waitKey(0)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):

  image_dir = FLAGS.image_dir
  annotations_dir = FLAGS.annotations_dir
  dataset_file = FLAGS.dataset_file
  label_map_file = FLAGS.label_map_file
  view_mode = FLAGS.view_mode

  if view_mode == False:
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(label_map_file)

  logging.info('Reading from %s dataset.', dataset_file)
  examples_list = dataset_util.read_examples_list(dataset_file)
  for idx, example in enumerate(examples_list):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples_list))
    path = os.path.join(annotations_dir, example + '.xml')
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    tf_example = dict_to_tf_example(data, image_dir, label_map_dict,
                                    FLAGS.ignore_difficult_instances, view_mode=view_mode)
    if view_mode == False:
      writer.write(tf_example.SerializeToString())

  if view_mode == False:
    writer.close()


if __name__ == '__main__':
  tf.app.run()
