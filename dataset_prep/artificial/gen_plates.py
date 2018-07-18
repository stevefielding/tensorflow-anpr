#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

# Original gen.py modified to generate CA specific plates with random characters

"""
Generate training and test images.
USAGE:
python gen_plates.py --numImages 1000 --imagePath artificial_images/CA --xmlPath artificial_images/CA_ann

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import argparse
import cv2
import numpy
import re

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common
from base2designs.utils.pascalVocXml import PascalVocXml

FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized


OUTPUT_SHAPE = (1080, 1920)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors(enable_rand_polarity_bg=True):
    first = True
    while first or abs(plate_color - text_color) < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if (enable_rand_polarity_bg == False and text_color > plate_color):
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color

# scale_variation seems to determine percentage of images that are considered out of bounds. Assume
# that rotation_variation and translation_variation do something similar
def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    # (0.6 + 0.875) * 0.5 - (0.875 - 0.6) * 0.5 * 1.5 ...
    # (0.7375 - 0.20625) to (0.7375 + 0.20625)
    # 0.53125 to 0.944
    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    #roll = random.uniform(-0.3, 0.3) * rotation_variation
    #pitch = random.uniform(-0.2, 0.2) * rotation_variation
    #yaw = random.uniform(-1.2, 1.2) * rotation_variation
    roll = random.uniform(-0.1, 0.1) * rotation_variation
    pitch = random.uniform(-0.1, 0.1) * rotation_variation
    yaw = random.uniform(-1.0, 1.0) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    weightAxes = numpy.array([[1.0],[10.0]])
    trans = (numpy.random.random((2,1)) - 0.5) * weightAxes * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code():
    return "{}{}{}{}{}{}{}".format(
        random.choice(common.CHARS),
        random.choice(common.CHARS),
        random.choice(common.CHARS),
        random.choice(common.CHARS),
        random.choice(common.CHARS),
        random.choice(common.CHARS),
        random.choice(common.CHARS))


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims, enable_rand_polarity_bg):
    h_padding = random.uniform(0.05, 0.11) * font_height
    v_padBot = random.uniform(0.15, 0.25) * font_height
    #v_padTop = random.uniform(0.6, 0.7) * font_height
    v_padTop = random.uniform(0.15, 0.25) * font_height

    # Note that the font already contains spacing, so it is possible to have negative spacing
    spacing = font_height * random.uniform(-0.065, -0.06)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padBot + v_padTop),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors(enable_rand_polarity_bg)

    # init the plate image
    text_mask = numpy.zeros(out_shape)

    # Add characters to the plate image
    x = h_padding
    y = v_padTop
    charShapes = []
    for c in code:
        # do not understand how char_im height can match the arguement font_height ?????
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        # copy char_im into the plate image
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im

        # get the char shape
        charShape = numpy.shape(char_im)
        # charShape[0:3] become: left, top, right, bottom. All measured relative to plate position
        charShape = (ix - spacing, iy, charShape[1] + ix + spacing, charShape[0] + iy)
        charShapes.append(charShape)

        # update the horizontal position for the next char_im
        x += char_im.shape[1] + spacing

    # Add plate chars to plate background
    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    code = code.lower()
    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""), charShapes


def generate_bg(num_bg_images):
    fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
    bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
    bg = cv2.resize(bg, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
    return bg

# transform the box and then convert 4 co-ordinate quad to 2 co-ordinate rectangle
# box that bounds the plate
def transBox(box, transMat):
  # reverse the order of shape dims
  #boxShape = box[::-1]
  boxShape = box
  # Add extra element to shape array, and assign value 1
  #boxShape = boxShape + (1,)
  # preparing shape array in format 4,3 in preparation for matrix dot product transform
  # create 4 co-ordinates: origin (top left), bottom right, bottom left, top right
  # Not sure about the extra dimension, but believe that you have to have the extra dimension
  # because of the dimensions (2,3)  of the rotation/scaling matrix
  boxShape = numpy.array([[boxShape[0], boxShape[1], 1], [boxShape[2], boxShape[3], 1], [boxShape[0], boxShape[3], 1], [boxShape[2], boxShape[1], 1]])
  # transform the plate polygon
  boxPolygon = numpy.array((transMat.dot(boxShape.T)).T)
  boxPolygon = boxPolygon.astype(int)
  # convert the 4 co-ordinate plate quadrilateral to a 2 co-ordinate rectangular box
  boxOut = numpy.array([numpy.min(boxPolygon, axis=0), numpy.max(boxPolygon, axis=0)])
  return boxOut

def generate_im(char_ims, num_bg_images, enable_image_ann, enable_rand_polarity_bg):
    # Generate background image and plate image
    bg = generate_bg(num_bg_images)
    plate, plate_mask, code, charShapes = generate_plate(FONT_HEIGHT, char_ims, enable_rand_polarity_bg)
    # get the plate shape
    plateOriginalShape = numpy.shape(plate)
    # Add the origin
    plateOriginalShape = [0, 0, plateOriginalShape[1], plateOriginalShape[0]]
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.1,
                            max_scale=0.2,
                            rotation_variation=0.5,
                            scale_variation=2.0,
                            translation_variation=0.1)
    # warp the plate image and scale to the same size as the background
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    # warp the plate mask in the same way as the plate image
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    # warp the plate box in the same way as the plate image
    plate_box = transBox(plateOriginalShape, M)

    # debug: add the plate box to the image
    if enable_image_ann == True:
      cv2.rectangle(out, (plate_box[0,0],plate_box[0,1]), (plate_box[1,0],plate_box[1,1]), (0,255,0))

    # generate the character bounding boxes
    char_boxes = []
    for charShape in charShapes:
      char_box = transBox(charShape, M)
      char_boxes.append(char_box)
      if enable_image_ann == True:
        cv2.rectangle(out, (char_box[0, 0], char_box[0, 1]), (char_box[1, 0], char_box[1, 1]), (0, 255, 0))

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
    #out = cv2.resize(plate, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    # add some noise
    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds, plate_box, char_boxes


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims(enable_image_ann, enable_rand_polarity_bg):
  """
  Generate number plate images.
  :return:
      Iterable of number plate images.
  """
  variation = 1.0
  fonts, font_char_ims = load_fonts(FONT_DIR)
  num_bg_images = len(os.listdir("bgs"))
  while True:
      yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images, enable_image_ann, enable_rand_polarity_bg)



if __name__ == "__main__":
  # construct the argument parser and parse command line arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--imagePath", required=True,
                  help="path to image files")
  ap.add_argument("-x", "--xmlPath", required=True,
                  help="path to pascal VOC xml files")
  ap.add_argument("-n", "--numImages", required=True,
                  help="number of image files to generate")
  ap.add_argument("-d", "--debug", default="false",
                  help="enable debug by adding annotation to output images")
  ap.add_argument("-b", "--enable_rand_polarity_bg", default="false",
                  help="enable white on black AND black on white plates")
  args = vars(ap.parse_args())

  if args["debug"] == "true":
    enable_image_ann = True
  else:
    enable_image_ann = False

  if args["enable_rand_polarity_bg"] == "true":
    enable_rand_polarity_bg = True
  else:
    enable_rand_polarity_bg = False
    
  os.mkdir(args["imagePath"])
  os.mkdir(args["xmlPath"])
  pascalVocXml = PascalVocXml()
  im_gen = itertools.islice(generate_ims(enable_image_ann, enable_rand_polarity_bg), int(args["numImages"]))
  for img_idx, (im, c, p, plate_box, char_boxes) in enumerate(im_gen):
    fname = "{}/{:08d}_{}_{}.jpg".format(args["imagePath"],img_idx, c,
                                           "1" if p else "0")
    print (fname)
    im = (im * 255).astype(numpy.uint8)
    grey = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(fname, grey)

    # open the xml output file for writing
    fnMatch = re.search(r".*\/(.*)\..*$", fname)
    xmlPath = args["xmlPath"] + os.sep + fnMatch.group(1) + ".xml"
    xmlFile = open(xmlPath, "w")
    xmlFile.write(pascalVocXml.xmlStart(fname))

    # write the plate box info to file
    xmlFile.write(pascalVocXml.xmlBox("plate", plate_box[0,0], plate_box[0,1], plate_box[1,0], plate_box[1,1]))

    # write the char box info to file
    for char, char_box in zip(c, char_boxes):
      xmlFile.write(pascalVocXml.xmlBox(char, char_box[0, 0], char_box[0, 1], char_box[1, 0], char_box[1, 1]))

    # write final text to xml file and close
    xmlFile.write(pascalVocXml.xmlEnd())
    xmlFile.close()

