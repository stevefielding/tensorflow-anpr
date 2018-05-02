# usage
# python genImageListForAWS.py --image_dir SJ7STAR_images/2018_03_02 --output_file image_catalog.csv
from imutils import paths
import argparse
import os

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", required=True, help="path to the input images")
ap.add_argument("-o", "--output_file", required=True, help="path to the csv output file")
args = vars(ap.parse_args())

imagePaths = paths.list_images(args["image_dir"])
csvFile = open(args["output_file"], "w")

csvFile.write("objects_to_find,image_url\n")

for (i,imagePath) in enumerate(imagePaths):
  splitPath = imagePath.split("/")
  newPath = os.path.join(splitPath[-2], splitPath[-1])
  csvFile.write("License plates and characters,{}\n".format(newPath))

csvFile.close()
