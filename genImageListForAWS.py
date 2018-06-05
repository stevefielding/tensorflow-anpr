# usage
# python genImageListForAWS.py --image_dir SJ7STAR_images/2018_03_02 --output_file image_catalog.csv
# Generate a csv file sthat can be uploaded to MTURK
from imutils import paths
import argparse
import os

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", required=True, help="path to the input images")
ap.add_argument("-o", "--output_file", required=True, help="path to the csv output file")
args = vars(ap.parse_args())

# get the list of input images, and create the csv output file
imagePaths = paths.list_images(args["image_dir"])
csvFile = open(args["output_file"], "w")

# Add the header to the csv file
csvFile.write("objects_to_find,image_url\n")

# Loop over all the input images, adding each image path to the csv file
for (i,imagePath) in enumerate(imagePaths):
  splitPath = imagePath.split("/")
  newPath = os.path.join(splitPath[-2], splitPath[-1])
  csvFile.write("License plates and characters,{}\n".format(newPath))

csvFile.close()
