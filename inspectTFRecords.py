import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import argparse

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--filePath", required=True,
  help="path to Records file")
args = vars(ap.parse_args())

file = args["filePath"]
fileNum=1
for example in tf.python_io.tf_record_iterator(file):
  jsonMessage = MessageToJson(tf.train.Example.FromString(example))
  with open("temp/image_{}".format(fileNum),"w") as text_file:
    print(jsonMessage,file=text_file)
  fileNum+=1

