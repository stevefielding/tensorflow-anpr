from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from predict_images import predictImages
import shutil
import os
import subprocess

application = Flask(__name__)

@application.route('/upload')
def upload_render():
   return render_template('upload.html')
	
@application.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    shutil.rmtree("uploadedImages")
    os.mkdir("uploadedImages")
    #f = request.files['file']
    files = request.files.getlist("file[]")
    for f in files:
      #file = request.files.get(f)
      imageFile = "uploadedImages/" + secure_filename(f.filename)
      print("Image file: {}".format(imageFile))
      f.save(imageFile)
    # modelArg, labelsArg, imagePathArg, num_classesArg, min_confidenceArg, image_displayArg, pred_stagesArg
    objectDetectResults = predictImages ("/home/stevefielding_ca/github/tensorflow-anpr/datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb",
                      "/home/stevefielding_ca/github/tensorflow-anpr/classes/classes.pbtxt",
                      "/home/stevefielding_ca/github/tensorflow-anpr/uploadedImages", 37, 0.5, False, 2)
    return jsonify(objectDetectResults)
		
if __name__ == '__main__':
   application.run(debug = True)

