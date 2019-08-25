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
    f = request.files['file']
    imageFile = "uploadedImages/" + secure_filename(f.filename)
    shutil.rmtree("uploadedImages")
    os.mkdir("uploadedImages")
    f.save(imageFile)
    # modelArg, labelsArg, imagePathArg, num_classesArg, min_confidenceArg, image_displayArg, pred_stagesArg
    objectDetectResults = predictImages ("/home/steve/github/tensorflow-anpr/datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb",
                      "/home/steve/github/tensorflow-anpr/classes/classes.pbtxt",
                      "/home/steve/github/tensorflow-anpr/uploadedImages", 37, 0.5, False, 2)
    return jsonify(objectDetectResults)
		
if __name__ == '__main__':
   application.run(debug = True)

