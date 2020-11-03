
from flask import Flask, jsonify, request, session, Response
from flask_cors import CORS

import os
from werkzeug.utils import secure_filename

import cv2



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './data/'
CORS(app, expose_headers='Authorization')

# Falsk uploadfunction requirements
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@app.route('/api', methods=["GET"])
def index():
    return jsonify('API is working')


@app.route("/api/fileUpload", methods=["GET", "POST"])
def fileUpload():
    if request.method == "POST":
        target = UPLOAD_FOLDER
        if not os.path.isdir(target):
            os.mkdir(target)
        file = request.files['file']
        filename = secure_filename(file.filename)
        destination = target.join([filename])
        file.save(destination)
        session['uploadFilePath'] = destination
        return jsonify(f"A post request was made with..")

    elif request.method == "GET":
        return jsonify("API is ready for a file upload!!")

def getvideo(vidpath):
    video= cv2.VideoCapture(vidpath)
    while(video.isOpen):
        ret,frame = video.read()
        if not ret:
            yield({'currFrame':frame, 'status':200, 'anamoly':False})
        else:
            break
    video.close()

@app.route('/api/video_feed/<vidname>')
def video_feed(vidname):
    return Response(getvideo(vidname),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/setanamoly', methods=["GET"])
def setanamolies():
    with open('anamoly.txt','r') as f:
        return {'anamoly':f.readline()}

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(host="0.0.0.0", port="5000")
