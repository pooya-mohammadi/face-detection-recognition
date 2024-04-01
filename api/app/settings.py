import os
from werkzeug.datastructures import FileStorage
from inference import Inference
from flask import Flask
from flask_restful import Api, reqparse

# define the app and the api variables
APP_NAME = os.getenv("APP_NAME", 'face_recognition')
APP_ROOT = os.getenv('APP_ROOT', '/face_recognition')
HOST = os.getenv("HOST", "127.0.0.1")
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 8000))
FACE_RECOGNITION_PATH = os.getenv('FACE_RECOGNITION_PATH', "weights/senet50_ft_weight.pkl")
FACE_RECOGNITION_MODEL_NAME = os.getenv('senet50', "senet50")
ENCODINGS_PATH = os.getenv("ENCODINGS_PATH", "weights/people.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.0175"))
POST_TYPE = os.getenv('POST_TYPE', "JSON")
DEVICE = os.getenv("DEVICE", 'cpu')
DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "euclidean")
# load the model
inference = Inference(FACE_RECOGNITION_PATH, ENCODINGS_PATH,
                      threshold=THRESHOLD, device=DEVICE,
                      face_recognition_model_name=FACE_RECOGNITION_MODEL_NAME,
                      distance_metric=DISTANCE_METRIC)

app = Flask(APP_NAME)
api = Api(app)
# global variables
app.config['inference'] = inference
app.config['POST_TYPE'] = POST_TYPE
app.config['PARSER'] = reqparse.RequestParser()

if POST_TYPE == 'JSON':
    app.config['PARSER'].add_argument('image',
                                      required=True,
                                      help='provide an image as base64')
elif POST_TYPE == 'FORM':
    # go for forms
    app.config['PARSER'].add_argument('image',
                                      type=FileStorage,
                                      location='files',
                                      required=True,
                                      help='provide a file')
