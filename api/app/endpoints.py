from flask import jsonify
from flask_restful import Resource
from settings import app
from deep_utils import b64_to_img
from PIL import Image
import numpy as np


class FaceDetectionRecognition(Resource):
    @staticmethod
    def post():
        args = app.config['PARSER'].parse_args()
        contents = args['image']
        if app.config['POST_TYPE'] == 'JSON':
            img = b64_to_img(contents)
        else:
            img = np.array(Image.open(contents))
        res = app.config['inference'].infer(img)
        return jsonify(res)

    @staticmethod
    def get():
        """
        Bug test
        :return: some text
        """
        return jsonify({"Status": "Alive"})
