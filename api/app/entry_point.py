from endpoints import FaceDetectionRecognition
from settings import app, HOST, PORT_NUMBER, api, APP_ROOT

api.add_resource(FaceDetectionRecognition, APP_ROOT)

if __name__ == '__main__':
    app.run(HOST, port=PORT_NUMBER)
