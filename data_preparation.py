from argparse import ArgumentParser
from deep_utils import VggFace2TorchFaceRecognition, UltralightTorchFaceDetector

parser = ArgumentParser()
parser.add_argument("--dataset_dir", default="dataset/people", help="path to the dataset")
args = parser.parse_args()

face_detector = UltralightTorchFaceDetector()
face_recognizer = VggFace2TorchFaceRecognition(model="api/app/weights/senet50_ft_weight.pkl")

if __name__ == '__main__':
    face_detector.detect_crop_dir_of_dir(args.dataset_dir, get_biggest=True)
    face_recognizer.extract_dir_of_dir(args.dataset_dir, get_mean=True)
