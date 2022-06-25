from deep_utils import MTCNNTorchFaceDetector, VggFace2TorchFaceRecognition

dataset_dir = "dataset/people"
face_detector = MTCNNTorchFaceDetector()
face_recognizer = VggFace2TorchFaceRecognition()

if __name__ == '__main__':
    face_detector.detect_crop_dir_of_dir(dataset_dir, get_biggest=True)
    face_recognizer.extract_dir_of_dir(dataset_dir, get_mean=True)

