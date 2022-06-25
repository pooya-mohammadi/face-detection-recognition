from deep_utils import MTCNNTorchFaceDetector

dataset_dir = "dataset/people"

face_detector = MTCNNTorchFaceDetector()

if __name__ == '__main__':
    face_detector.detect_crop_dir_of_dir(dataset_dir, get_biggest=True)
