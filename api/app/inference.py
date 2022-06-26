import numpy as np
from deep_utils import Box, load_pickle
from deep_utils import VggFace2TorchFaceRecognition, UltralightTorchFaceDetector
from scipy.spatial.distance import cdist


class Inference:
    def __init__(self, face_recognition_path, encoding_path, threshold, unknown="unknown", device='cpu'):
        self.face_detector = UltralightTorchFaceDetector(device=device)
        self.face_recognizer = VggFace2TorchFaceRecognition(model=face_recognition_path, device=device)
        names_encodings: dict = load_pickle(encoding_path)
        self.encodings = np.array(list(names_encodings.values()))
        self.names = list(names_encodings.keys())
        self.unknown = unknown
        self.threshold = threshold

    @staticmethod
    def preprocessing(img) -> np.ndarray:
        if type(img) is not np.ndarray:
            img = np.array(img).astype(np.uint8)
        return img

    def get_most_similar(self, features: np.ndarray, distance='euclidean'):
        p = cdist(self.encodings, features, metric=distance)
        closest_indices, closest_distances = np.argmax(p, axis=0), np.max(p, axis=0)
        results = [(self.names[index] if distance < self.threshold else self.unknown, distance) for index, distance in
                   zip(closest_indices, closest_distances)]
        return results

    def infer(self, img):
        img = self.preprocessing(img)
        faces = dict()
        face_output = self.face_detector.detect_faces(img, is_rgb=True)
        if face_output['boxes'] and len(face_output['boxes'][0]):
            img_parts = Box.get_box_img(img, face_output['boxes'])
            results = self.face_recognizer.extract_faces(img_parts, is_rgb=True)
            names_distances = self.get_most_similar(results.encodings)
            for e, ((name, distance), box) in enumerate(zip(names_distances, face_output['boxes'])):
                faces[f"face_{e:02}"] = (name, round(distance, 3), box)
        return faces
