from argparse import ArgumentParser
import requests
import cv2
from deep_utils import BinaryUtils, CVUtils, Box, split_extension
import os

parser = ArgumentParser()
parser.add_argument("--endpoint", default="http://127.0.0.1:8000/face_recognition")
parser.add_argument("--img_address", default='../dataset/test/friends.jpg')
parser.add_argument("--show", action="store_true", help="shows the output")
args = parser.parse_args()


def infer(endpoint, img_address) -> dict:
    image = cv2.imread(img_address)[..., ::-1]
    byte_img = BinaryUtils.img_to_b64(image)
    data = {'image': byte_img}
    response = requests.post(endpoint, json=data, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    return response.json()


def _get_file_content(file_path: str, file_content_type: str = 'multipart/form-data') -> tuple:
    file_name = os.path.basename(file_path)
    file_content = open(file_path, 'rb')
    return file_name, file_content, file_content_type


if __name__ == "__main__":
    output = infer(args.endpoint, args.img_address)
    img = cv2.imread(args.img_address)
    for name, distance, box in output.values():
        img = Box.put_box_text(img, box, label=f"{name}-{distance}")
    cv2.imwrite(split_extension(args.img_address, suffix="_res"), img)
    if args.show:
        CVUtils.show_destroy_cv2(img)
