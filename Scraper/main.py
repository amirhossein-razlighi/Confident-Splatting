import streetview
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


def open_image_with_cv2(filepath):
    try:
        # Read the image using OpenCV
        cv_image = cv2.imread(filepath)
        if cv_image is None:
            raise ValueError("Image not found or unable to read")
        # Convert the image from BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Convert the OpenCV image to a PIL image
        pil_image = Image.fromarray(cv_image)
        return pil_image
    except Exception as e:
        print(f"Error reading image {filepath} with OpenCV: {e}")
        return None


if __name__ == "__main__":
    tile_width = 512
    tile_height = 512
    panorama = Image.new("RGB", (26 * tile_width, 13 * tile_height))
    tiles = list(os.walk("temp"))[0][2]
    directory = "temp"
    for fname in tiles:
        filepath = os.path.join(directory, fname)
        tile = open_image_with_cv2(filepath)
        if tile is None:
            print(fname)
            break
        splitted = fname.split("_")[1].split("x")
        x, y = int(splitted[0]), int(splitted[1].split(".")[0])
        panorama.paste(im=tile, box=(x * tile_width, y * tile_height))
    panorama.show()
