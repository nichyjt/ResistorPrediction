# This script does data preprocessing.
# It consumes the images in "./data/raw/", preprocesses it and dumps them to "./data/labelling"

import cv2 as cv
import numpy as np
import os
import datacleaner

# Constants
FILEPATH_RAW = "../data/raw"
FILEPATH_PROCESSED = "../data/labelling"
FILEEXT_JPG = ".jpg"

# Models
class Resistor:
    # input colors Left 2 Right 
    def __init__(self, filename: str, img: cv.Mat):
        self.name = filename
        self.img = img

# --- Functions ---

# IO
def load_data():
    files = os.listdir(FILEPATH_RAW)
    data = [file for file in files if file.endswith(FILEEXT_JPG)]
    raw_data = []
    for filename in data:
        image_path = os.path.join(FILEPATH_RAW, filename)
        image = cv.imread(image_path)
        if image is not None:
            raw_data.append(Resistor(filename, image))
        else:
            print("Error loading", filename)
    return raw_data

def save_data(processed_data):
    for data in processed_data:
        save_path = os.path.join(FILEPATH_PROCESSED, data.name)
        ok = cv.imwrite(save_path, data.img)
        if not ok:
            print("Error writing image")

# Apply the preprocessing to every image file
def process_data(raw_data):
    # Keep separate list (dont care memory efficiency for now)
    processed_data = []
    for resistor in raw_data:
        # print(resistor.name)    
        processed_img = datacleaner.process_image(resistor.img)
        print(resistor.name, datacleaner.estimate_image_temperature(processed_img))
        processed_data.append(Resistor(resistor.name, processed_img))
    return processed_data

if __name__ == "__main__":
    print("Preprocessing images from data/raw ...")
    raw_data = load_data()
    processed_data = process_data(raw_data)
    save_data(processed_data)
    print("Done!")
    