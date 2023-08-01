import math
import cv2 as cv
import numpy as np
import pandas as pd
import os
import json
import datetime

# Constants
FILEPATH_LABEL_INPUT = "../data/labels_json"
FILEPATH_IMG_DATA = "../data/labelling"
FILEPATH_OUTPUT = "../data/training"
FILEEXT_JSON = ".json"
FILEEXT_CSV = ".csv"
DELIMITER_DASH = "-"
today = datetime.date.today()
formatted_date = today.strftime("%Y-%m-%d")
FILENAME_OUTPUT = "labels_" + formatted_date

COLORSPACE_BGR = "BGR"
COLORSPACE_LAB = "LAB"
COLORSPACE_HSV = "HSV"
COLORSPACE_YCRCB = "YCRCB"

LABEL_BLACK = "black"
LABEL_BROWN = "brown"
LABEL_RED = "red"
LABEL_ORANGE = "orange"
LABEL_YELLOW = "yellow"
LABEL_GREEN = "green"
LABEL_BLUE = "blue"
LABEL_PURPLE = "purple"
LABEL_GREY = "grey"
LABEL_WHITE = "white"
LABEL_GOLD = "gold"
LABEL_SILVER = "silver"
LABEL_NOISE = "noise"


# Convenience wrappers
class PointLabel:
    def __init__(self, name:str, label: str, BGR:tuple, LAB: tuple, HSV: tuple, YCRCB: tuple) -> None:
        self.name = name
        self.label = label
        self.BGR = BGR
        self.LAB = LAB
        self.HSV = HSV
        self.YCRCB = YCRCB

# IO
# Loads JSON label jsons and dumps them  
def load_label_data():
    # Iterate over each file in the directory
    json_data = []
    for filename in os.listdir(FILEPATH_LABEL_INPUT):
        if filename.endswith(FILEEXT_JSON):
            # print(filename)
            # if filename == "debug.json":
            filepath = os.path.join(FILEPATH_LABEL_INPUT, filename)
            # Open the file and parse the JSON data
            with open(filepath) as file:
                # WARNING: no error checking
                data = json.load(file)
                json_data.append(data)
    return json_data

def save_label_data(df: pd.DataFrame):
    df.to_csv(FILEPATH_OUTPUT + "/" + FILENAME_OUTPUT + FILEEXT_CSV, index=True, index_label="id")
    
# Loads the image file and automatically converts to LAB
def load_img_file(filename: str):
    # Assume that the filename has extension already
    img = cv.imread(FILEPATH_IMG_DATA + "/" + filename)
    return img

def get_colorspace_values(filename:str, row: int, col: int, colorspace):
    img = load_img_file(filename)
    assert row <= img.shape[0] and col <= img.shape[1]
    assert row >= 0 and col >= 0
    if colorspace == COLORSPACE_BGR:
        # assume image is already in BGR
        r, g, b = img[row, col]
        return (r, g, b)
    elif colorspace == COLORSPACE_LAB:
        img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        return get_LAB_values(img, row, col)
    elif colorspace == COLORSPACE_HSV:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
        return get_HSV_values(img, row, col)
    elif colorspace == COLORSPACE_YCRCB:
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        return get_YCRCB_VALUES(img, row, col)

    raise "WARNING: Invalid colorspace argument"

# returns a tuple of (L, a, b) where L*a*b is the colorspace
def get_LAB_values(img:cv.Mat, row: int, col: int):
    L = int(img[row, col, 0] * 100 / 255)
    a = img[row, col, 1] - 128
    b = img[row, col, 2] - 128
    return (L, a, b)

def get_HSV_values(img:cv.Mat, row: int, col: int):
    h, s, v = img[row, col]
    return (h,s,v)

def get_YCRCB_VALUES(img:cv.Mat, row: int, col: int):
    y, cr, cb = img[row, col]
    return (y, cr, cb)

# Cleans the filename to return the original filename
# e.g. "\/data\/upload\/3\/d52ffd44-black-brown-9.png"
#      gets transformed into "black-brown-9.png"
def parse_filename(name: str):
    # Split by forward slash and get the last string
    filename = name.split("/")[-1]
    # We are OK with file extensions
    filename = filename.split(DELIMITER_DASH, 1)[1]
    return filename
    
# Takes in "kp-1" keypoint object for further unmarshalling
# Returns a list of PointLabel
def parse_keypoints(keypoints, filename):
    pointlabels = []
    for point in keypoints:
        shape_rows = point["original_height"]
        shape_cols = point["original_width"]
        # x, y are percentages relative to the shapes
        x = point["x"] 
        y = point["y"]
        # Calculate PointLabel params
        assert len(point["keypointlabels"]) >= 1
        row = math.ceil(y * shape_rows / 100)
        col = math.ceil(x * shape_cols / 100)
        label = point["keypointlabels"][0]
        B, G, R = get_colorspace_values(filename, row, col, COLORSPACE_BGR)
        L, a, b = get_colorspace_values(filename, row, col, COLORSPACE_LAB)
        H, S, V = get_colorspace_values(filename, row, col, COLORSPACE_HSV)
        Y, Cr, Cb = get_colorspace_values(filename, row, col, COLORSPACE_YCRCB)
        pointlabels.append(PointLabel(filename, label, (B,G,R), (L,a,b), (H,S,V), (Y,Cr,Cb)))
    return pointlabels

# Converts a list of PointLabel into a nice dataframe
def df_from_image_labels(datapoints: list):
    DATAFRAME_HEADERS = ["name", "label",
                            "L", "a", "b",
                            "B", "G", "R",
                            "H", "S", "V",
                            "Y", "Cr", "Cb"]
    data = []
    for point_label in datapoints:
        name = point_label.name
        label = point_label.label
        L, a, b = point_label.LAB
        B, G, R = point_label.BGR
        H, S, V = point_label.HSV
        Y, Cr, Cb = point_label.YCRCB
        data.append([name, label, L, a, b, B, G, R, H, S, V, Y, Cr, Cb])
    df = pd.DataFrame(data, columns=DATAFRAME_HEADERS)
    return df

# Takes in a labelstudio json file and process it
# @return a list of PointLabel tuples
def parse_data(json_data):
    # json data from labelstudio has this format:
    # [[{ metadata, kp-1: [{...}] }]]
    datapoints = []
    for overall_data in json_data:
        for img_data in overall_data:
            img_name = img_data["img"]
            img_name = parse_filename(img_name)
            keypoint_labels = img_data["kp-1"]
            datapoints.extend(parse_keypoints(keypoint_labels, img_name))
    return datapoints

            

# Main control function
# Abstracted into a function for re-use if necessary
def labelparser():
    json_data = load_label_data()
    datapoints = parse_data(json_data)
    df = df_from_image_labels(datapoints)
    save_label_data(df)

# Data will be stored in a csv file with the following schema 
#  | id | filename_w_path | L | A | B | Class |
# NOTE: We MUST use LAB to get better color space segmentation regardless of luminances

if __name__ == "__main__":
    print("Creating csv file from labelled data...")
    labelparser()
    print("Done!")
