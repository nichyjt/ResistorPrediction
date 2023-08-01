# This file contains logic to clean raw images cropped from YOLO or camera.
import copy
import cv2 as cv
import numpy as np

# Crop the middle section
def cropMiddle(image: cv.Mat, width_ratio=0.9, height_ratio=0.3):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the new width and crop positions for the x-axis
    crop_width = int(width * width_ratio)
    left = int((width - crop_width) / 2)
    right = left + crop_width

    # Calculate the new height and crop positions for the y-axis
    crop_height = int(height * height_ratio)
    top = crop_height
    bottom = 2 * crop_height

    # Perform the crop
    cropped_image = image[top:bottom, left:right]

    return cropped_image

def columnMedianReplace(img: cv.Mat):
    imgClean = img.copy()
    rows, cols, _ = img.shape
    for col in range(0, cols):
        colVals = imgClean[:, col, :]
        medianValue = np.median(colVals, axis=0)
        imgClean[:, col, :] = medianValue
    return imgClean

def columnAverageReplace(img: cv.Mat):
    imgClean = img.copy()
    rows, cols, _ = img.shape
    for col in range(0, cols):
        colVals = imgClean[:, col, :]
        medianValue = np.average(colVals, axis=0)
        imgClean[:, col, :] = medianValue
    return imgClean

def get_avg_brightness(img: cv.Mat):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Calculate the average brightness using mean pixel intensity
    average_brightness = cv.mean(gray_image)[0]
    return average_brightness

# Color correction
def apply_clahe(img, threshold: int):
    clipLimit = 2.0
    avg_brightness = get_avg_brightness(img)
    # print(avg_brightness)
    if avg_brightness < threshold:
        clipLimit = 12.0
    # print(clipLimit)
    # Convert the image to LAB color space
    lab_image = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    # Split the LAB image into channels
    L, A, B = cv.split(lab_image)
    # Create CLAHE object and apply it to the L channel
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    L = clahe.apply(L)

    # Merge the channels back into LAB image
    lab_image = cv.merge([L, A, B])

    # Convert the LAB image back to BGR color space
    equalized_image = cv.cvtColor(lab_image, cv.COLOR_LAB2BGR)

    return equalized_image

# Saturate images
def increase_saturation(image: cv.Mat, value=50):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # Convert image to HSV color space
    h, s, v = cv.split(hsv_image)  # Split the channels

    # Increase the saturation channel by the specified value
    s = cv.add(s, value)
    s = np.clip(s, 0, 255)  # Ensure that the pixel values are within the valid range
    hsv_image = cv.merge([h, s, v])  # Merge the channels back
    saturated_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)  # Convert back to BGR color space
    return saturated_image

def remove_background(image, threshold = 50):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)
    # Define the criteria for K-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Perform K-means clustering
    _, _, centers = cv.kmeans(pixels, 1, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    background_color = centers[0].astype(int)
    
    # Define the threshold for color similarity (adjust as needed)
    similarity_threshold = threshold
    
    # Calculate color similarity with the background color
    color_diff = np.abs(pixels - background_color)
    similarity_mask = np.all(color_diff < similarity_threshold, axis=1)

    # Replace similar colors with a placeholder color
    placeholder_color = (255, 255, 255)  # Replace with desired placeholder color
    pixels[similarity_mask] = placeholder_color
    # Reshape the modified pixels back to the original image shape
    modified_image = pixels.reshape(image.shape)
    modified_image = modified_image.astype(np.uint8)
    return modified_image

def columnMedianReplaceStep(img: cv.Mat, sz=3):
        # Get the dimensions of the image
        height, width, _ = img.shape
        # Create a copy of the image
        new_img = img.copy()
        # Iterate over every N columns
        for col in range(0, width, sz):
            # Determine the range of columns to process
            start_col = col
            end_col = min(col + sz, width)
            # Extract the region of interest (ROI)
            roi = img[:, start_col:end_col]
            # Calculate the median RGB color for the ROI
            median_color = np.median(roi, axis=(0, 1))
            median_color = np.uint8(median_color)
            # Replace all pixels in the N columns with the median color
            new_img[:, start_col:end_col] = median_color
        return new_img

def increase_brightness(image, value, threshold = 255//2):
    if get_avg_brightness(image) >= threshold:
        return image
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = np.where((255 - v) < value, 255, v + value)
    final_hsv = cv.merge((h, s, v))
    brighter_image = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return brighter_image

def increase_black_level(image, factor=0.7):
    image_float = image.astype(np.float32)
    # Define the factor to increase black level
    black_level_factor = factor
    # Scale the pixel values to increase black level
    darkened_image = image_float * black_level_factor
    # Clip the pixel values to ensure they remain within the valid range of 0-255
    darkened_image = np.clip(darkened_image, 0, 255)
    # Convert the image back to the unsigned 8-bit integer representation
    darkened_image = darkened_image.astype(np.uint8)
    return darkened_image

def estimate_image_temperature(image):
    # Load the image
    # Convert the image to LAB color space
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    # Extract the A and B channels
    _, a, b = cv.split(lab)
    # Calculate the mean values of the A and B channels
    mean_a = cv.mean(a)[0]
    mean_b = cv.mean(b)[0]
    # Define the mapping range for A and B values
    min_a = 0
    max_a = 255
    min_b = 0
    max_b = 255
    # Define the temperature range
    min_temp = 2000
    max_temp = 8000
    # Map the mean A and B values to the temperature range
    temperature = ((mean_a - min_a) / (max_a - min_a)) * (max_temp - min_temp) + min_temp
    return temperature

"""
Use this function to clean 
"""
def process_image(img: cv.Mat):
    img_copy = copy.deepcopy(img)  # Create a deep copy of the image
    img_copy = cv.GaussianBlur(img_copy, (13,13), 1.0)
    img_copy = cv.medianBlur(img_copy, 47)
    img_copy = apply_clahe(img_copy, 90)
    img_copy = increase_saturation(img_copy, 10)
    img_copy = cropMiddle(img_copy, width_ratio=0.85, height_ratio=0.40)  
    img_copy = columnAverageReplace(img_copy)
    # img_copy = columnMedianReplace(img_copy)
    img_copy = columnMedianReplaceStep(img_copy, 3)
    img_copy = columnMedianReplaceStep(img_copy, 5)
    # img_copy = columnMedianReplaceStep(img_copy, 15)
    # img_copy = remove_background(img_copy, 22)

    # Color adjust
    img_copy = increase_black_level(img_copy, 0.9)
    img_copy = increase_brightness(img_copy, 20, 200)
    return img_copy

if __name__ == "__main__":
    print("This file should not be run as main.")
    