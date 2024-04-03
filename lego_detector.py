import cv2
import os
import json
from utils import *


def process_image(image_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (0, 0), fx = 0.15, fy = 0.15)
    image = original_image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    complexity, _ = get_image_complexity(gray)

    hue_tolerance = 20
    saturation_tolerance = 90
    value_tolerance = 120
    blur_kernel = (5, 5)
    if complexity == 2:
        hue_tolerance = 5
        saturation_tolerance = 90
        value_tolerance = 120
        blur_kernel = (5, 5)
    elif complexity == 3:
        hue_tolerance = 20
        saturation_tolerance = 110
        value_tolerance = 140
        blur_kernel = (11, 11)

    bg_median_mask = background_median(hsv, hue_tolerance, saturation_tolerance, value_tolerance, blur_kernel)

    blur_kernel = (7, 7)
    low_threshold = 40
    high_threshold = 120
    if complexity == 2:
        blur_kernel = (5, 5)
        low_threshold = 85
        high_threshold = 180
    elif complexity == 3:
        blur_kernel = (11, 11)
        low_threshold = 90
        high_threshold = 200
        
    canny_mask = canny_edges(gray, blur_kernel, low_threshold, high_threshold)

    final_mask = cv2.bitwise_or(bg_median_mask, canny_mask)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours)
    detections = []
    colors = []
    color_index = 0
    for contour in contours:
        color_index += 1
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        detections.append({"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h})
        roi = original_image[y:y+h, x:x+w]
        lego_color = calculate_lego_color(roi)
        colors.append(lego_color)
    unique_colors = count_unique_colors(colors)

    print(image_path)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return len(detections), detections, unique_colors


if __name__ == "__main__":
    images_dir = './samples'
    input_file = 'input.json'
    output_file = 'output.json'

    with open(input_file, 'r') as json_file:
        data = json.load(json_file)
    image_paths = data['image_files']
    results = []
    for image_path in image_paths:
        num_detections, detected_objects, unique_colors = process_image(os.path.join(images_dir, image_path))
        results.append({
            "file_name": image_path,
            "num_colors": unique_colors,
            "num_detections": num_detections,
            "detected_objects": detected_objects
        })
        
    output_data = {"results": results}
    with open(output_file, 'w') as json_out:
        json.dump(output_data, json_out, indent=4)
    print("Results saved to", output_file)
