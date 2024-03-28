import cv2
import os
import json

def process_image(image_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (0, 0), fx=0.15, fy=0.15)

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    gaussian = cv2.GaussianBlur(original_image, (7, 7), 0)

    canny = cv2.Canny(gaussian, 100, 200)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        detections.append({"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h})

    return len(contours), detections

if __name__ == "__main__":
    images_dir = './samples'
    input_file = 'input.json'
    output_file = 'output.json'

    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    image_paths = data['image_files']
    results = []

    for image_path in image_paths:
        num_detections, detected_objects = process_image(os.path.join(images_dir, image_path))
        results.append({
            "file_name": image_path,
            "num_colors": 0,
            "num_detections": num_detections,
            "detected_objects": detected_objects
        })

    output_data = {"results": results}
    with open(output_file, 'w') as json_out:
        json.dump(output_data, json_out, indent=4)
