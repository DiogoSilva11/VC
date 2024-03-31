import cv2
import os
import json
import numpy as np


def count_unique_colors(object_colors, tolerance=50):
    def similar_colors(color1, color2):
        distance = np.linalg.norm(color1 - color2)
        return distance <= tolerance

    unique_colors = []

    for color in object_colors:
        is_unique = True
        for existing_color in unique_colors:
            if similar_colors(color, existing_color):
                is_unique = False
                break
        if is_unique:
            unique_colors.append(color)


    return len(unique_colors)

def process_image(image_path):
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (0, 0), fx=0.15, fy=0.15)

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    gaussian = cv2.GaussianBlur(original_image, (7, 7), 0)

    canny = cv2.Canny(gaussian, 100, 200)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    colors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        detections.append({"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h})
        roi = original_image[y:y+h, x:x+w]
        medium_rgb = np.mean(roi, axis=(0, 1)).astype(int)
        colors.append(medium_rgb)

    unique_colors = count_unique_colors(colors)

    cv2.imshow(image_path, original_image)

    print(f"Image path: {image_path}")
    print(f"Original Colors: {len(colors)}")
    print(f"Unique colors: {unique_colors}")
    print("\n\n")

    return len(contours), detections, unique_colors

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
