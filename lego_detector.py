import cv2
import os
import json
import numpy as np

def similar_colors(color1, color2, tolerance):
    distance = np.linalg.norm(color1 - color2)
    return distance <= tolerance

def count_unique_colors(object_colors, tolerance=50):
    unique_colors = []
    for color in object_colors:
        is_unique = True
        for existing_color in unique_colors:
            if similar_colors(color, existing_color, tolerance):
                is_unique = False
                break
        if is_unique:
            unique_colors.append(color)
    return len(unique_colors)

def is_inside(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return x2 <= x1 and y2 <= y1 and x2 + w2 >= x1 + w1 and y2 + h2 >= y1 + h1

def median_background(hsv):
    hsv_flattened = hsv.reshape((-1, 3))
    hsv_flattened = hsv_flattened[hsv_flattened[:, 2] > 0]
    background_color = np.median(hsv_flattened, axis=0)
    return background_color

def limits(background_color, hue_tolerance, saturation_tolerance, value_tolerance):
    lower_limit_0 = max(background_color[0] - hue_tolerance, 0)
    upper_limit_0 = min(background_color[0] + hue_tolerance, 180)
    lower_limit_1 = max(background_color[1] - saturation_tolerance, 0)
    upper_limit_1 = min(background_color[1] + saturation_tolerance, 255)
    lower_limit_2 = max(background_color[2] - value_tolerance, 0)
    upper_limit_2 = min(background_color[2] + value_tolerance, 255)
    lower_background = np.array([lower_limit_0, lower_limit_1, lower_limit_2])
    upper_background = np.array([upper_limit_0, upper_limit_1, upper_limit_2])
    return lower_background, upper_background

def background_median(hsv):
    background_color = median_background(hsv)
    hue_tolerance = 105
    saturation_tolerance = 105
    value_tolerance = 105
    lower_background, upper_background = limits(background_color, hue_tolerance, saturation_tolerance, value_tolerance)
    blur = cv2.GaussianBlur(hsv, (9, 9), 0)
    gray_mask = cv2.inRange(blur, lower_background, upper_background)
    non_gray_mask = cv2.bitwise_not(gray_mask)
    non_gray_mask = cv2.erode(non_gray_mask, None, iterations=3)
    return non_gray_mask

def kmeans_segments(image):
    reshaped_image = image.reshape((-1,1))
    reshaped_image = np.float32(reshaped_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 50
    _, label, center = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape((image.shape))
    return result

def canny_edges(gray):
    gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
    #kmeans = kmeans_segments(gaussian)
    canny = cv2.Canny(gaussian, 100, 200)
    return canny

def process_image(image_path):
    original_image = cv2.imread(image_path)
    image = cv2.resize(original_image, (0, 0), fx=0.15, fy=0.15)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    bg_median = background_median(hsv)
    #canny = canny_edges(gray)

    #combined_mask = cv2.bitwise_or(bg_median, canny)

    contours, _ = cv2.findContours(bg_median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    removed = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        inside_other_box = False
        for j in range(i+1, len(contours)):
            prev_x, prev_y, prev_w, prev_h = cv2.boundingRect(contours[j])
            if is_inside((x, y, w, h), (prev_x, prev_y, prev_w, prev_h)):
                inside_other_box = True
                break
        if not inside_other_box and w * h > 400:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        else:
            removed.append(i)
    contours = [contour for i, contour in enumerate(contours) if i not in removed]

    detections = []
    colors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        detections.append({"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h})
        roi = original_image[y:y+h, x:x+w]
        medium_rgb = np.mean(roi, axis=(0, 1)).astype(int)
        colors.append(medium_rgb)
    unique_colors = count_unique_colors(colors)

    print(image_path)
    cv2.imshow(image_path, image)
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
