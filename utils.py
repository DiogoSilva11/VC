# ----------------------------------------------

import cv2
import numpy as np

# ----------------------------------------------

def get_image_complexity(gray, threshold1=280, threshold2=900):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    num_keypoints = len(keypoints)
    if num_keypoints < threshold1:
        complexity = 1
    elif num_keypoints < threshold2:
        complexity = 2
    else:
        complexity = 3
    return complexity, num_keypoints

# ----------------------------------------------

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

def background_median(hsv, hue_tolerance=20, saturation_tolerance=90, value_tolerance=90, blur_kernel=(5, 5)):
    background_color = median_background(hsv)
    lower_background, upper_background = limits(background_color, hue_tolerance, saturation_tolerance, value_tolerance)
    gaussian = cv2.blur(hsv, blur_kernel)
    gray_mask = cv2.inRange(gaussian, lower_background, upper_background)
    non_gray_mask = cv2.bitwise_not(gray_mask)
    non_gray_mask = cv2.erode(non_gray_mask, None, iterations=3)
    return non_gray_mask

# ----------------------------------------------

def kmeans_segments(image, k=50):
    reshaped_image = image.reshape((-1, 3))
    reshaped_image = np.float32(reshaped_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape((image.shape))
    return result

def canny_edges(gray, blur_kernel=(7, 7), low_threshold=100, high_threshold=200):
    gaussian = cv2.GaussianBlur(gray, blur_kernel, 0)
    canny = cv2.Canny(gaussian, low_threshold, high_threshold)
    return canny

# ----------------------------------------------

def grab_cut(image, blur_kernel=(7, 7), bb_size=50, iterations=5):
    gaussian = cv2.GaussianBlur(image, blur_kernel, 0)
    mask = np.zeros(gaussian.shape[:2], np.uint8)
    bb = (bb_size, bb_size, gaussian.shape[1] - bb_size, gaussian.shape[0] - bb_size)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    (mask, bgModel, fgModel) = cv2.grabCut(gaussian, mask, bb, bgModel, fgModel, iterations, cv2.GC_INIT_WITH_RECT)
    output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    output_mask = (output_mask * 255).astype("uint8")
    return output_mask

# ----------------------------------------------

def calculate_intersection_area(square1, square2):
    x1, y1, w1, h1 = square1
    x2, y2, w2, h2 = square2
    x_intersect = max(x1, x2)
    y_intersect = max(y1, y2)
    w_intersect = min(x1 + w1, x2 + w2) - x_intersect
    h_intersect = min(y1 + h1, y2 + h2) - y_intersect
    if w_intersect <= 0 or h_intersect <= 0:
        return 0
    intersection_area = w_intersect * h_intersect
    return intersection_area

def is_intersected(square1, square2, ratio_threshold):
    area_square1 = square1[2] * square1[3]
    area_square2 = square2[2] * square2[3]
    area_intersection = calculate_intersection_area(square1, square2)
    ratio1 = area_intersection / area_square1
    ratio2 = area_intersection / area_square2
    if ratio1 > ratio_threshold and ratio1 > ratio2:
        return True
    else:
        return False

def filter_contours(contours, min_area=400, min_length=15, ratio_threshold=0.5):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    removed = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        inside_other_box = False
        for j in range(i+1, len(contours)):
            prev_x, prev_y, prev_w, prev_h = cv2.boundingRect(contours[j])
            if is_intersected((x, y, w, h), (prev_x, prev_y, prev_w, prev_h), ratio_threshold):
                inside_other_box = True
                break
            elif is_intersected((prev_x, prev_y, prev_w, prev_h),(x,y,w,h), ratio_threshold):
                removed.append(j)
        if inside_other_box or w * h < min_area or w < min_length or h < min_length:
            removed.append(i)
    contours = [contour for i, contour in enumerate(contours) if i not in removed]
    return contours

# ----------------------------------------------

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

# ----------------------------------------------