# ----------------------------------------------

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

# ----------------------------------------------

def get_image_complexity(image, threshold1=280, threshold2=900):
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(image, None)
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

def grab_cut(image, blur_kernel=(7, 7), bb_size=10, iterations=3):
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

def contour_miss(x, y, w, h, image_shape, num_keypoints):
    image_width, image_height = image_shape
    top_left = x < 3 and y < 3
    top_right = x + w > image_width - 3 and y < 3
    bottom_left = x < 3 and y + h > image_height - 3
    bottom_right = x + w > image_width - 3 and y + h > image_height - 3
    if (top_left or top_right or bottom_left or bottom_right) and (num_keypoints == 0 or num_keypoints > 250):
        return True
    return False

def filter_contours(contours, original_image, min_area=400, min_length=15, ratio_threshold=0.4):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    removed = []
    sift = cv2.SIFT_create()
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
        roi = original_image[y:y+h, x:x+w]
        keypoints, _ = sift.detectAndCompute(roi, None)
        if inside_other_box or w * h < min_area or w < min_length or h < min_length or contour_miss(x, y, w, h, original_image.shape[:2], len(keypoints)):
            removed.append(i)
    contours = [contour for i, contour in enumerate(contours) if i not in removed]
    return contours

# ----------------------------------------------


def similar_colors(color1, color2):

    TOLERANCE_LAB = 12.5

    #display_color(color1, color1)
    #display_color(color2, color2)

    # Converting colors from BGR to Lab color space
    #color bgr to rgb
    color1_RGB = color1[::-1]
    color2_RGB = color2[::-1]

    color1_RGB = np.uint8([[[color1_RGB[0],color1_RGB[1],color1_RGB[2]]]])
    color2_RGB = np.uint8([[[color2_RGB[0], color2_RGB[1], color2_RGB[2]]]])

    color1_LAB = rgb2lab(color1_RGB)
    color2_LAB = rgb2lab(color2_RGB)

    # Calculating the difference between colors using the CIEDE2000 formula
    diff = deltaE_ciede2000(color1_LAB, color2_LAB)
    print("Difference:", float(diff))

    similar = False
    if(diff <= TOLERANCE_LAB):
        similar = True
    return similar



def count_unique_colors(object_colors):
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

# ----------------------------------------------

def display_color(color, index):
    height, width = 100, 100
    color_image = np.zeros((height, width, 3), np.uint8)
    color_image[:, :] = color
    cv2.imshow(str(index), color_image)

# ----------------------------------------------

def calculate_lego_color(roi):
    lego_color = central_area_color(roi)
    return lego_color

def central_area_color(roi):
    height, width, _ = roi.shape
    central_area = roi[height//4:3*height//4, width//4:3*width//4]
    central_color = median_color(central_area)
    
    return central_color

def median_color(roi):
    median_color = np.median(roi, axis=(0, 1)).astype(int)
    return median_color


def mean_color(roi):
    mean_color = np.mean(roi, axis=(0, 1)).astype(int)
    return mean_color




# ----------------------------------------------