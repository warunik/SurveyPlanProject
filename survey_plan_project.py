import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calculate_pixel_area(image_path):
    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow_green = np.array([30, 100, 100])
    upper_yellow_green = np.array([90, 255, 255]) 

    mask = cv2.inRange(hsv_image, lower_yellow_green, upper_yellow_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            side_lengths = [distance(pts[i], pts[(i+1)%4]) for i in range(4)]
            width_pixels = side_lengths[0]
            height_pixels = side_lengths[1]
            
            real_width_mm = real_height_mm = 15
            pixel_size_mm_w = real_width_mm / width_pixels
            pixel_size_mm_h = real_height_mm / height_pixels
            pixel_area_mm2 = pixel_size_mm_w * pixel_size_mm_h
            pixel_area_um2 = pixel_area_mm2 * 1e6
            
            return pixel_area_mm2 
    return None

def calculate_total_land_area(image_path, pixel_area_mm2):
    root = tk.Tk()
    root.withdraw() 

    scale_input = simpledialog.askstring("Input", "Enter the plan scale (1:2000, 1:1000 or 1:500):")

    if scale_input == "1:1000":
        scale_factor = 1000 * 1000  
    elif scale_input == "1:500":
        scale_factor = 500 * 500  
    elif scale_input == "1:2000":
        scale_factor = 2000 * 2000  
    else:
        messagebox.showerror("Error", "Invalid scale selected.")
        return

    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude)

    _, binary_edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    contour_image = np.zeros_like(gray_image)
    cv2.drawContours(contour_image, [largest_contour], -1, (255), 2)

    moments = cv2.moments(largest_contour)
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])
    flood_fill_mask = np.zeros((contour_image.shape[0] + 2, contour_image.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(contour_image, flood_fill_mask, (centroid_x, centroid_y), 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    cleaned_shape = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

    num_white_pixels = np.sum(cleaned_shape == 255)

    total_land_area_mm2 = num_white_pixels * pixel_area_mm2
    total_land_area_cm2 = total_land_area_mm2 / 100  
    total_land_area_scaled_cm2 = total_land_area_cm2 * scale_factor

    total_land_area_m2 = total_land_area_scaled_cm2 / 10000 

    num_portions = simpledialog.askinteger("Input", "To how many portions should the land get divided:")

    pixels_per_portion = num_white_pixels // num_portions
    portion_area_m2 = total_land_area_m2 / num_portions

    def generate_random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    portion_colors = [generate_random_color() for _ in range(num_portions)]

    divided_image = np.zeros_like(image)
    current_portion = 0
    colored_pixels_count = 0

    for y in range(cleaned_shape.shape[0]):
        for x in range(cleaned_shape.shape[1]):
            if cleaned_shape[y, x] == 255:
                divided_image[y, x] = portion_colors[current_portion]
                colored_pixels_count += 1

                if colored_pixels_count == pixels_per_portion:
                    current_portion += 1
                    colored_pixels_count = 0

                if current_portion == num_portions:
                    break
        if current_portion == num_portions:
            break

    original_with_contours = image.copy()
    for portion_color in portion_colors:
        mask = cv2.inRange(divided_image, portion_color, portion_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original_image_divided = cv2.drawContours(original_with_contours, contours, -1, (255, 0, 0), 2)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # plt.subplot(2, 3, 2)
    # plt.imshow(contour_image, cmap='gray')
    # plt.title('Land Boundary Contour (Sobel)')
    # plt.axis('off')

    # plt.subplot(2, 3, 3)
    # plt.imshow(cleaned_shape, cmap='gray')
    # plt.title('Cleaned Shape')
    # plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(divided_image, cv2.COLOR_BGR2RGB))
    plt.title('Colored Portions')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(original_image_divided, cv2.COLOR_BGR2RGB))
    plt.title('Divided Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, f'Total Land Area: {total_land_area_m2:.2f} m²', fontsize=14)
    plt.text(0.1, 0.6, f'Portion Area: {portion_area_m2:.2f} m²', fontsize=14)
    plt.axis('off')

    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()

    image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/ac/plan (100).jpg'

    if not image_path:
        messagebox.showerror("Error", "No image.")
        return

    pixel_area_mm2 = calculate_pixel_area(image_path)

    if pixel_area_mm2 is not None:
        calculate_total_land_area(image_path, pixel_area_mm2)
    else:
        messagebox.showerror("Error", "Could not detect the square for pixel area calculation.")

if __name__ == "__main__":
    main()
