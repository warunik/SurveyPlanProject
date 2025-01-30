import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/ac/plan (100).jpg'

if not image_path:
    messagebox.showerror("Error", "No image.")

# Load the image
image = cv2.imread(image_path)

# Convert image to HSV color space for better color detection
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for yellow to green color (HSV range)
lower_yellow_green = np.array([30, 100, 100])  # Lower bound of yellow-green
upper_yellow_green = np.array([90, 255, 255])  # Upper bound of yellow-green

# Create a mask for the square based on color
mask = cv2.inRange(hsv_image, lower_yellow_green, upper_yellow_green)

# Find contours of the detected areas
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

pixel_area_mm2 = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    
    if len(approx) == 4:  # Square detected
        pts = approx.reshape(4, 2)
        
        def distance(pt1, pt2):
            return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

        side_lengths = [distance(pts[i], pts[(i+1)%4]) for i in range(4)]
        width_pixels = side_lengths[0]
        height_pixels = side_lengths[1]
        
        # Real-world dimensions in mm (from physical measurements)
        real_width_mm = real_height_mm = 15
        pixel_size_mm_w = real_width_mm / width_pixels
        pixel_size_mm_h = real_height_mm / height_pixels
        pixel_area_mm2 = pixel_size_mm_w * pixel_size_mm_h
        break

if pixel_area_mm2 is None:
    messagebox.showerror("Error", "Could not detect the square for pixel area calculation.")
else:
    # Ask the user to select the plan's scale (1:2000, 1:1000 or 1:500)
    scale_input = simpledialog.askstring("Input", "Enter the plan scale (1:2000, 1:1000 or 1:500):")

    if scale_input == "1:1000":
        scale_factor = 1000 * 1000  # Scale for 1:1000 in cm²
    elif scale_input == "1:500":
        scale_factor = 500 * 500  # Scale for 1:500 in cm²
    elif scale_input == "1:2000":
        scale_factor = 2000 * 2000  # Scale for 1:2000 in cm²
    else:
        messagebox.showerror("Error", "Invalid scale selected.")

    # Convert the image to grayscale and apply Gaussian blur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # Edge detection using Sobel operator
    grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.uint8(gradient_magnitude)

    # Thresholding to get binary edges
    _, binary_edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    # Find contours of the land boundary
    contours, _ = cv2.findContours(binary_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a blank image for the contour
    contour_image = np.zeros_like(gray_image)
    cv2.drawContours(contour_image, [largest_contour], -1, (255), 2)

    # Perform flood fill to get the filled shape
    moments = cv2.moments(largest_contour)
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])
    flood_fill_mask = np.zeros((contour_image.shape[0] + 2, contour_image.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(contour_image, flood_fill_mask, (centroid_x, centroid_y), 255)

    # Morphological opening to clean up the shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    cleaned_shape = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

    # Calculate the number of white pixels in the filled shape
    num_white_pixels = np.sum(cleaned_shape == 255)

    # Calculate the total area of the land based on the pixel area and scale
    total_land_area_mm2 = num_white_pixels * pixel_area_mm2
    total_land_area_cm2 = total_land_area_mm2 / 100  # Convert mm² to cm²
    total_land_area_scaled_cm2 = total_land_area_cm2 * scale_factor  # Scale it to 1:1000 or 1:500
    total_land_area_m2 = total_land_area_scaled_cm2 / 10000  # Convert cm² to m²

    # Ask the user how many portions to divide the land into
    num_portions = simpledialog.askinteger("Input", "To how many portions should the land get divided:")

    pixels_per_portion = num_white_pixels // num_portions
    portion_area_m2 = total_land_area_m2 / num_portions

    # Function to generate random colors
    def generate_random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Generate colors for each portion
    portion_colors = [generate_random_color() for _ in range(num_portions)]

    # Create a copy of the original image to draw the divided land
    divided_image = np.zeros_like(image)
    current_portion = 0
    colored_pixels_count = 0

    # Divide the land into portions and color them
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

    # Overlay contours and borders on the original image
    original_with_contours = image.copy()
    for portion_color in portion_colors:
        mask = cv2.inRange(divided_image, portion_color, portion_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original_image_divided = cv2.drawContours(original_with_contours, contours, -1, (255, 0, 0), 2)

    # Display the original image, land boundary contour, cleaned shape, divided image, and borders
    # cv2.imshow("Image", cv2.resize(image, (0, 0), fx=0.1, fy=0.1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow("HSV Image", cv2.resize(hsv_image, (0, 0), fx=0.1, fy=0.1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow("Image", cv2.resize(mask, (0, 0), fx=0.1, fy=0.1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow("Image", cv2.resize(approx, (0, 0), fx=0.1, fy=0.1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    plt.figure(figsize=(15, 10))

# plt.subplot(2, 2, 1)
# plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
# plt.title('Blurred Image',fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# # Second subplot: HSV Image
# plt.subplot(2, 2, 2)
# plt.imshow(cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2RGB))
# plt.title('Gradient Magnitude Image', fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# # Third subplot: Mask Image
# plt.subplot(2, 2, 3)
# plt.imshow(contour_image, cmap='gray')
# plt.title('Contour Image',fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# plt.subplot(2, 2, 4)
# plt.imshow(cv2.cvtColor(cleaned_shape, cv2.COLOR_BGR2RGB))
# plt.title('After Opening - Cleaned Shape',fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# # Display the figure
# plt.show()

# plt.subplot(2, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image',fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# # Second subplot: HSV Image
# plt.subplot(2, 2, 2)
# plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB))
# plt.title('HSV Image', fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# # Third subplot: Mask Image
# plt.subplot(2, 2, 3)
# plt.imshow(mask, cmap='gray')
# plt.title('Mask for the Square',fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# plt.subplot(2, 2, 4)
# plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
# plt.title('Gray Image',fontsize=8)
# # plt.axis('off')
# plt.tick_params(labelsize=8) 

# # Display the figure
# plt.show()

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(divided_image, cv2.COLOR_BGR2RGB))
plt.title('Colored Portions',fontsize=8)
# plt.axis('off')
plt.tick_params(labelsize=8) 

# Second subplot: HSV Image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(original_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Contours', fontsize=8)
# plt.axis('off')
plt.tick_params(labelsize=8) 

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(original_image_divided, cv2.COLOR_BGR2RGB))
plt.title('Divided Image - Final Output',fontsize=8)
# plt.axis('off')
plt.tick_params(labelsize=8) 

# Display the figure
plt.show()