import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# First code for pixel area calculation (no changes here)
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calculate_pixel_area(image_path):
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

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4:  # Square detected
            pts = approx.reshape(4, 2)
            side_lengths = [distance(pts[i], pts[(i+1)%4]) for i in range(4)]
            width_pixels = side_lengths[0]
            height_pixels = side_lengths[1]
            
            # Real-world dimensions in mm (from physical measurements)
            real_width_mm = real_height_mm = 15
            pixel_size_mm_w = real_width_mm / width_pixels
            pixel_size_mm_h = real_height_mm / height_pixels
            pixel_size_mm = (pixel_size_mm_w + pixel_size_mm_h) / 2
            pixel_area_mm2 = pixel_size_mm ** 2
            # pixel_area_mm2 = (15*15)/(width_pixels*height_pixels)
            pixel_area_um2 = pixel_area_mm2 * 1e6
            
            print(f"Pixel size: {pixel_size_mm:.6f} mm")
            print(f"Pixel area: {pixel_area_mm2:.6f} mm^2")
            print(f"Pixel area: {pixel_area_um2:.2f} μm^2")
            
            return pixel_area_mm2  # Return the area of a single pixel in mm²
    return None

# Second code for pixel counting and land area calculation with scale and portions
def calculate_total_land_area(image_path, pixel_area_mm2):
    # Ask the user to select the plan's scale (1:1000 or 1:500)
    scale_input = input("Enter the plan scale (1:1000 or 1:500): ").strip()

    if scale_input == "1:1000":
        scale_factor = 1000 * 1000  # Scale for 1:1000 in cm²
    elif scale_input == "1:500":
        scale_factor = 500 * 500  # Scale for 1:500 in cm²
    else:
        print("Invalid scale selected.")
        return

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale and apply Gaussian blur
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    cleaned_shape = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

    # Calculate the number of white pixels in the filled shape
    num_white_pixels = np.sum(cleaned_shape == 255)
    print('Number of white pixels:', num_white_pixels)

    # Calculate the total area of the land based on the pixel area and scale
    total_land_area_mm2 = num_white_pixels * pixel_area_mm2
    total_land_area_cm2 = total_land_area_mm2 / 100  # Convert mm² to cm²
    total_land_area_scaled_cm2 = total_land_area_cm2 * scale_factor  # Scale it to 1:1000 or 1:500

    total_land_area_m2 = total_land_area_scaled_cm2 / 10000  # Convert cm² to m²
    print(f"Total land area at scale {scale_input}: {total_land_area_m2:.2f} m²")

    # Ask the user how many portions to divide the land into
    num_portions = int(input("To how many portions should the land get divided: "))
    pixels_per_portion = num_white_pixels // num_portions
    print(f'Number of pixels per portion: {pixels_per_portion}')

    # Calculate and display the area of each portion
    portion_area_m2 = total_land_area_m2 / num_portions
    print(f'Area of each portion: {portion_area_m2:.2f} m²')

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
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(contour_image, cmap='gray')
    plt.title('Land Boundary Contour (Sobel)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cleaned_shape, cmap='gray')
    plt.title('Cleaned Shape')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(divided_image, cmap='gray')
    plt.title('Divided Land Image')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(original_image_divided, cmap='gray')
    plt.title('Original Image with Borders')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'Total Land Area: {total_land_area_m2:.2f} m²', fontsize=14)
    plt.text(0.1, 0.6, f'Portion Area: {portion_area_m2:.2f} m²', fontsize=14)
    plt.axis('off')

    plt.show()

# File path to the image
image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/plan (17).jpg'

# 1. Calculate the pixel area in mm²
pixel_area_mm2 = calculate_pixel_area(image_path)

# 2. Calculate the total land area, divide it into portions, and display the results
if pixel_area_mm2:
    calculate_total_land_area(image_path, pixel_area_mm2)
