import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Known radius of the coin in centimeters
coin_radius_cm = 1.5

# Read the image
image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/Mini Project/images/plan (6).jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found. Check the path and try again.")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Calculate the gradients using the Sobel operator
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Compute the magnitude of gradients
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Normalize the magnitude to the range [0, 255]
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = np.uint8(magnitude)

# Perform thresholding to obtain binary edges
_, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour (assuming it's the land boundary)
land_contour = max(contours, key=cv2.contourArea)

# Create a blank image to draw the contour on
contour_image = np.zeros_like(gray)

# Draw the land boundary contour on the blank image
cv2.drawContours(contour_image, [land_contour], -1, (255), 2)

# Compute centroid of the contour
M = cv2.moments(land_contour)
centroid_x = int(M['m10'] / M['m00'])
centroid_y = int(M['m01'] / M['m00'])

# Create a mask for flood fill
mask = np.zeros((contour_image.shape[0] + 2, contour_image.shape[1] + 2), dtype=np.uint8)

# Perform a flood fill starting from the centroid to obtain the filled shape
cv2.floodFill(contour_image, mask, (centroid_x, centroid_y), 255)

# Apply morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
opening = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

# Count the number of white pixels
number_of_white_pix = np.sum(opening == 255)
print('Number of white pixels:', number_of_white_pix)

# Detect the coin using Hough Circle Transform
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=50)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Draw the center of the circle
        coin_radius_pixels = r  # Use the first detected circle
        break

    # Calculate the area of the coin in pixels
    coin_area_pixels = math.pi * (coin_radius_pixels ** 2)
    print('Radius of coin in pixels:', coin_radius_pixels)
    print('Area of coin in pixels:', coin_area_pixels)

    # Calculate the area of the coin in square centimeters
    coin_area_cm2 = math.pi * (coin_radius_cm ** 2)
    print('Area of coin in square centimeters:', coin_area_cm2)

    # Calculate the conversion factor (cm^2 per pixel^2)
    conversion_factor = coin_area_cm2 / coin_area_pixels
    print('Conversion factor (cm^2 per pixel^2):', conversion_factor)

    # Calculate the area of the land boundary in square centimeters
    land_area_cm2 = number_of_white_pix * conversion_factor
    print('Area of land boundary in square centimeters:', land_area_cm2)

    # Display the original image, land boundary contour, and filled shape
    plt.figure(figsize=(15, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(contour_image, cmap='gray')
    plt.title('Land Boundary Contour (Sobel)')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(opening, cmap='gray')
    plt.title('Recorrected Shape')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Coin')
    plt.axis('off')

    plt.show()
else:
    print("No coin detected. Ensure the coin is clearly visible in the image.")
