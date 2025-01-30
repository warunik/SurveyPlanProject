import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/plan (17).jpg'
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

# Draw the contours and highlight the square
detected_image = image.copy()
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:  # We assume a square has 4 sides
        cv2.drawContours(detected_image, [approx], 0, (0, 255, 0), 5)

# Plot the original and detected square image
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Detected square
plt.subplot(1, 2, 2)
plt.title("Detected Square")
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
