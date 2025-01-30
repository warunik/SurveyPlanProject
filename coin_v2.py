import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the distance between two points
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Load the image
image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/ac/plan (100).jpg'
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

# Find the first square and calculate its width and height
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    
    if len(approx) == 4:  # We assume a square has 4 sides
        # Extract the vertices of the square
        pts = approx.reshape(4, 2)
        
        # Calculate the lengths of the sides in pixels
        side_lengths = [distance(pts[i], pts[(i+1)%4]) for i in range(4)]
        
        # Print width and height in pixels
        width = side_lengths[0]
        height = side_lengths[1]
        print(f"Width: {width:.2f} pixels, Height: {height:.2f} pixels")
        
        con_image = image.copy()
        # Draw the detected square on the image
        cv2.drawContours(con_image, [approx], 0, (0, 255, 0), 5)
        
        break  # Stop after finding the first square

# Assume the real-world dimensions of the detected square (in mm)
real_width_mm = 15  # 15 mm
real_height_mm = 15  # 15 mm

# Width and height in pixels (already calculated in the previous step)
width_pixels = side_lengths[0]  # This is the detected width in pixels
height_pixels = side_lengths[1]  # This is the detected height in pixels

# Calculate the size of a pixel in mm (both width and height should give approximately the same value)
pixel_size_mm = real_width_mm / width_pixels  # Size of one pixel in mm

# Calculate the area of a pixel in square mm
pixel_area_mm2 = pixel_size_mm ** 2

# Convert the area of a pixel to square micrometers (1 mm^2 = 1e6 μm^2)
pixel_area_um2 = pixel_area_mm2 * 1e6

# Output the pixel size and area
print(f"Pixel size: {pixel_size_mm:.6f} mm")
print(f"Pixel area: {pixel_area_mm2:.6f} mm^2")
print(f"Pixel area: {pixel_area_um2:.2f} μm^2")


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
plt.imshow(cv2.cvtColor(con_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
