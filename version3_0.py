import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Read the image
image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/plan (10).jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found. Check the path and try again.")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Calculate the gradients using the Sobel operator
grad_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

# Compute the magnitude of gradients
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Normalize the magnitude to the range [0, 255]
gradient_magnitude = np.uint8(gradient_magnitude)

# Perform thresholding to obtain binary edges
_, binary_edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(binary_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour (assuming it's the land boundary)
largest_contour = max(contours, key=cv2.contourArea)

# Create a blank image to draw the contour on
contour_image = np.zeros_like(gray_image)

# Draw the land boundary contour on the blank image
cv2.drawContours(contour_image, [largest_contour], -1, (255), 2)

# Analyze middle space within the contour to extract the "real" shape
# Compute centroid of the contour
moments = cv2.moments(largest_contour)
centroid_x = int(moments['m10'] / moments['m00'])
centroid_y = int(moments['m01'] / moments['m00'])

# Create a mask for flood fill
flood_fill_mask = np.zeros((contour_image.shape[0] + 2, contour_image.shape[1] + 2), dtype=np.uint8)

# Perform a flood fill starting from the centroid to obtain the filled shape
cv2.floodFill(contour_image, flood_fill_mask, (centroid_x, centroid_y), 255)

# Morphological opening to clean the filled shape
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
cleaned_shape = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

# Calculate the number of white pixels (representing the land area)
num_white_pixels = np.sum(cleaned_shape == 255)
print('Number of white pixels:', num_white_pixels)

# Divide the land area into portions based on user input
num_portions = int(input("To how many portions should the land get divided: "))
pixels_per_portion = num_white_pixels // num_portions
print('Number of pixels per portion:', pixels_per_portion)

# Function to generate random colors
def generate_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Generate colors for each portion
portion_colors = [generate_random_color() for _ in range(num_portions)]

# Create a copy of the original image to draw the divided land
divided_image = np.zeros_like(image)

# Color the land portions
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

# Find contours for each portion and draw the borders
portion_borders = np.zeros_like(gray_image)

for portion_color in portion_colors:
    mask = cv2.inRange(divided_image, portion_color, portion_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(portion_borders, contours, -1, 255, 2)

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
plt.imshow(original_with_contours, cmap='gray')
plt.title('Original Image with Borders')
plt.axis('off')

plt.show()
