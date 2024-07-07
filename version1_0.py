import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Read the image
image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/plan (6).jpg'
image = cv2.imread(image_path)

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

# Analyze middle space within the contour to extract the "real" shape
# Compute centroid of the contour
M = cv2.moments(land_contour)
centroid_x = int(M['m10'] / M['m00'])
centroid_y = int(M['m01'] / M['m00'])

# Create a mask for flood fill
mask = np.zeros((contour_image.shape[0] + 2, contour_image.shape[1] + 2), dtype=np.uint8)

# Perform a flood fill starting from the centroid to obtain the filled shape
cv2.floodFill(contour_image, mask, (centroid_x, centroid_y), 255)

# Opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(35,35))
opening = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)

# Save the image to calculate the area
# cv2.imwrite('area6.jpg', opening)

# Invert the filled image
# filled_image = cv2.bitwise_not(opening)

divided_image = image.copy()
  
# counting the number of pixels 
number_of_white_pix = np.sum(opening == 255) 
  
print('Number of white pixels:', number_of_white_pix) 

# divide the white pixels using user input
no_of_portions = int(input("To how many portions should the land get divided: "))

pixels_per_portion = number_of_white_pix // no_of_portions

print('Number of pixels per portion: ', pixels_per_portion)

def generate_color():
    return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

colors = [generate_color() for _ in range(no_of_portions)]  # Generate colors for each portion

portion_index = 0  # Index to keep track of current color
pixels_colored = 0  # Counter to keep track of colored pixels

for y in range(opening.shape[0]):
    for x in range(opening.shape[1]):
        if opening[y, x] == 255:
            divided_image[y, x] = colors[portion_index]  # Set pixel to current portion's color
            pixels_colored += 1  # Increment counter
            
            if pixels_colored == pixels_per_portion:  # Check if enough pixels for current portion are colored
                portion_index += 1  # Move to next color for next portion
                pixels_colored = 0  # Reset counter
                
            if portion_index == no_of_portions:  # Check if all portions have been colored
                break
    else:
        continue
    break



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
plt.imshow(divided_image, cmap='gray')
plt.title('Filled Image')
plt.axis('off')

plt.show()
