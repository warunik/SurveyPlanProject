# Land Boundary Detection and Division

This project processes an image to detect the boundary of a land area, fills the detected boundary, and divides the filled area into specified portions. It utilizes OpenCV for image processing and Matplotlib for visualization.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

Install the required libraries using pip:

```bash
pip install opencv-python numpy matplotlib
```

## Usage

1. Place your image in the appropriate directory and update the `image_path` variable in the code.
2. Run the script. It will:
   - Read and preprocess the image.
   - Detect edges using the Sobel operator.
   - Find and draw the largest contour, assumed to be the land boundary.
   - Fill the detected boundary.
   - Divide the filled area into user-defined portions.

3. Follow the prompts to input the number of portions you want to divide the land into.

## Code Breakdown

### Import Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
```

### Read and Preprocess Image

- Read the image from the specified path.
- Convert the image to grayscale.
- Apply Gaussian blur to reduce noise.

```python
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```

### Edge Detection

- Calculate gradients using the Sobel operator.
- Compute the magnitude of gradients and normalize it.
- Perform thresholding to obtain binary edges.

```python
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
magnitude = np.uint8(magnitude)
_, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
```

### Contour Detection and Filling

- Find contours in the edge-detected image.
- Select the largest contour (assumed to be the land boundary).
- Compute the centroid of the contour.
- Perform flood fill starting from the centroid.

```python
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
land_contour = max(contours, key=cv2.contourArea)
contour_image = np.zeros_like(gray)
cv2.drawContours(contour_image, [land_contour], -1, (255), 2)
M = cv2.moments(land_contour)
centroid_x = int(M['m10'] / M['m00'])
centroid_y = int(M['m01'] / M['m00'])
mask = np.zeros((contour_image.shape[0] + 2, contour_image.shape[1] + 2), dtype=np.uint8)
cv2.floodFill(contour_image, mask, (centroid_x, centroid_y), 255)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
opening = cv2.morphologyEx(contour_image, cv2.MORPH_OPEN, kernel)
```

### Area Calculation and Division

- Count the number of white pixels in the filled image.
- Divide the area into user-defined portions and color each portion uniquely.

```python
number_of_white_pix = np.sum(opening == 255)
no_of_portions = int(input("To how many portions should the land get divided: "))
pixels_per_portion = number_of_white_pix // no_of_portions

def generate_color():
    return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

colors = [generate_color() for _ in range(no_of_portions)]
portion_index = 0
pixels_colored = 0

for y in range(opening.shape[0]):
    for x in range(opening.shape[1]):
        if opening[y, x] == 255:
            divided_image[y, x] = colors[portion_index]
            pixels_colored += 1
            if pixels_colored == pixels_per_portion:
                portion_index += 1
                pixels_colored = 0
            if portion_index == no_of_portions:
                break
    else:
        continue
    break
```

### Display Results

- Display the original image, land boundary contour, filled shape, and divided image using Matplotlib.

```python
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
```

---

