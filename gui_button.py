import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

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
            # pixel_area_mm2 = (15*15)/(width_pixels*height_pixels)
            pixel_area_mm2 = pixel_size_mm ** 2
            pixel_area_um2 = pixel_area_mm2 * 1e6
            
            print(f"Pixel size: {pixel_size_mm:.6f} mm")
            print(f"Pixel area: {pixel_area_mm2:.6f} mm^2")
            print(f"Pixel area: {pixel_area_um2:.2f} μm^2")
            
            return pixel_area_mm2  # Return the area of a single pixel in mm²
    return None

# Function to handle button clicks and close the scale selection window
def on_scale_select(scale_value, window, scale_var):
    scale_var.set(scale_value)
    window.destroy()

# Function to ask user for the plan scale using buttons
def ask_scale():
    scale_var = tk.StringVar()  # Variable to store the selected scale
    scale_window = tk.Toplevel()  # Create a new window for scale selection
    scale_window.title("Select Plan Scale")

    label = tk.Label(scale_window, text="Select the plan scale:")
    label.pack(pady=10)

    # Create buttons for 1:1000 and 1:500
    button_1000 = tk.Button(scale_window, text="1:1000", command=lambda: on_scale_select("1:1000", scale_window, scale_var))
    button_1000.pack(pady=5)

    button_500 = tk.Button(scale_window, text="1:500", command=lambda: on_scale_select("1:500", scale_window, scale_var))
    button_500.pack(pady=5)

    scale_window.mainloop()
    
    return scale_var.get()

# Function to calculate total land area
def calculate_total_land_area(image_path, pixel_area_mm2):
    root = tk.Tk()
    root.withdraw()

    # Call ask_scale() to display buttons for scale selection
    scale_input = ask_scale()

    if scale_input == "1:1000":
        scale_factor = 1000 * 1000  # Scale for 1:1000 in cm²
    elif scale_input == "1:500":
        scale_factor = 500 * 500  # Scale for 1:500 in cm²
    else:
        messagebox.showerror("Error", "Invalid scale selected.")
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
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    plt.imshow(cv2.cvtColor(divided_image, cv2.COLOR_BGR2RGB))
    plt.title('Divided Land Image')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(original_image_divided, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Borders')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, f'Total Land Area: {total_land_area_m2:.2f} m²', fontsize=14)
    plt.text(0.1, 0.6, f'Portion Area: {portion_area_m2:.2f} m²', fontsize=14)
    plt.axis('off')

    plt.show()

# Main code execution
def main():
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    image_path = 'D:/UOP/3rd year/Sem 5/CSC3141 - Image Processing Laboratory/SurveyPlanProject/images/plan (17).jpg'

    if not image_path:
        messagebox.showerror("Error", "No image selected.")
        return

    # Calculate pixel area in mm²
    pixel_area_mm2 = calculate_pixel_area(image_path)

    if pixel_area_mm2 is not None:
        # Calculate the total land area
        calculate_total_land_area(image_path, pixel_area_mm2)
    else:
        messagebox.showerror("Error", "Could not detect the square for pixel area calculation.")

if __name__ == "__main__":
    main()
