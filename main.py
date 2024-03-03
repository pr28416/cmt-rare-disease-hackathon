import cv2
import numpy as np

# Load the image
image = cv2.imread("image3.jpg")

# Denoise the image using median blur
denoised_image = cv2.medianBlur(image, 13)

# Convert the denoised image to grayscale
gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

# Detect edges using the Canny edge detector
edges = cv2.Canny(gray, 100, 200)

# Thresholding to remove black areas
ret, thresh = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Set the radius threshold
min_radius = 30

# Set the radius for counting nearby points
px = 80

# Iterate through contours and find red dots
for contour in contours:
    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # If the area is small, skip
    if area < 5:
        continue

    # Approximate the contour to a polygon
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # If the polygon has very few vertices, it might be a dot
    if len(approx) < 10:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI) from the original image
        roi = denoised_image[y : y + h, x : x + w]

        # Convert the ROI to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the red color
        lower_red = np.array([0, 75, 50])
        upper_red = np.array([10, 255, 255])

        # Create a mask for the red color
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Find contours in the mask
        red_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate the centroid of the contour
        for red_contour in red_contours:
            M = cv2.moments(red_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Count the number of nearby points in a px-pixel radius
                num_nearby_points = np.sum(
                    mask[cy - px : cy + px, cx - px : cx + px] == 255
                )

                # Calculate the radius of the circle proportional to the number of nearby points
                radius = int(3 * np.sqrt(num_nearby_points / np.pi))

                # Draw a red dot at the centroid with proportional size
                if radius >= min_radius:
                    cv2.circle(
                        denoised_image, (x + cx, y + cy), radius, (0, 255, 0), -1
                    )

# Display the final result
cv2.imshow("Denoised Image with Red Dots (Proportional Size)", denoised_image)
cv2.imwrite("heatmap.jpg", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
