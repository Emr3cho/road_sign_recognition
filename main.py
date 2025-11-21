import cv2
import numpy as np


def detect_traffic_signs(image_path=None, frame=None):
    """
    Detects Stop Signs and No Entry Signs based on Color and Shape.
    Can accept an image path OR a video frame.
    """

    # 1. Input Handling
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return
    else:
        img = frame.copy()

    # 2. Preprocessing [cite: 6488, 6490]
    # Apply Gaussian Blur to reduce noise/details for better contour detection
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to HSV color space
    # HSV separates Luma (Intensity) from Chroma (Color), making it robust to lighting changes.
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 3. Color Segmentation (Red Color)
    # Red wraps around the HSV spectrum (0-180), so we need two masks.

    # Lower Red range (0-10)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    # Upper Red range (170-180)
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.add(mask1, mask2)

    # 4. Morphological Operations
    # Remove small white noise (Opening) and fill small holes (Closing)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. Contour Analysis [cite: 6268]
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        # Filter small noise contours
        if area < 1000 or perimeter == 0:
            continue

        # 6. Shape Approximation
        # Epsilon is the approximation accuracy.
        # 0.04 (4%) of perimeter is standard for geometric shapes.
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)

        # Bounding Box for ROI (Region of Interest) analysis
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Shape Classification Logic

        # --- STOP SIGN DETECTION (Octagon) ---
        # Ideally 8 vertices, but due to perspective/noise, we accept 7-9.
        if 7 <= vertices <= 9 and 0.8 < aspect_ratio < 1.2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "STOP", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # --- NO ENTRY SIGN DETECTION (Circle with internal feature) ---
        # Circles have many vertices in approximation (>8).
        # We also check Circularity: 4*pi*Area / Perimeter^2
        else:
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            # If it looks like a circle (circularity > 0.7) or has many vertices
            if vertices > 8 and circularity > 0.7 and 0.8 < aspect_ratio < 1.2:

                # Internal Verification: Check for White Bar
                # Extract ROI (Region of Interest)
                roi = img[y:y + h, x:x + w]

                # Convert ROI to HSV to find White
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Sensitivity for white color (low Saturation, high Value)
                lower_white = np.array([0, 0, 200])
                upper_white = np.array([180, 50, 255])
                white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)

                # Calculate white pixel ratio
                white_pixels = cv2.countNonZero(white_mask)
                total_pixels = w * h
                white_ratio = white_pixels / total_pixels

                # A 'No Entry' sign usually has a white bar covering roughly 15-30% of the area
                if white_ratio > 0.1:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "NO ENTRY", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display Result
    cv2.imshow("Traffic Sign Detection", img)
    if image_path:
        cv2.waitKey(0)


if __name__ == "__main__":
    image_files = ['stop_sign.jpg', 'two_road_signs.png', 'stop-sign-along-country_side.jpg']

    print("Processing Images...")
    for img_file in image_files:
        try:
            detect_traffic_signs(image_path="./images/" + img_file)
        except Exception as e:
            print(f"Skipping {img_file}: {e}")

    cv2.destroyAllWindows()
