import cv2
import numpy as np
import time
import os

# ==========================================
#               CONFIGURATION
# ==========================================
CONFIG = {
    # TARGET_WIDTH: We resize every image to 800px wide.
    # This ensures a 4K image and a 720p image are processed the same way.
    'TARGET_WIDTH': 800,

    # Color Detection (HSV) - To FIND the Red Box
    'RED_LOWER_1': np.array([0, 70, 50]),
    'RED_UPPER_1': np.array([10, 255, 255]),
    'RED_LOWER_2': np.array([170, 70, 50]),
    'RED_UPPER_2': np.array([180, 255, 255]),

    # Filtering
    'BLUR_KERNEL': (5, 5),
    'MORPH_KERNEL': (3, 3),
    'MIN_AREA': 1500,
    'ASPECT_RATIO_MIN': 0.75,
    'ASPECT_RATIO_MAX': 1.25,

    # Logic (YOUR CHANGES):
    # 0.55: The center strip must be 55% white to be "Restricted".
    # 0.10: We only analyze the middle 10% of the sign's height.
    'WHITE_BAR_THRESHOLD': 0.55,
    'CENTER_STRIP_HEIGHT': 0.10,
}


# ==========================================
#            HELPER FUNCTIONS
# ==========================================

def normalize_image(image, target_width):
    h, w = image.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def is_box_inside(inner_box, outer_box):
    xi, yi, wi, hi = inner_box
    xo, yo, wo, ho = outer_box
    buffer = 2
    if (xi >= xo - buffer) and (yi >= yo - buffer) and \
            (xi + wi <= xo + wo + buffer) and (yi + hi <= yo + ho + buffer):
        return True
    return False


def draw_dashed_horizontal_line(img, x_start, x_end, y, color, thickness=2, dash_len=10):
    for x in range(x_start, x_end, dash_len * 2):
        cv2.line(img, (x, y), (min(x + dash_len, x_end), y), color, thickness)


# ==========================================
#              MAIN LOGIC
# ==========================================

def detect_signs_with_masks(input_folder):
    if not os.path.exists(input_folder):
        print(f"Error: Directory '{input_folder}' not found.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    print(f"{'=' * 60}")
    print(" PRESS 'q' TO QUIT | PRESS ANY KEY FOR NEXT IMAGE")
    print(f"{'=' * 60}\n")

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        original_image = cv2.imread(img_path)
        if original_image is None: continue

        start_time = time.perf_counter()

        # 1. Normalize
        image = normalize_image(original_image, CONFIG['TARGET_WIDTH'])
        display_img = image.copy()

        # 2. Pre-processing
        blurred = cv2.GaussianBlur(image, CONFIG['BLUR_KERNEL'], 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. Red Mask
        mask1 = cv2.inRange(hsv, CONFIG['RED_LOWER_1'], CONFIG['RED_UPPER_1'])
        mask2 = cv2.inRange(hsv, CONFIG['RED_LOWER_2'], CONFIG['RED_UPPER_2'])
        mask_red_raw = mask1 + mask2
        mask_red = cv2.dilate(mask_red_raw, np.ones(CONFIG['MORPH_KERNEL'], np.uint8), iterations=1)
        cv2.imshow("mask_red", mask_red)
        cv2.imshow("mask_red_raw", mask_red_raw)

        contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > CONFIG['MIN_AREA']:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                if CONFIG['ASPECT_RATIO_MIN'] <= aspect_ratio <= CONFIG['ASPECT_RATIO_MAX']:

                    # Capture the ROIs
                    roi_color = image[y:y + h, x:x + w]
                    roi_red_mask = mask_red[y:y + h, x:x + w]

                    # --- WHITE DETECTION LOGIC (CLAHE + OTSU) ---
                    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

                    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    roi_enhanced = clahe.apply(roi_gray)
                    _, mask_white_roi = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Logic Calculation
                    h_roi = mask_white_roi.shape[0]
                    margin = (1.0 - CONFIG['CENTER_STRIP_HEIGHT']) / 2
                    local_y_start = int(h_roi * margin)
                    local_y_end = int(h_roi * (1.0 - margin))

                    center_strip = mask_white_roi[local_y_start:local_y_end, :]
                    white_pixels = cv2.countNonZero(center_strip)
                    total_strip_pixels = max(1, center_strip.size)
                    white_fill_ratio = white_pixels / total_strip_pixels

                    roi_red = mask_red[y:y + h, x:x + w]
                    red_pixels = cv2.countNonZero(roi_red)
                    total_box_pixels = w * h
                    red_fill_ratio = red_pixels / total_box_pixels

                    if white_fill_ratio > CONFIG['WHITE_BAR_THRESHOLD']:
                        label = "R"
                        color = (0, 255, 255)  # Yellow
                    else:
                        label = "S"
                        color = (0, 0, 255)  # Red

                    candidates.append({
                        'box': (x, y, w, h),
                        'label': label,
                        'color': color,
                        'white_ratio': white_fill_ratio,
                        'red_ratio': red_fill_ratio,
                        'area': area,
                        'vis_orig': roi_color,
                        'vis_red': roi_red_mask,
                        'vis_white': mask_white_roi
                    })

        # Filter Nested Boxes
        candidates.sort(key=lambda x: x['area'], reverse=True)
        final_detections = []
        processed_indices = set()
        for i in range(len(candidates)):
            if i in processed_indices: continue
            outer = candidates[i]
            final_detections.append(outer)
            for j in range(i + 1, len(candidates)):
                if j in processed_indices: continue
                if is_box_inside(candidates[j]['box'], outer['box']):
                    processed_indices.add(j)

        # Draw & Display
        print(f"IMAGE: {img_name}")

        # --- MASK DEBUG VISUALIZATION (FIXED WITH PADDING) ---
        if final_detections:
            debug_views = []
            fixed_height = 200

            # 1. Generate individual debug strips
            for det in final_detections:
                orig = det['vis_orig']

                # Convert Masks to Color for Stacking
                red_vis = cv2.cvtColor(det['vis_red'], cv2.COLOR_GRAY2BGR)
                red_vis[:, :, 0] = 0  # Remove Blue
                red_vis[:, :, 1] = 0  # Remove Green

                white_vis = cv2.cvtColor(det['vis_white'], cv2.COLOR_GRAY2BGR)

                # Resize to fixed height, keeping aspect ratio
                scale = fixed_height / orig.shape[0]
                new_w = int(orig.shape[1] * scale)

                v1 = cv2.resize(orig, (new_w, fixed_height))
                v2 = cv2.resize(red_vis, (new_w, fixed_height))
                v3 = cv2.resize(white_vis, (new_w, fixed_height))

                # Borders
                cv2.rectangle(v1, (0, 0), (new_w - 1, fixed_height - 1), det['color'], 2)
                cv2.rectangle(v2, (0, 0), (new_w - 1, fixed_height - 1), (0, 0, 255), 2)
                cv2.rectangle(v3, (0, 0), (new_w - 1, fixed_height - 1), (255, 255, 255), 2)

                combined = np.hstack([v1, v2, v3])
                debug_views.append(combined)

            # 2. Pad widths to match the widest row
            if debug_views:
                max_width = max(view.shape[1] for view in debug_views)
                padded_views = []

                for view in debug_views:
                    h, w, c = view.shape
                    if w < max_width:
                        # Add black padding to the right
                        padding = np.zeros((h, max_width - w, c), dtype=np.uint8)
                        view = np.hstack([view, padding])
                    padded_views.append(view)

        else:
            empty = np.zeros((100, 300, 3), dtype=np.uint8)
            cv2.putText(empty, "No Detection", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw on Main Image
        for det in final_detections:
            x, y, w, h = det['box']
            label = det['label']

            print(f"  -> {label} | White%: {det['white_ratio'] * 100:.1f}%")

            cv2.rectangle(display_img, (x, y), (x + w, y + h), det['color'], 3)

            margin = (1.0 - CONFIG['CENTER_STRIP_HEIGHT']) / 2
            strip_top = int(y + (h * margin))
            strip_bottom = int(y + (h * (1.0 - margin)))
            draw_dashed_horizontal_line(display_img, x, x + w, strip_top, (255, 255, 0))
            draw_dashed_horizontal_line(display_img, x, x + w, strip_bottom, (255, 255, 0))

            label_txt = f"{label} ({det['white_ratio']:.2f})"
            (t_w, t_h), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.6, 2)
            cv2.rectangle(display_img, (x, y - 25), (x + t_w, y), det['color'], -1)
            cv2.putText(display_img, label_txt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000
        print(f"Image processing time: {execution_time:.2f} ms")
        print("-" * 40)
        cv2.imshow("Main View", display_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_signs_with_masks("./images")
