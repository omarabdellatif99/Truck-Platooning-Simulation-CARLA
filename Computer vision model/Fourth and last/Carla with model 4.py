#!/usr/bin/env python3
"""
CARLA template: constant forward speed + steering from student model.

Students: put your own logic inside predict_steering(img).
The function must return a value between -1 and 1.

Dependencies
------------
Only CARLA is required.  If your model needs numpy, torch, etc.,
import them at the top and add them to requirements as needed.
"""
import cv2
import numpy as np
import os
import math # For angle calculations
import joblib # For loading models and scalers
import pandas as pd # Used to define feature_columns consistently

# ------------------------ CONFIGURATION --------------------------------------
HOST            = "localhost"
PORT            = 2000
TIMEOUT_S       = 5.0          # seconds

THROTTLE        = 0.5          # constant forward throttle (0..1)
DEFAULT_STEER   = 0.0          # fallback if no camera frame yet
PRINT_EVERY_N   = 30           # console frames between logs
# -----------------------------------------------------------------------------

# --- Configuration: Model & Scaler Paths ---
# IMPORTANT: These paths must point to where your trained model and scaler are saved.
MODEL_DIR = "model2" # Directory created by train_svr_model.py
SVR_MODEL_FILENAME = "svm_lane_deviation_model.joblib"
SCALER_FILENAME = "scaler_lane_deviation.joblib"

# --- Configuration: Hardcoded Parameters (MUST MATCH THOSE USED FOR FEATURE EXTRACTION & TRAINING) ---
# Copy these values EXACTLY from your 'feature_extractor.py' or 'train_svr_model.py' script.
HARDCODED_PARAMETERS = {
    # HSV Color Space
    "hsv_lower": np.array([6,62, 155]),
    "hsv_upper": np.array([13, 106, 255]),

    # CLAHE Parameters
    "clahe_clip_limit": 93.0,
    "clahe_tile_grid_size": 20,

    # Processing Scale (0.01 to 1.0)
    "processing_scale_percent": 1.0,

    # Morphological Kernel for Color Mask
    "color_morph_kernel_size": 2,

    # Edge Pre-processing Filter
    "use_bilateral_filter": 1,
    "bilateral_d": 14,
    "bilateral_sigma_color": 49,
    "bilateral_sigma_space": 129,

    # Canny Edge Detector Thresholds
    "canny_thresh1": 36,
    "canny_thresh2": 107,

    # Morphological Kernel for Edge Mask
    "edge_morph_kernel_size": 19,

    # Hough Line Transform Parameters
    "hough_threshold": 8,
    "hough_min_length": 32,
    "hough_max_gap": 5,

    # Line Filtering Parameters
    "max_line_angle_deg": 79, # Filter lines by angle (e.g., 20 degrees from horizontal)

    # Image Cropping Parameters (in percentage of original image dimensions)
    "crop_percent_top": 56,
    "crop_percent_left": 15,
    "crop_percent_right": 15,
}

# --- Feature Columns (MUST BE IN THE EXACT SAME ORDER AS USED FOR TRAINING) ---
# This list must precisely match the 'feature_columns' in your train_svr_model.py script.
FEATURE_COLUMNS = [
    "lane_centroid_x_cropped_px",
    "num_detected_lines",
    "avg_line_slope_deg",
    "avg_line_length_px",
    "avg_line_x_at_bottom_px",
    "avg_line_x_at_top_px",
    "final_mask_white_pixels",
    "final_mask_centroid_x_px",
    "final_mask_centroid_y_px",
]

# --- Function to Extract Features from a Single Image ---
def extract_features_for_prediction(image, params):
    """
    Processes a single image to detect the orange lane line and extract relevant features
    using the new OpenCV pipeline.
    Returns a NumPy array of features and an annotated image (for visualization, not used in CARLA).
    """
    
    original_h, original_w, _ = image.shape
    annotated_image = image.copy() # This will not be used in CARLA, but kept for consistency

    # Initialize all features with default values
    features_dict = {
        "lane_centroid_x_cropped_px": 0.0,
        "num_detected_lines": 0,
        "avg_line_slope_deg": 0.0,
        "avg_line_length_px": 0.0,
        "avg_line_x_at_bottom_px": 0.0,
        "avg_line_x_at_top_px": 0.0,
        "final_mask_white_pixels": 0,
        "final_mask_centroid_x_px": 0.0,
        "final_mask_centroid_y_px": 0.0,
    }

    # --- 1. Apply Cropping ---
    crop_y_start = int(original_h * params["crop_percent_top"] / 100)
    crop_x_start = int(original_w * params["crop_percent_left"] / 100)
    crop_x_end = original_w - int(original_w * params["crop_percent_right"] / 100)
    if crop_x_start >= crop_x_end: # Fallback for invalid crop
        crop_x_start = 0
        crop_x_end = original_w
    if crop_y_start >= original_h: # Fallback for invalid crop
        crop_y_start = original_h - 1

    cropped_image = image[crop_y_start:original_h, crop_x_start:crop_x_end].copy()
    cropped_h, cropped_w, _ = cropped_image.shape

    if cropped_h == 0 or cropped_w == 0:
        # print("Warning: Cropped image has zero height or width. Returning default features.")
        ordered_features_array = np.array([features_dict[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
        return ordered_features_array, annotated_image # annotated_image is just the original image in this case


    # --- 2. Resize for Processing Performance ---
    processing_width = int(cropped_w * params["processing_scale_percent"])
    processing_height = int(cropped_h * params["processing_scale_percent"])
    if processing_width < 1: processing_width = 1
    if processing_height < 1: processing_height = 1

    cropped_image_for_processing = cv2.resize(cropped_image, (processing_width, processing_height), interpolation=cv2.INTER_LINEAR)

    # --- 3. Illumination Normalization (CLAHE) ---
    hsv_image = cv2.cvtColor(cropped_image_for_processing, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"],
                            tileGridSize=(params["clahe_tile_grid_size"], params["clahe_tile_grid_size"]))
    v_clahe = clahe.apply(v)
    
    normalized_hsv = cv2.merge([h, s, v_clahe])
    normalized_image = cv2.cvtColor(normalized_hsv, cv2.COLOR_HSV2BGR)

    # --- 4. Color Masking (HSV on CLAHE-processed image) ---
    hsv_normalized = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
    lower_hsv = params["hsv_lower"]
    upper_hsv = params["hsv_upper"]
    color_mask = cv2.inRange(hsv_normalized, lower_hsv, upper_hsv)

    # --- 5. Morphological Operations on the Color Mask ---
    color_morph_kernel = np.ones((params["color_morph_kernel_size"], params["color_morph_kernel_size"]), np.uint8)
    color_mask_morphed = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, color_morph_kernel, iterations=1)
    color_mask_morphed = cv2.dilate(color_mask_morphed, color_morph_kernel, iterations=1)
    color_mask_morphed = cv2.morphologyEx(color_mask_morphed, cv2.MORPH_CLOSE, color_morph_kernel, iterations=1)
    
    # --- 6. Edge Detection (Canny) ---
    gray_normalized = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    
    if params["use_bilateral_filter"]:
        filtered_gray = cv2.bilateralFilter(gray_normalized, params["bilateral_d"],
                                             params["bilateral_sigma_color"], params["bilateral_sigma_space"])
    else:
        filtered_gray = cv2.GaussianBlur(gray_normalized, (5, 5), 0)

    edge_mask = cv2.Canny(filtered_gray, params["canny_thresh1"], params["canny_thresh2"])
    edge_morph_kernel = np.ones((params["edge_morph_kernel_size"], params["edge_morph_kernel_size"]), np.uint8)
    edge_mask_morphed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, edge_morph_kernel, iterations=1)

    # --- 7. Combine Masks ---
    final_mask = cv2.bitwise_and(color_mask_morphed, edge_mask_morphed)

    # --- 8. Hough Line Transform ---
    lines = cv2.HoughLinesP(final_mask, 1, np.pi / 180,
                            params["hough_threshold"],
                            minLineLength=params["hough_min_length"],
                            maxLineGap=params["hough_max_gap"])

    # --- Calculate mask centroid if there are white pixels ---
    if np.count_nonzero(final_mask) > 0:
        features_dict["final_mask_white_pixels"] = np.count_nonzero(final_mask)
        # Find coordinates of all white pixels
        coords = cv2.findNonZero(final_mask)
        # Calculate centroid
        mask_cx = int(np.mean(coords[:,:,0]))
        mask_cy = int(np.mean(coords[:,:,1]))
        # Scale mask centroid back to cropped_image dimensions for consistency
        scale_x_mask = cropped_w / processing_width
        scale_y_mask = cropped_h / processing_height
        features_dict["final_mask_centroid_x_px"] = int(mask_cx * scale_x_mask)
        features_dict["final_mask_centroid_y_px"] = int(mask_cy * scale_y_mask)

    filtered_lines_data = [] # Store (x1, y1, x2, y2, angle_deg, length)
    if lines is not None:
        # Scale coordinates back to original cropped_image dimensions
        scale_x_line = cropped_w / processing_width
        scale_y_line = cropped_h / processing_height
        
        for line in lines:
            x1_scaled, y1_scaled, x2_scaled, y2_scaled = line[0]
            
            x1 = int(x1_scaled * scale_x_line)
            y1 = int(y1_scaled * scale_y_line)
            x2 = int(x2_scaled * scale_x_line)
            y2 = int(y2_scaled * scale_y_line)

            # Ensure no division by zero for angle calculation
            if (x2 - x1) == 0 and (y2 - y1) == 0:
                angle_deg = 0.0
            else:
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
            
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Original angle filtering logic (keeping lines close to horizontal)
            if (abs(angle_deg) < params["max_line_angle_deg"] or
                abs(angle_deg - 180) < params["max_line_angle_deg"] or
                abs(angle_deg + 180) < params["max_line_angle_deg"]):
                filtered_lines_data.append((x1, y1, x2, y2, angle_deg, length))

    if filtered_lines_data:
        features_dict["num_detected_lines"] = len(filtered_lines_data)
        
        all_midpoints_x = []
        all_angles = []
        all_lengths = []
        all_bottom_xs = [] # X-coordinate at y=cropped_h-1
        all_top_xs = []    # X-coordinate at y=0

        for x1, y1, x2, y2, angle_deg, length in filtered_lines_data:
            all_midpoints_x.append((x1 + x2) // 2)
            all_angles.append(angle_deg)
            all_lengths.append(length)

            # Calculate X-coordinate at the bottom (cropped_h-1) and top (0) of the cropped image
            if abs(x2 - x1) > 0.1: # Not a perfectly vertical line (avoid div by zero)
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 1e-6: # Avoid division by zero for horizontal lines
                    # X at bottom (y=cropped_h-1)
                    x_at_bottom = ((cropped_h - 1) - y1) / slope + x1
                    all_bottom_xs.append(x_at_bottom)

                    # X at top (y=0)
                    x_at_top = (0 - y1) / slope + x1
                    all_top_xs.append(x_at_top)
                else: # Nearly horizontal line, use average X
                    avg_x = (x1 + x2) / 2
                    all_bottom_xs.append(avg_x)
                    all_top_xs.append(avg_x)
            else: # Vertical line
                all_bottom_xs.append(x1) # For vertical line, x is constant
                all_top_xs.append(x1)

        features_dict["lane_centroid_x_cropped_px"] = float(np.mean(all_midpoints_x))
        features_dict["avg_line_slope_deg"] = float(np.mean(all_angles))
        features_dict["avg_line_length_px"] = float(np.mean(all_lengths))
        
        if all_bottom_xs:
            features_dict["avg_line_x_at_bottom_px"] = float(np.mean(all_bottom_xs))
        if all_top_xs:
            features_dict["avg_line_x_at_top_px"] = float(np.mean(all_top_xs))

    # Convert features_dict to an ordered NumPy array based on FEATURE_COLUMNS
    ordered_features_array = np.array([features_dict[col] for col in FEATURE_COLUMNS]).reshape(1, -1)
    
    return ordered_features_array, annotated_image # Return features and the image with annotations


# ------------------------------------------------------------------ STUDENTS --
def predict_steering(img):
    """
    Returns steering prediction from the SVM model based on the input image.

    Parameters
    ----------
    img : carla.Image
        The latest RGB camera frame (BGRA byte-buffer).

    Returns
    -------
    float
        Steering value in [-1, 1].
    """
    # -------------- load the model and scaler only once ---------------------------
    if not hasattr(predict_steering, "_model"):
        predict_steering._model = None
        predict_steering._scaler = None
        predict_steering._feature_columns_order = FEATURE_COLUMNS # Use the defined global feature columns
        predict_steering._hardcoded_parameters = HARDCODED_PARAMETERS

        try:
            model_path = os.path.join(MODEL_DIR, SVR_MODEL_FILENAME)
            scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)
            
            predict_steering._model = joblib.load(model_path)
            predict_steering._scaler = joblib.load(scaler_path)
            print(f"[INFO] SVM model and scaler loaded successfully from '{model_path}' and '{scaler_path}'.")
            print(f"[INFO] Feature column order: {predict_steering._feature_columns_order}")
        except FileNotFoundError as e:
            print(f"[ERROR] Model or scaler file not found. Ensure '{MODEL_DIR}' exists and contains '{SVR_MODEL_FILENAME}' and '{SCALER_FILENAME}'.")
            print(f"Details: {e}")
            print(f"[WARN] Only default steering will be used as a fallback.")
        except Exception as e:
            print(f"[ERROR] Error loading model or scaler: {e}")
            print(f"[WARN] Only default steering will be used as a fallback.")

    model = predict_steering._model
    scaler = predict_steering._scaler
    feature_columns_order = predict_steering._feature_columns_order
    params = predict_steering._hardcoded_parameters

    # If model or scaler failed to load, return default steering
    if model is None or scaler is None or feature_columns_order is None:
        return DEFAULT_STEER

    # Convert carla.Image to an OpenCV image (NumPy array)
    # The image is BGRA, so we convert it to BGR
    img_array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    bgr_image = img_array[:, :, :3] # Take only BGR channels

    # 1. Extract features
    raw_features_array, _ = extract_features_for_prediction(bgr_image, params)
    
    # Determine if a line was "detected" based on the number of detected lines
    num_detected_lines_feature = raw_features_array[0, FEATURE_COLUMNS.index('num_detected_lines')]
    is_line_detected_flag = 1 if num_detected_lines_feature > 0 else 0

    predicted_steering_value = 0.0 # Default to 0.0 if no line is detected

    if is_line_detected_flag == 1:
        # 2. Scale the extracted features
        # Ensure the raw_features_array has the correct shape and column order for the scaler
        # We assume extract_features_for_prediction already returns features in FEATURE_COLUMNS order
        
        try:
            scaled_features = scaler.transform(raw_features_array)

            # 3. Predict deviation score
            predicted_deviation = model.predict(scaled_features)[0] # .predict returns an array, get the first element
            
            # Map the predicted deviation to a steering value between -1 and 1
            # Assuming predicted_deviation is centered around 0, positive for right, negative for left
            # Adjust the scaling factor (e.g., 0.1) based on your model's output range and desired steering sensitivity
            # For instance, if your model outputs deviation in pixels from the center,
            # you'll need to normalize it to the [-1, 1] steering range.
            
            # Example mapping: A simple linear mapping for demonstration.
            # You will need to fine-tune this based on your SVR's output range.
            # Let's assume your SVR outputs values roughly between -WIDTH/2 and WIDTH/2
            # for `lane_centroid_x_cropped_px` relative to the center.
            # The range of your SVR output directly corresponds to the steering range.
            # If your SVR was trained to output values from -1 to 1 directly, then `predicted_deviation` is already the steering.
            # If it predicts deviation in pixels, you need to convert it.
            
            # For demonstration, let's assume a predicted deviation of -50 to 50 pixels maps to -1 to 1 steering.
            # Adjust `max_deviation_for_steering` based on your actual pixel deviation range.
            # A common approach is to use a P-controller like mapping if deviation is in spatial units:
            # steering = Kp * deviation_from_center
            # Where Kp is a proportional gain.
            
            # Let's assume your SVR directly predicts a normalized steering value in [-1, 1].
            # If not, you need to implement the mapping here.
            # For now, let's just use predicted_deviation as the steering value directly,
            # assuming your SVR was trained to output in that range.
            
            predicted_steering_value = predicted_deviation
            predicted_steering_value = max(-1.0, min(1.0, predicted_steering_value)) # Clip to [-1, 1]

        except Exception as e:
            print(f"[ERROR] Error during prediction: {e}")
            predicted_steering_value = DEFAULT_STEER # Fallback if prediction fails
    else:
        predicted_steering_value = 0.0 # Steering is zero if no line is detected

    return predicted_steering_value


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world  = client.get_world()

    vehicles = world.get_actors().filter("vehicle.*")
    if not vehicles:
        print("No vehicles found. Start a scene first.")
        return
    vehicle = vehicles[0]
    print("Controlling vehicle id=%d type=%s" % (vehicle.id, vehicle.type_id))
    vehicle.set_autopilot(False)

    camera = pick_center_camera(world, vehicle)
    if camera is None:
        print("No center RGB camera attached to the vehicle.")
        return
    print("Using camera id=%d for live feed" % camera.id)

    state = {"frames": 0, "first_ts": None, "latest_img": None}

    def cam_cb(img):
        state["latest_img"] = img
        state["frames"] += 1
        if state["frames"] % PRINT_EVERY_N == 0:
            if state["first_ts"] is None:
                state["first_ts"] = img.timestamp
            elapsed = img.timestamp - state["first_ts"]
            fps = state["frames"] / elapsed if elapsed else 0.0
            print("camera frames: %d   %.1f FPS" % (state["frames"], fps))

    camera.listen(cam_cb)

    try:
        while True:
            img = state["latest_img"]
            if img is not None:
                steer = float(max(-1.0, min(1.0, predict_steering(img))))
            else:
                steer = DEFAULT_STEER  # if no frame yet
            vehicle.apply_control(carla.VehicleControl(throttle=THROTTLE,
                                                       steer=steer))
            time.sleep(0.01)  # ~100 Hz loop

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        camera.stop()
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        sys.stderr.write("[ERROR] " + str(err) + "\n"
                         "Is the CARLA server running on this host/port?\n")