#!/usr/bin/env python3
"""
CARLA template: constant forward speed + steering from student model.

Students: put your own logic inside predict_steering(img).
The function must return a value between -1 and 1.

"""
import carla
import random
import time
import sys
import math
import joblib
import numpy as np
import os
import cv2
import pandas as pd

# ------------------------ CONFIGURATION --------------------------------------
HOST            = "localhost"
PORT            = 2000
TIMEOUT_S       = 5.0      # seconds

THROTTLE        = 0.5      # constant forward throttle (0..1)
DEFAULT_STEER   = 0.0      # fallback if no camera frame yet
PRINT_EVERY_N   = 30       # console frames between logs
# -----------------------------------------------------------------------------

# --- Configuration: Model & Scaler Paths ---
# IMPORTANT: These paths must point to where your trained model and scaler are saved.
 # Directory created by train_svr_model.py
SVR_MODEL_FILENAME = "svm_lane_deviation_model.joblib"
SCALER_FILENAME = "scaler_lane_deviation.joblib"

# --- Configuration: Hardcoded Parameters (MUST MATCH THOSE USED FOR FEATURE EXTRACTION & TRAINING) ---
# Copy these values EXACTLY from your 'feature_extractor.py' or 'train_svr_model.py' script.
HARDCODED_PARAMETERS = {
    # HSV Color Space
    "hsv_lower": np.array([6, 62, 155]),
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
    Returns a dictionary of features.
    """

    original_h, original_w, _ = image.shape

    # Initialize all features with default values for "no line detected" scenario
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

    # Ensure valid cropping boundaries
    if crop_x_start >= crop_x_end:
        crop_x_start = 0
        crop_x_end = original_w
    if crop_y_start >= original_h:
        crop_y_start = original_h - 1

    cropped_image = image[crop_y_start:original_h, crop_x_start:crop_x_end].copy()
    cropped_h, cropped_w, _ = cropped_image.shape

    if cropped_h == 0 or cropped_w == 0:
        # If cropped image is empty, return default features
        return features_dict

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

    # --- Calculate mask centroid if there are white pixels ---
    if np.count_nonzero(final_mask) > 0:
        features_dict["final_mask_white_pixels"] = np.count_nonzero(final_mask)
        M = cv2.moments(final_mask)
        if M["m00"] != 0:
            mask_cx = M["m10"] / M["m00"]
            mask_cy = M["m01"] / M["m00"]
            # Scale mask centroid back to cropped_image dimensions for consistency
            scale_x_mask = cropped_w / processing_width
            scale_y_mask = cropped_h / processing_height
            features_dict["final_mask_centroid_x_px"] = mask_cx * scale_x_mask
            features_dict["final_mask_centroid_y_px"] = mask_cy * scale_y_mask

    # --- 8. Hough Line Transform ---
    lines = cv2.HoughLinesP(final_mask, 1, np.pi / 180,
                            params["hough_threshold"],
                            minLineLength=params["hough_min_length"],
                            maxLineGap=params["hough_max_gap"])

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

    return features_dict

# ------------------------------------------------------------------ STUDENTS --
def predict_steering(img):
    """
    Returns steering prediction from the SVR model based on the input image.

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
            

            predict_steering._model = joblib.load(SVR_MODEL_FILENAME)
            predict_steering._scaler = joblib.load(SCALER_FILENAME)
           
            print(f"[INFO] Feature column order: {predict_steering._feature_columns_order}")
        except FileNotFoundError as e:
           
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
    if model is None or scaler is None:
        return DEFAULT_STEER

    # Convert carla.Image to an OpenCV image (NumPy array)
    # The image is BGRA, so we convert it to BGR
    img_array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    bgr_image = img_array[:, :, :3] # Take only BGR channels

    # 1. Extract features using the dedicated function
    current_features_dict = extract_features_for_prediction(bgr_image, params)

    # Convert features to DataFrame and ensure column order
    input_features_df = pd.DataFrame([current_features_dict])

    try:
        # Ensure the DataFrame has the columns in the order the scaler expects
        input_features_df = input_features_df[feature_columns_order]
    except KeyError as e:
        print(f"[ERROR] Mismatch in feature columns. Missing column: {e}. Check your feature extraction and scaler's expected features.")
        return DEFAULT_STEER # Fallback if features don't match

    # Get the number of detected lines to determine if a line was found
    num_detected_lines_feature = input_features_df.loc[0, 'num_detected_lines']

    # Default steering
    predicted_steering = DEFAULT_STEER

    # Only predict steering if a line was detected
    if num_detected_lines_feature > 0:
        # 2. Scale the extracted features
        scaled_features = scaler.transform(input_features_df)

        # 3. Predict deviation score
        predicted_deviation = model.predict(scaled_features)[0] # .predict returns an array, get the first element

        # 4. Map the predicted deviation to a steering value [-1, 1]
        # This mapping is crucial and depends on what your SVR model was trained to predict.
        # If predicted_deviation is in pixels, you'll need to normalize it.
        # For example, if -100 means far left and 100 means far right, and 0 is center:
        # A simple linear mapping: steering = predicted_deviation / MAX_EXPECTED_DEVIATION
        # You'll need to determine MAX_EXPECTED_DEVIATION based on your training data's range.
        
        # Example: Assume a max deviation of +/- 200 pixels from the center of the image
        # that should correspond to max steering. Adjust this `max_deviation` value.
        
        # Let's assume the `predicted_deviation` represents the horizontal pixel difference
        # from the image center where the lane should be.
        # A good starting point for mapping is to normalize the deviation by the image width
        # and then multiply by a sensitivity factor.
        
        # Determine the effective width of the cropped image for normalization
        # Use the actual cropped width for more accurate normalization
        _ , original_w, _ = bgr_image.shape
        crop_x_start = int(original_w * params["crop_percent_left"] / 100)
        crop_x_end = original_w - int(original_w * params["crop_percent_right"] / 100)
        effective_cropped_width = crop_x_end - crop_x_start

        # Calculate the center of the cropped image
        center_x_cropped = effective_cropped_width / 2.0

        # The 'lane_centroid_x_cropped_px' feature represents the X coordinate in the cropped image
        # So, deviation is `lane_centroid_x_cropped_px - center_x_cropped`
        # Let's directly use the SVR's `predicted_deviation` as a pre-normalized value if your
        # SVR output directly maps to steering, or if it predicts `lane_centroid_x_cropped_px`
        # then calculate the offset here.

        # If your SVR is trained to predict a direct steering value:
        # predicted_steering = predicted_deviation # If SVR directly outputs a steering value from -1 to 1

        # If your SVR is trained to predict the ideal X coordinate of the lane centroid:
        # You need to calculate the deviation from the center of the image.
        # For simplicity, let's assume `predicted_deviation` is the ideal `lane_centroid_x_cropped_px`
        # and we want to steer based on the difference from the center_x_cropped.
        
        # A sensitivity factor will control how aggressively the car steers.
        # This will likely require tuning.
        STEERING_SENSITIVITY = 0.005 # Adjust this value (e.g., 0.001 to 0.01)

        # Assuming `predicted_deviation` from the SVR model is a horizontal offset
        # (e.g., in pixels) from the desired center of the lane in the image.
        # Positive `predicted_deviation` means the lane is to the right of ideal center,
        # so we need to steer left (negative steering).
        # Negative `predicted_deviation` means the lane is to the left of ideal center,
        # so we need to steer right (positive steering).

        # Let's make a strong assumption that `predicted_deviation` already represents
        # a "normalized" deviation from a central point that can be directly scaled to steering.
        # If your SVR was trained to output deviation in pixels, you'll need to divide by a max
        # pixel deviation to normalize.

        # Given your first code used labels -1, 0, 1 for steering, let's try to map the
        # SVR output to that range and then scale to [-1, 1].

        # One way is to treat `predicted_deviation` as a value representing how far off center the car is.
        # Let's say `predicted_deviation` from SVR ranges from roughly -X to +X.
        # We need to map this to -1 to 1.
        # A simple approach: steering = - (predicted_deviation / some_normalizing_factor)
        # The normalizing factor would be the `predicted_deviation` value that should result in max steering.
        
        # A common way to map is to treat the SVR output as a *target x-coordinate* or *deviation*.
        # If `predicted_deviation` is a target x-coordinate (e.g., in pixels in the cropped image):
        # deviation_from_center = predicted_deviation - center_x_cropped
        # steering = - (deviation_from_center / (effective_cropped_width / 2.0)) * STEERING_SENSITIVITY_FACTOR
        # The negative sign is because positive deviation (lane is right) means steer left (negative).

        # For simplicity and given the SVM was -1, 0, 1, let's assume the SVR is trained to
        # output a value where 0 means "center", positive means "too far right", negative means "too far left".
        # We then apply a gain.
        
        # This `STEERING_GAIN` will need careful tuning based on your SVR's output range.
        # If your SVR output range is small (e.g., -5 to 5), this gain might be higher (e.g., 0.1-0.2).
        # If your SVR output range is large (e.g., -200 to 200 pixels), this gain would be very small (e.g., 0.005).
        STEERING_GAIN = 0.01 # Initial guess, TUNE THIS!

        # Assuming positive `predicted_deviation` means the car is to the left of the lane,
        # and needs to steer right (positive steering). Adjust if your model outputs the opposite.
        predicted_steering = predicted_deviation * STEERING_GAIN

    # Ensure the steering value is within the valid range [-1, 1]
    pred_clipped = max(-1.0, min(1.0, predicted_steering))

    # print(f"[INFO] SVR prediction: {predicted_deviation:.3f} -> Steering: {pred_clipped:.3f}")
    return pred_clipped

# ---------------------------- UTILITIES --------------------------------------
def parent_of(actor):
    if hasattr(actor, "get_parent"):
        return actor.get_parent()
    return getattr(actor, "parent", None)

def ang_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def pick_center_camera(world, vehicle):
    v_yaw = vehicle.get_transform().rotation.yaw
    best = None
    for s in world.get_actors().filter("sensor.camera.rgb"):
        p = parent_of(s)
        if p and p.id == vehicle.id:
            delta = abs(ang_diff_deg(s.get_transform().rotation.yaw, v_yaw))
            if best is None or delta < best[0]:
                best = (delta, s)
    return best[1] if best else None
# -----------------------------------------------------------------------------


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT_S)
    world = client.get_world()

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
            print("camera frames: %d   %.1f FPS" % (state["frames"], fps))

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