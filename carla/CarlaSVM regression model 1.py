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
TIMEOUT_S       = 5.0          # seconds

THROTTLE        = 0.5          # constant forward throttle (0..1)
DEFAULT_STEER   = 0.0          # fallback if no camera frame yet
PRINT_EVERY_N   = 30           # console frames between logs
# -----------------------------------------------------------------------------





# Define a default steer value in case models fail to load or other issues
DEFAULT_STEER = 0.0

def predict_steering(img):
    """
    Returns steering prediction from the SVM regression model based on the input image.
    The prediction is based on the scaled 'cx' (line center) value.

    Parameters
    ----------
    img : carla.Image
        The latest RGB camera frame (BGRA byte-buffer).

    Returns
    -------
    float
        Steering value in [-1, 1].
    """
    # -------------- load the model and scalers only once ---------------------------
    # This block ensures that the model and scalers are loaded into memory
    # only when the function is called for the very first time.
    # Subsequent calls will use the already loaded objects.
    if not hasattr(predict_steering, "_regression_model"):
        # !!! IMPORTANT: Update these filenames for your regression model and scalers !!!
        # These paths assume the .joblib files are in the same directory as this script.
        regression_model_filename = "svr_line_position_model.joblib"
        feature_scaler_filename = "scaler_X.joblib"
        target_scaler_filename = "scaler_y.joblib" # This scaler is for the target 'cx'

        # Initialize attributes to None. They will be populated upon successful loading.
        predict_steering._regression_model = None
        predict_steering._feature_scaler = None
        predict_steering._target_scaler = None
        predict_steering._feature_columns_order = None

        # User-defined optimized parameters for image processing and feature extraction.
        # These should match the parameters used during the training of your model.
        predict_steering._optimized_params = {
            "lower_L": 137,
            "upper_L": 255,
            "lower_A": 134,
            "upper_A": 161,
            "lower_B": 138,
            "upper_B": 165,
            "color_morph_kernel_size": 3,
            "edge_morph_kernel_size": 7,
            "canny_thresh1": 18,
            "canny_thresh2": 66,
            "hough_threshold": 57,      # Min votes for a line
            "hough_min_length": 18,     # Min line length
            "hough_max_gap": 17,        # Max gap to connect segments
            "crop_percent": 55,         # Percentage of image to crop from the top
            "line_center_tolerance_percent": 10 # Not directly used in prediction, but kept for consistency
        }

        try:
            # Attempt to load the regression model and both scalers
            predict_steering._regression_model = joblib.load(regression_model_filename)
            predict_steering._feature_scaler = joblib.load(feature_scaler_filename)
            predict_steering._target_scaler = joblib.load(target_scaler_filename) # Load target scaler for 'cx'
            
            # Retrieve the expected feature column order from the feature scaler.
            # This is crucial to ensure input features are in the correct order for prediction.
            predict_steering._feature_columns_order = predict_steering._feature_scaler.feature_names_in_.tolist()
            
            print(f"[INFO] Regression model and scalers loaded successfully from '{regression_model_filename}', '{feature_scaler_filename}', and '{target_scaler_filename}'.")
            print(f"[INFO] Feature column order retrieved from feature scaler: {predict_steering._feature_columns_order}")
        except FileNotFoundError:
            # Handle cases where model or scaler files are not found.
            print(f"[ERROR] Model or scaler file not found. Ensure '{regression_model_filename}', '{feature_scaler_filename}', and '{target_scaler_filename}' are in the correct directory.")
            print(f"[WARN] Only default steering ({DEFAULT_STEER}) will be used as a fallback.")
        except Exception as e:
            # Handle any other exceptions during loading.
            print(f"[ERROR] Error loading model or scalers: {e}")
            print(f"[WARN] Only default steering ({DEFAULT_STEER}) will be used as a fallback.")

    # Assign loaded components to local variables for easier access within the function.
    model = predict_steering._regression_model
    feature_scaler = predict_steering._feature_scaler
    target_scaler = predict_steering._target_scaler # Access target scaler
    feature_columns_order = predict_steering._feature_columns_order
    OPTIMIZED_PARAMS = predict_steering._optimized_params

    # If any essential component failed to load, return the default steering value immediately.
    if model is None or feature_scaler is None or target_scaler is None or feature_columns_order is None:
        return DEFAULT_STEER

    # --- Convert carla.Image to an OpenCV image (NumPy array) ---
    # carla.Image provides raw_data as a BGRA byte-buffer.
    # Reshape it to (height, width, 4) and then slice to get only BGR channels.
    img_array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    bgr_image = img_array[:, :, :3] # Take only BGR channels

    current_features = {} # Dictionary to store extracted features for the current image.

    # --- Image Processing and Feature Extraction ---
    # Crop the image from the top as specified in OPTIMIZED_PARAMS.
    crop_y = int(bgr_image.shape[0] * OPTIMIZED_PARAMS["crop_percent"] / 100)
    # Ensure crop_y does not exceed image height.
    if crop_y >= bgr_image.shape[0]:
        crop_y = bgr_image.shape[0] - 1
    cropped_image = bgr_image[crop_y:, :].copy()

    # Handle cases where the cropped image might be empty (e.g., if crop_percent is too high).
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        # If the image is empty, set all features to default "no line detected" values.
        # This ensures the DataFrame creation and scaling steps do not fail.
        current_features = {
            "cx": -1.0, "num_detected_lines": 0.0, "avg_line_length": 0.0, "total_line_length": 0.0,
            "std_line_length": 0.0, "avg_line_angle_deg": 0.0, "std_line_angle_deg": 0.0,
            "line_cx_mean": -1.0, "line_cx_std": 0.0, "line_cy_mean": -1.0, "longest_line_length": 0.0,
            "longest_line_angle_deg": 0.0, "mask_pixel_count": 0.0, "mask_area_ratio": 0.0,
            "mask_centroid_x_norm": 0.5, "mask_centroid_y_norm": 0.5,
            "mask_hu_moment_1": 0.0, "mask_hu_moment_2": 0.0, "mask_hu_moment_3": 0.0,
            "mask_hu_moment_4": 0.0, "mask_hu_moment_5": 0.0, "mask_hu_moment_6": 0.0, "mask_hu_moment_7": 0.0,
            "color_mask_pixel_count": 0.0, "color_mask_area_ratio": 0.0, "is_line_detected_binary": 0.0
        }
    else:
        # Convert to LAB color space for color-based segmentation.
        lab_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_cropped)
        
        # Apply CLAHE for contrast enhancement on the L-channel.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        blurred_lab_eq = cv2.medianBlur(lab_eq, 5)

        # Create a color mask based on LAB color thresholds for orange/line color.
        lower_orange_lab = np.array([OPTIMIZED_PARAMS["lower_L"], OPTIMIZED_PARAMS["lower_A"], OPTIMIZED_PARAMS["lower_B"]])
        upper_orange_lab = np.array([OPTIMIZED_PARAMS["upper_L"], OPTIMIZED_PARAMS["upper_A"], OPTIMIZED_PARAMS["upper_B"]])
        color_mask = cv2.inRange(blurred_lab_eq, lower_orange_lab, upper_orange_lab)

        # Apply morphological operations to clean up the color mask.
        color_morph_kernel = np.ones((OPTIMIZED_PARAMS["color_morph_kernel_size"], OPTIMIZED_PARAMS["color_morph_kernel_size"]), np.uint8)
        color_mask_morphed = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, color_morph_kernel, iterations=1)
        color_mask_morphed = cv2.dilate(color_mask_morphed, color_morph_kernel, iterations=1)
        color_mask_morphed = cv2.morphologyEx(color_mask_morphed, cv2.MORPH_CLOSE, color_morph_kernel, iterations=1)

        # Convert to grayscale and apply Gaussian blur for edge detection.
        gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray_cropped, (5, 5), 0)
        
        # Apply Canny edge detector.
        edge_mask = cv2.Canny(blurred_gray, OPTIMIZED_PARAMS["canny_thresh1"], OPTIMIZED_PARAMS["canny_thresh2"])

        # Apply morphological closing to the edge mask to connect broken edge segments.
        edge_morph_kernel = np.ones((OPTIMIZED_PARAMS["edge_morph_kernel_size"], OPTIMIZED_PARAMS["edge_morph_kernel_size"]), np.uint8)
        edge_mask_morphed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, edge_morph_kernel, iterations=1)

        # Combine color mask and edge mask to get the final line mask.
        final_mask = cv2.bitwise_and(color_mask_morphed, edge_mask_morphed)

        # Use Hough Line Transform (Probabilistic) to detect line segments.
        lines = cv2.HoughLinesP(final_mask, 1, np.pi / 180,
                                OPTIMIZED_PARAMS["hough_threshold"],
                                minLineLength=OPTIMIZED_PARAMS["hough_min_length"],
                                maxLineGap=OPTIMIZED_PARAMS["hough_max_gap"])

        # Initialize features with default "no line" values.
        # These defaults are used if no lines are detected or if lists are empty.
        num_detected_lines = 0.0
        avg_line_length = 0.0; total_line_length = 0.0; std_line_length = 0.0
        avg_line_angle_deg = 0.0; std_line_angle_deg = 0.0
        line_cx_mean = -1.0; line_cx_std = 0.0; line_cy_mean = -1.0
        longest_line_length = 0.0; longest_line_angle_deg = 0.0
        is_line_detected_binary = 0.0 # Flag: 0 for no line, 1 for line detected
        cx = -1.0 # Primary feature: overall center of detected lines

        if lines is not None:
            is_line_detected_binary = 1.0 # Set flag to indicate line detection
            num_detected_lines = float(len(lines))
            
            # Lists to store properties of detected line segments.
            all_line_midpoints_x, all_line_midpoints_y, line_lengths, line_angles_rad = [], [], [], []
            max_length_found, angle_of_longest_line = 0.0, 0.0

            for line_segment in lines:
                x1, y1, x2, y2 = line_segment[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                line_lengths.append(length)
                
                # Track the longest line and its angle.
                if length > max_length_found:
                    max_length_found = length
                    angle_of_longest_line = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Calculate angle in radians and store midpoints.
                line_angles_rad.append(np.arctan2(y2 - y1, x2 - x1))
                all_line_midpoints_x.append((x1 + x2) / 2.0)
                all_line_midpoints_y.append((y1 + y2) / 2.0)

            # Calculate aggregate features from line segments if any were found.
            if line_lengths:
                avg_line_length = np.mean(line_lengths)
                total_line_length = np.sum(line_lengths)
                std_line_length = np.std(line_lengths) if len(line_lengths) > 1 else 0.0
                longest_line_length = max_length_found
                longest_line_angle_deg = angle_of_longest_line
            
            if all_line_midpoints_x:
                cx = np.mean(all_line_midpoints_x) # Calculate the mean X-coordinate of line midpoints.
                line_cx_mean = cx
                line_cx_std = np.std(all_line_midpoints_x) if len(all_line_midpoints_x) > 1 else 0.0
                line_cy_mean = np.mean(all_line_midpoints_y)
            
            if line_angles_rad:
                # Normalize angles to be within [0, 180) degrees for consistency.
                normalized_angles_deg = [angle % 180 for angle in np.degrees(line_angles_rad)]
                avg_line_angle_deg = np.mean(normalized_angles_deg)
                std_line_angle_deg = np.std(normalized_angles_deg) if len(normalized_angles_deg) > 1 else 0.0
        
        # --- Features from the final combined mask ---
        mask_pixel_count = float(np.sum(final_mask > 0)) # Count non-zero pixels in the mask.
        mask_area_ratio = mask_pixel_count / (final_mask.shape[0] * final_mask.shape[1]) if (final_mask.shape[0] * final_mask.shape[1]) > 0 else 0.0
        
        # Calculate image moments and Hu moments for shape description.
        M = cv2.moments(final_mask)
        mask_centroid_x_norm, mask_centroid_y_norm = 0.5, 0.5 # Default to center if no moments.
        hu_moments = np.zeros(7) # Initialize Hu Moments array.
        if M["m00"] != 0: # Avoid division by zero if mask is empty.
            mask_centroid_x = M["m10"] / M["m00"]
            mask_centroid_y = M["m01"] / M["m00"]
            if final_mask.shape[1] > 0: mask_centroid_x_norm = mask_centroid_x / final_mask.shape[1]
            if final_mask.shape[0] > 0: mask_centroid_y_norm = mask_centroid_y / final_mask.shape[0]
            hu_moments_calc = cv2.HuMoments(M)
            if hu_moments_calc is not None: hu_moments = hu_moments_calc.flatten()

        # --- Features from the color mask (after morphology) ---
        color_mask_pixel_count = float(np.sum(color_mask_morphed > 0))
        color_mask_area_ratio = color_mask_pixel_count / (color_mask_morphed.shape[0] * color_mask_morphed.shape[1]) if (color_mask_morphed.shape[0] * color_mask_morphed.shape[1]) > 0 else 0.0

        # Populate the current_features dictionary with all calculated features.
        current_features = {
            "cx": cx, "num_detected_lines": num_detected_lines,
            "avg_line_length": avg_line_length, "total_line_length": total_line_length,
            "std_line_length": std_line_length, "avg_line_angle_deg": avg_line_angle_deg,
            "std_line_angle_deg": std_line_angle_deg, "line_cx_mean": line_cx_mean,
            "line_cx_std": line_cx_std, "line_cy_mean": line_cy_mean,
            "longest_line_length": longest_line_length, "longest_line_angle_deg": longest_line_angle_deg,
            "mask_pixel_count": mask_pixel_count, "mask_area_ratio": mask_area_ratio,
            "mask_centroid_x_norm": mask_centroid_x_norm, "mask_centroid_y_norm": mask_centroid_y_norm,
            "mask_hu_moment_1": hu_moments[0] if len(hu_moments) > 0 else 0.0,
            "mask_hu_moment_2": hu_moments[1] if len(hu_moments) > 1 else 0.0,
            "mask_hu_moment_3": hu_moments[2] if len(hu_moments) > 2 else 0.0,
            "mask_hu_moment_4": hu_moments[3] if len(hu_moments) > 3 else 0.0,
            "mask_hu_moment_5": hu_moments[4] if len(hu_moments) > 4 else 0.0,
            "mask_hu_moment_6": hu_moments[5] if len(hu_moments) > 5 else 0.0,
            "mask_hu_moment_7": hu_moments[6] if len(hu_moments) > 6 else 0.0,
            "color_mask_pixel_count": color_mask_pixel_count,
            "color_mask_area_ratio": color_mask_area_ratio,
            "is_line_detected_binary": is_line_detected_binary
        }

    # --- Convert features to DataFrame and ensure column order ---
    # Create a Pandas DataFrame from the extracted features.
    input_features_df = pd.DataFrame([current_features])
    try:
        # Reindex the DataFrame to ensure the columns are in the exact order
        # that the feature scaler expects, filling any missing columns with 0.0.
        input_features_df_ordered = input_features_df.reindex(columns=feature_columns_order, fill_value=0.0)
    except KeyError as e:
        print(f"[ERROR] Mismatch in feature columns. Missing column: {e}. Check your feature extraction and scaler's expected features.")
        return DEFAULT_STEER # Fallback if feature columns don't match.
    except Exception as e:
        print(f"[ERROR] Error reordering feature columns: {e}")
        return DEFAULT_STEER

    # --- Scale the Input Features ---
    try:
        # Transform the ordered input features using the loaded feature scaler.
        input_features_scaled = feature_scaler.transform(input_features_df_ordered)
    except ValueError as e:
        print(f"[ERROR] Error during feature scaling: {e}. Provided shape: {input_features_df_ordered.shape}, Columns: {input_features_df_ordered.columns.tolist()}")
        return DEFAULT_STEER
    except Exception as e:
        print(f"[ERROR] Unexpected error during feature scaling: {e}")
        return DEFAULT_STEER

    # --- Make the Prediction (Output is in Scaled Target Space) ---
    scaled_prediction_value = None

    # Check if a line was detected by OpenCV.
    # If not, override the scaled prediction value to 0.0 as per your requirement.
    if current_features.get("is_line_detected_binary", 1.0) == 0.0:
        # print("[INFO] No line detected by OpenCV. Setting scaled prediction value to 0.0.")
        scaled_prediction_value = 0.0
    else:
        try:
            # If a line was detected, use the regression model to predict the 'cx' value
            # in the scaled target space.
            scaled_prediction_value = model.predict(input_features_scaled)[0]
        except Exception as e:
            print(f"[ERROR] Error during model prediction: {e}")
            return DEFAULT_STEER # Fallback if prediction fails.

    # --- Map the scaled_prediction_value to a steering value in [-1, 1] ---
    # The 'scaled_prediction_value' represents the line's center (cx) in a scaled space.
    # We assume that 0.0 in this scaled space corresponds to the line being centered,
    # negative values mean the line is to the left, and positive values mean it's to the right.
    # A direct mapping is applied, and the result is clipped to the [-1, 1] range.
    # You might need to adjust the scaling factor (e.g., `scaled_prediction_value * K`)
    # if the natural range of your scaled_prediction_value is not directly suitable for [-1, 1] steering.
    steering = scaled_prediction_value # Direct mapping of scaled cx to steering.
    
    # Clip the steering value to ensure it's within the valid range [-1.0, 1.0].
    pred_clipped = max(-1.0, min(1.0, steering))
    
    # Optional: print the scaled prediction and final steering for debugging.
    # print(f"[INFO] Scaled Prediction Value (cx): {scaled_prediction_value:.4f} -> Steering: {pred_clipped:.3f}")
    
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