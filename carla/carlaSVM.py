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
        model_filename = "svm_line_classifier.joblib"
        scaler_filename = "scaler_line_features.joblib"

        predict_steering._model = None
        predict_steering._scaler = None
        predict_steering._feature_columns_order = None
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
            "hough_threshold": 57,
            "hough_min_length": 18,
            "hough_max_gap": 17,
            "crop_percent": 55,
            "line_center_tolerance_percent": 10
        }

        try:
            predict_steering._model = joblib.load(model_filename)
            predict_steering._scaler = joblib.load(scaler_filename)
            predict_steering._feature_columns_order = predict_steering._scaler.feature_names_in_.tolist()
            print(f"[INFO] SVM model and scaler loaded successfully from '{model_filename}' and '{scaler_filename}'.")
            print(f"[INFO] Feature column order retrieved from scaler: {predict_steering._feature_columns_order}")
        except FileNotFoundError:
            print(f"[ERROR] Model or scaler file not found. Ensure '{model_filename}' and '{scaler_filename}' are in the correct directory.")
            print(f"[WARN] Only random steering will be used as a fallback.")
        except Exception as e:
            print(f"[ERROR] Error loading model or scaler: {e}")
            print(f"[WARN] Only random steering will be used as a fallback.")

    model = predict_steering._model
    scaler = predict_steering._scaler
    feature_columns_order = predict_steering._feature_columns_order
    OPTIMIZED_PARAMS = predict_steering._optimized_params

    # If model or scaler failed to load, return default or random steering
    if model is None or scaler is None or feature_columns_order is None:
        return DEFAULT_STEER # Or random.uniform(-1.0, 1.0) if you prefer

    # Convert carla.Image to an OpenCV image (NumPy array)
    # The image is BGRA, so we convert it to BGR
    img_array = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
    bgr_image = img_array[:, :, :3] # Take only BGR channels

    current_features = {}
    # No need for time profiling here, as it's for a real-time system

    crop_y = int(bgr_image.shape[0] * OPTIMIZED_PARAMS["crop_percent"] / 100)
    if crop_y >= bgr_image.shape[0]:
        crop_y = bgr_image.shape[0] - 1
    cropped_image = bgr_image[crop_y:, :].copy()

    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        # print("Warning: Cropped image is empty. Defaulting to 'No Line Detected' features.")
        current_features = {
            "cx": -1,
            "num_detected_lines": 0,
            "avg_line_length": 0,
            "total_line_length": 0,
            "std_line_length": 0,
            "avg_line_angle_deg": 0,
            "std_line_angle_deg": 0,
            "line_cx_mean": -1,
            "line_cx_std": 0,
            "line_cy_mean": -1,
            "longest_line_length": 0,
            "longest_line_angle_deg": 0,
            "mask_pixel_count": 0,
            "mask_area_ratio": 0,
            "mask_centroid_x_norm": 0.5,
            "mask_centroid_y_norm": 0.5,
            "mask_hu_moment_1": 0, "mask_hu_moment_2": 0, "mask_hu_moment_3": 0,
            "mask_hu_moment_4": 0, "mask_hu_moment_5": 0, "mask_hu_moment_6": 0, "mask_hu_moment_7": 0,
            "color_mask_pixel_count": 0,
            "color_mask_area_ratio": 0,
            "is_line_detected_binary": 0
        }
    else:
        lab_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_cropped)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        blurred_lab_eq = cv2.medianBlur(lab_eq, 5)

        lower_orange_lab = np.array([OPTIMIZED_PARAMS["lower_L"], OPTIMIZED_PARAMS["lower_A"], OPTIMIZED_PARAMS["lower_B"]])
        upper_orange_lab = np.array([OPTIMIZED_PARAMS["upper_L"], OPTIMIZED_PARAMS["upper_A"], OPTIMIZED_PARAMS["upper_B"]])
        color_mask = cv2.inRange(blurred_lab_eq, lower_orange_lab, upper_orange_lab)

        color_morph_kernel = np.ones((OPTIMIZED_PARAMS["color_morph_kernel_size"], OPTIMIZED_PARAMS["color_morph_kernel_size"]), np.uint8)
        color_mask_morphed = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, color_morph_kernel, iterations=1)
        color_mask_morphed = cv2.dilate(color_mask_morphed, color_morph_kernel, iterations=1)
        color_mask_morphed = cv2.morphologyEx(color_mask_morphed, cv2.MORPH_CLOSE, color_morph_kernel, iterations=1)

        gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray_cropped, (5, 5), 0)
        edge_mask = cv2.Canny(blurred_gray, OPTIMIZED_PARAMS["canny_thresh1"], OPTIMIZED_PARAMS["canny_thresh2"])

        edge_morph_kernel = np.ones((OPTIMIZED_PARAMS["edge_morph_kernel_size"], OPTIMIZED_PARAMS["edge_morph_kernel_size"]), np.uint8)
        edge_mask_morphed = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, edge_morph_kernel, iterations=1)

        final_mask = cv2.bitwise_and(color_mask_morphed, edge_mask_morphed)

        lines = cv2.HoughLinesP(final_mask, 1, np.pi / 180,
                                    OPTIMIZED_PARAMS["hough_threshold"],
                                    minLineLength=OPTIMIZED_PARAMS["hough_min_length"],
                                    maxLineGap=OPTIMIZED_PARAMS["hough_max_gap"])

        num_detected_lines = 0
        avg_line_length = 0
        total_line_length = 0
        std_line_length = 0
        avg_line_angle_deg = 0
        std_line_angle_deg = 0
        line_cx_mean = -1
        line_cx_std = 0
        line_cy_mean = -1
        longest_line_length = 0
        longest_line_angle_deg = 0
        is_line_detected_binary = 0
        cx = -1

        if lines is not None:
            is_line_detected_binary = 1
            num_detected_lines = len(lines)

            all_line_midpoints_x = []
            all_line_midpoints_y = []
            line_lengths = []
            line_angles_rad = []

            max_length_found = 0
            angle_of_longest_line = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                line_lengths.append(length)

                if length > max_length_found:
                    max_length_found = length
                    angle_of_longest_line = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                line_angles_rad.append(angle_rad)

                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                all_line_midpoints_x.append(mid_x)
                all_line_midpoints_y.append(mid_y)

            if line_lengths:
                avg_line_length = np.mean(line_lengths)
                total_line_length = np.sum(line_lengths)
                std_line_length = np.std(line_lengths) if len(line_lengths) > 1 else 0
                longest_line_length = max_length_found
                longest_line_angle_deg = angle_of_longest_line

            if all_line_midpoints_x:
                cx = int(np.mean(all_line_midpoints_x))
                line_cx_mean = cx
                line_cx_std = np.std(all_line_midpoints_x) if len(all_line_midpoints_x) > 1 else 0
                line_cy_mean = int(np.mean(all_line_midpoints_y))

            if line_angles_rad:
                
                avg_line_angle_deg = np.degrees(np.mean(line_angles_rad))
                std_line_angle_deg = np.degrees(np.std(line_angles_rad)) if len(line_angles_rad) > 1 else 0


        mask_pixel_count = np.sum(final_mask > 0)
        mask_area_ratio = mask_pixel_count / (final_mask.shape[0] * final_mask.shape[1]) if (final_mask.shape[0] * final_mask.shape[1]) > 0 else 0

        M = cv2.moments(final_mask)
        mask_centroid_x_norm = 0.5
        mask_centroid_y_norm = 0.5
        hu_moments = np.zeros(7)
        if M["m00"] != 0:
            mask_centroid_x = M["m10"] / M["m00"]
            mask_centroid_y = M["m01"] / M["m00"]
            mask_centroid_x_norm = mask_centroid_x / final_mask.shape[1]
            mask_centroid_y_norm = mask_centroid_y / final_mask.shape[0]
            hu_moments = cv2.HuMoments(M).flatten()
        
        color_mask_pixel_count = np.sum(color_mask_morphed > 0)
        color_mask_area_ratio = color_mask_pixel_count / (color_mask_morphed.shape[0] * color_mask_morphed.shape[1]) if (color_mask_morphed.shape[0] * color_mask_morphed.shape[1]) > 0 else 0

        current_features = {
            "cx": cx,
            "num_detected_lines": num_detected_lines,
            "avg_line_length": avg_line_length,
            "total_line_length": total_line_length,
            "std_line_length": std_line_length,
            "avg_line_angle_deg": avg_line_angle_deg,
            "std_line_angle_deg": std_line_angle_deg,
            "line_cx_mean": line_cx_mean,
            "line_cx_std": line_cx_std,
            "line_cy_mean": line_cy_mean,
            "longest_line_length": longest_line_length,
            "longest_line_angle_deg": longest_line_angle_deg,
            "mask_pixel_count": mask_pixel_count,
            "mask_area_ratio": mask_area_ratio,
            "mask_centroid_x_norm": mask_centroid_x_norm,
            "mask_centroid_y_norm": mask_centroid_y_norm,
            "mask_hu_moment_1": hu_moments[0], "mask_hu_moment_2": hu_moments[1], "mask_hu_moment_3": hu_moments[2],
            "mask_hu_moment_4": hu_moments[3], "mask_hu_moment_5": hu_moments[4], "mask_hu_moment_6": hu_moments[5], "mask_hu_moment_7": hu_moments[6],
            "color_mask_pixel_count": color_mask_pixel_count,
            "color_mask_area_ratio": color_mask_area_ratio,
            "is_line_detected_binary": is_line_detected_binary
        }

    # Convert features to DataFrame and ensure column order
    input_features = pd.DataFrame([current_features])
    # Ensure the DataFrame has the columns in the order the scaler expects
    try:
        input_features = input_features[feature_columns_order]
    except KeyError as e:
        print(f"[ERROR] Mismatch in feature columns. Missing column: {e}. Check your feature extraction and scaler's expected features.")
        return DEFAULT_STEER # Fallback if features don't match

    # Scale the features
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    pred_label = model.predict(input_features_scaled)[0]

    # Example mapping (adjust as needed):
    if pred_label == -1: # Line on Left
        pred_steering = -0.9 # Steer left
    elif pred_label == 0: # Line in Center
        pred_steering = 0.0 # Go straight
    elif pred_label == 1: # Line on Right
        pred_steering = 0.9 # Steer right
    elif pred_label == 2: 
        pred_steering = 0.0  
    else:
        pred_steering = DEFAULT_STEER # Fallback for unknown labels

    pred_clipped = max(-1.0, min(1.0, pred_steering))
    
    # print(f"[INFO] SVM prediction: {pred_label} -> Steering: {pred_clipped:.3f}")
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