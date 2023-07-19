import os
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

def get_boundaries(lines):
    # Filter and separate lines as left and right lane boundaries
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    # Fit lines to left and right lane boundaries if lines were detected
    left_lane = np.mean(left_lines, axis=0, dtype=np.int32) if left_lines else None
    right_lane = np.mean(right_lines, axis=0, dtype=np.int32) if right_lines else None

    # Extend lines
    height, width = binary_mask.shape
    y1 = height - 1  # Bottom of the image
    y2 = int(height * 0.6)  # Extend lines up to 40% of the image height
    left_lane_extended = None
    right_lane_extended = None

    if left_lane is not None:
        left_x1, left_y1, left_x2, left_y2 = left_lane
        left_slope = (left_y2 - left_y1) / (left_x2 - left_x1)
        left_x1_extended = int(left_x1 + (y1 - left_y1) / left_slope)
        left_x2_extended = int(left_x1 + (y2 - left_y1) / left_slope)
        left_lane_extended = np.array([left_x1_extended, y1, left_x2_extended, y2], dtype=np.int32)

    if right_lane is not None:
        right_x1, right_y1, right_x2, right_y2 = right_lane
        right_slope = (right_y2 - right_y1) / (right_x2 - right_x1)
        right_x1_extended = int(right_x1 + (y1 - right_y1) / right_slope)
        right_x2_extended = int(right_x1 + (y2 - right_y1) / right_slope)
        right_lane_extended = np.array([right_x1_extended, y1, right_x2_extended, y2], dtype=np.int32)
    
    return left_lane_extended, right_lane_extended, y1, y2 

def middle_lane(left_lane, right_lane, y1, y2):
    # Calculate the middle line
    middle_x_bottom = int((left_lane[0] + right_lane[0]) / 2)
    middle_x_top = int((left_lane[2] + right_lane[2]) / 2)
    middle_y_bottom = y1
    middle_y_top = y2

    return middle_x_bottom, middle_x_top, middle_y_bottom, middle_y_top
def get_angle(middle_x_bottom, middle_x_top, middle_y_bottom, middle_y_top):
    # Calculate the angle of the middle line
    delta_x = middle_x_top - middle_x_bottom
    delta_y = middle_y_top - middle_y_bottom
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.abs(np.degrees(angle_rad))
    deg_off = 90 - angle_deg
    
    font_color = (50, 200, 0)
    if deg_off < -10 or deg_off > 10:
        font_color = (0, 0, 255)
        
    # Display the angle on the frame
    cv2.putText(frame, "Angle: {:.2f} degrees".format(deg_off), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, font_color, 2)

def set_threshold(left_lane, right_lane, threshold, threshold_from_frame, center_of_cam, threshold_x, cam_center_x):
    if threshold_from_frame:
        # Set threshold coordinates
        l_threshold_x_bottom = left_lane[0] + threshold
        l_threshold_x_top = left_lane[2] + threshold

        r_threshold_x_bottom = right_lane[0] - threshold
        r_threshold_x_top = right_lane[2] - threshold

        if not center_of_cam:
            cam_center_x = middle_x_bottom

    else: # If threshold is manually set

        # Angle from y-axis (change to modify the threshold inclination)
        angle_from_yaxis = 5 
        angle = np.radians(90 + angle_from_yaxis) 

        # Set threshold coordinates
        l_threshold_x_bottom = threshold_x - threshold
        l_threshold_x_top = l_threshold_x_bottom + (y2 - y1) / np.tan(angle)


        r_threshold_x_bottom = threshold_x + threshold
        r_threshold_x_top = r_threshold_x_bottom + (y2 - y1) / np.tan(-angle)

        if not center_of_cam:
            cam_center_x = threshold_x

    return l_threshold_x_bottom, l_threshold_x_top, r_threshold_x_bottom, r_threshold_x_top, cam_center_x
# Load a model
model = YOLO('/Users/joaquinarias/Documents/Jobs/DFKI/RoLand/YOLO/runs/segment/train/weights/best.pt')  # Load a custom model

# Create VideoCapture object to read the input video
video_capture = cv2.VideoCapture('/Users/joaquinarias/Downloads/webcam_lane.mp4')

save = False # Save the video
threshold_from_frame = False # Take the threshold automatically from frame 
threshold = 50 # Threshold size 
threshold_x = 300 # x coordinate of threshold
center_of_cam = False # Use the center of the image as center to calculate displacement
there_is_threshold = None

if save:
    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    output_video = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))

if center_of_cam:
    cam_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_center_x = cam_width // 2
else:
    cam_center_x = None


while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    # Predict with the model
    results = model(frame)
    masks = results[0].masks if results[0].masks is not None else None 
    if masks is not None:
        mask = masks.data[0].numpy()  

        # Create a binary image from the mask
        binary_mask = (mask > 0).astype(np.uint8) * 255

        # Apply Canny edge detection
        edges = cv2.Canny(binary_mask, 50, 150)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        left_lane, right_lane, y1, y2 = get_boundaries(lines)

        # Calculate the middle points based on x-coordinates of the lane boundaries
        if left_lane is not None and right_lane is not None:
            middle_x_bottom, middle_x_top, middle_y_bottom, middle_y_top = middle_lane(left_lane, right_lane, y1, y2)

            if there_is_threshold is None:
                l_threshold_x_bottom, l_threshold_x_top, r_threshold_x_bottom, r_threshold_x_top, cam_center_x = set_threshold(left_lane, right_lane, threshold, threshold_from_frame, center_of_cam, threshold_x, cam_center_x)
            
            if l_threshold_x_bottom is not None and r_threshold_x_bottom is not None:
                there_is_threshold = True
                threshold_image = np.zeros_like(frame)
                cv2.line(threshold_image, (int(l_threshold_x_bottom), middle_y_bottom), (int(l_threshold_x_top), middle_y_top), (255, 0, 0), thickness=3)
                cv2.line(threshold_image, (int(r_threshold_x_bottom), middle_y_bottom), (int(r_threshold_x_top), middle_y_top), (255, 0, 0), thickness=3)
            # Get the angle
            get_angle(middle_x_bottom, middle_x_top, middle_y_bottom, middle_y_top)

            # Calculate the distancce from center
            distance = middle_x_bottom - cam_center_x

            # Display the distance on the frame
            cv2.putText(frame, "Distance: {:.2f} pixels".format(distance), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, "|", (cam_center_x, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)


            # Draw the detected lane boundaries and the middle line on the frame
            lane_image = np.zeros_like(frame)
            
            '''
            if left_lane_extended is not None:
                cv2.line(lane_image, (left_lane_extended[0], left_lane_extended[1]), (left_lane_extended[2], left_lane_extended[3]), (255, 0, 0), thickness=5)
            if right_lane_extended is not None:
                cv2.line(lane_image, (right_lane_extended[0], right_lane_extended[1]), (right_lane_extended[2], right_lane_extended[3]), (255, 0, 0), thickness=5)
            '''

            # Determine if the middle of the lane intersects with the threshold
            line_color = (0, 255, 0)
            if middle_x_bottom < l_threshold_x_bottom or middle_x_bottom > r_threshold_x_bottom or middle_x_top < l_threshold_x_top or middle_x_top > r_threshold_x_top:
                line_color = (0, 0, 255)

            # Draw the middle line
            cv2.line(lane_image, (middle_x_bottom, middle_y_bottom), (middle_x_top, middle_y_top), line_color, thickness=5)

 
            lane_overlay = cv2.addWeighted(lane_image, 1, threshold_image, 0.5, 0)
            # Combine the lane image with the original frame
            result = cv2.addWeighted(frame, 1, lane_overlay, 0.5, 0)
            
            if save:
                output_video.write(result)

            # Display the resulting frame
            cv2.imshow('Lane Detection', result)
    else:
        if save:
            output_video.write(frame)
        # Display the resulting frame
        cv2.imshow('Lane Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and VideoWriter objects
video_capture.release()

if save:
    output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
