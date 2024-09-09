import cv2
import mediapipe as mp
import pygame
import time

# Initialize MediaPipe BlazePose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utils for visualizing landmarks
mp_drawing = mp.solutions.drawing_utils

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("audio/siren.wav")

# Function to play the siren sound
def play_siren():
    pygame.mixer.music.play()

# Open video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Flag to control siren sound playing
siren_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        # Gesture Detection Logic: Check if both hands are above the shoulders
        left_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        left_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

        # If both hands are above the shoulders, play the siren
        if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
            cv2.putText(frame, "Both hands are raised! This is a danger sign!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Danger sensed!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not siren_playing:  # Play the siren only if it's not already playing
                play_siren()
                siren_playing = True
        else:
            if siren_playing:  # Stop the siren when hands are not raised
                pygame.mixer.music.stop()
                siren_playing = False

    # Display the frame with the pose landmarks
    cv2.imshow('Mediapipe - Gesture Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
