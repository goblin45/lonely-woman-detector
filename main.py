import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load the pre-trained gender classification model
# Replace with the correct path to your pre-trained model
model = load_model('gender_model.hdf5', compile=False)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize HOG people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def classify_gender(face_roi):
    """
    Classifies the gender of the face ROI using the pre-trained model.
    :param face_roi: Region of Interest (face) from the frame.
    :return: 'Female' or 'Male'
    """
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (64, 64))  # Resize face ROI to match model input
    face_normalized = face_resized / 255.0  # Normalize the pixel values
    face_reshaped = np.reshape(face_normalized, (1, 64, 64, 1))  # Reshape for the model input

    # Predict gender (assuming binary output 0: Male, 1: Female)
    prediction = model.predict(face_reshaped)
    return 'Female' if prediction[0][0] > 0.5 else 'Male'

def detect_lonely_woman(frame):
    """
    Detects whether there is exactly one woman in the frame and no other people.
    :param frame: The camera frame.
    :return: True if a lonely woman is detected, False otherwise.
    """
    # Detect people in the frame
    people, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    if len(people) == 0:
        return False  # No people detected

    women_detected = 0

    # Iterate through detected people
    for (x, y, w, h) in people:
        # Extract face ROI (for simplicity, assume the face is in the upper part of the detected body)
        face_roi = frame[y:y + h // 2, x:x + w]

        # Ensure the face_roi is not empty
        if face_roi.size == 0:
            continue

        # Classify gender
        gender = classify_gender(face_roi)

        if gender == 'Female':
            women_detected += 1

    # A lonely woman is detected if exactly one woman is found and no other people
    return women_detected == 1 and len(people) == 1

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        males = 0
        females = 0
        man_cord = 0, 0

        for (x, y, w, h) in faces:
            # Get the Region of Interest (face) from the frame
            face_roi = frame[y:y + h, x:x + w]

            # Classify the gender of the detected face
            gender = classify_gender(face_roi)
            lonely_woman = False
            if gender == 'Male':
                males += 1
                man_cord = x, y
                # lonely_woman = 
            else: 
                females += 1
            
            # if males == 1:
            #     cv2.putText(frame, 'Lonely Man',  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # else:
            cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Draw a rectangle around the face and display the gender label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, 'Males: ' + str(males), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, 'Females: ' + str(females), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if females == 1:
            cv2.putText(frame, 'Lonely Woman',  (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Gender Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
