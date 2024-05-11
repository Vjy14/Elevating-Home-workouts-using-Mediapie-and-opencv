Backend code :

#biceup
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3


def calculate_angle(x, y, z):
    x = np.array(x)  # First joint
    y = np.array(y)  # Mid joint
    z = np.array(z)  # End joint

    radians = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
    left_angle = np.abs(radians * 180.0 / np.pi)

    if left_angle > 180.0:
        left_angle = 360 - left_angle
    return left_angle


# Initialize the text-to-speech engine
engine = pyttsx3.init()
def process_biceupcurl_video():
    cap = cv2.VideoCapture(0)
    right_counter = 0
    right_stage = None
    left_counter = 0
    left_stage = None

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Recolor the image as MEDIAPIPE needs RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make the detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                right_landmarks = results.pose_landmarks.landmark

                # Coordinates of shoulder, elbow, and wrist
                right_shoulder = [right_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  right_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [right_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               right_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [right_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               right_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                left_landmarks = results.pose_landmarks.landmark

                # Coordinates of shoulder, elbow, and wrist
                left_shoulder = [left_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 left_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [left_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                              left_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [left_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              left_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_knee = [left_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             left_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_hip = [left_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            left_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                back_angle = calculate_angle(left_shoulder, left_hip, left_knee)

                if back_angle < 170:
                    print("Keep your back straight!")
                    # engine.say("Keep your back straight!")
                    # engine.runAndWait()

                else:
                    if right_angle > 160:
                        right_stage = "down"
                    if right_angle < 30 and right_stage == 'down':
                        right_stage = "up"
                        right_counter += 1
                        # engine.say("Right curl")
                        # engine.runAndWait()

                    if left_angle > 160:
                        left_stage = "down"
                    if left_angle < 30 and left_stage == 'down':
                        left_stage = "up"
                        left_counter += 1
                        # engine.say("Left curl")
                        # engine.runAndWait()

                # Visualize the angle
                cv2.putText(image, str(right_angle),
                            tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(left_angle),
                            tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(back_angle),
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            # Right Curl Counter render
            cv2.rectangle(image, (0, 0), (275, 85), (245, 117, 16), -1)

            # Right Rep data
            cv2.putText(image, 'RIGHT REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(right_counter), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Right Stage data
            cv2.putText(image, 'RIGHT STAGE', (120, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, right_stage, (110, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Left Curl Counter render
            cv2.rectangle(image, (680, 0), (275, 85), (245, 117, 16), -1)

            # Left Rep data
            cv2.putText(image, 'LEFT REPS', (340, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(left_counter), (340, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Left Stage data
            cv2.putText(image, 'LEFT STAGE', (450, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, left_stage, (450, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(37, 190, 37), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(37, 190, 37), thickness=2, circle_radius=2))

            cv2.imshow('Bicep-Curls', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Breaks out of the video
                break

                # Release the video capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
# Pushup backend code
import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(x, y, z):
    x = np.array(x)  # First joint
    y = np.array(y)  # Mid joint
    z = np.array(z)  # End joint

    radians = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_pushup_video():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    rep_started = False

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Recolor the image as MEDIAPIPE needs RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make the detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Coordinates of shoulder, elbow, and wrist
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Visualize the angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle < 150:
                    if not rep_started:
                        stage = "down"
                        rep_started = True
                if angle > 160 and stage == 'down':
                    stage = "up"
                    counter += 1
                    rep_started = False
                    print(counter)

            except:
                pass

            # Pushup Counter render
            cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Pushups', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Breaks out of the video
                break

        # Release the video capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()

# Squat backend code
import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(x, y, z):
    x = np.array(x)  # First joint
    y = np.array(y)  # Mid joint
    z = np.array(z)  # End joint

    radians = np.arctan2(z[1] - y[1], z[0] - y[0]) - np.arctan2(x[1] - y[1], x[0] - y[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_squat_video():
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    rep_started = False

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Recolor the image as MEDIAPIPE needs RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make the detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Coordinates of hips, knees, and ankles
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                open_angle = calculate_angle(shoulder, hip, right_knee)
                angle = calculate_angle(hip, right_knee, ankle)

                cv2.putText(image, str(angle),
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160:
                    if not rep_started:
                        stage = "up"
                        rep_started = True
                        # engine.say("UP!")
                        # engine.runAndWait()
                if angle < 140 and stage == 'up':
                    stage = "down"
                    counter += 1
                    rep_started = False
                    print(counter)
                    # engine.say("Down!")
                    # engine.runAndWait()

            except:
                pass

            # Squat Counter render
            cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(37, 190, 37), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(37, 190, 37), thickness=2, circle_radius=2))

            cv2.imshow('Squats', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Breaks out of the video
                break

        # Release the video capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
