import cv2
import mediapipe as mp
import time

# Video face detector

capture = cv2.VideoCapture('acgt.mp4')

faceDetector = mp.solutions.face_detection

with faceDetector.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:

    while True:

        isTrue, frame = capture.read()
        start = time.time()

        if not isTrue:
            print("Ignoring empty camera frame.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                if detection.score[0] > 0.5:
                    # Creating a bounding box
                    b_box = detection.location_data.relative_bounding_box
                    ht, wd, ch = frame.shape
                    b_boxD = int(b_box.xmin * wd), int(b_box.ymin * ht), \
                        int(b_box.width * wd), int(b_box.height * ht)
                    cv2.rectangle(frame, b_boxD, (0, 0, 255), 2)
                    cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (b_boxD[0], b_boxD[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

capture.release()