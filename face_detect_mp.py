import cv2
import glob
import mediapipe as mp

# Static image face detector

# Load images from a folder
images_folder = glob.glob("images/*.jpg")

# Loop through all the images
for img_path in images_folder:
    print("Image path", img_path)
    image = cv2.imread(img_path)

    faceDetector = mp.solutions.face_detection

    with faceDetector.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                if detection.score[0] > 0.5:

                    # Creating a bounding box
                    b_box = detection.location_data.relative_bounding_box
                    ht, wd, ch = image.shape
                    b_boxD = int(b_box.xmin * wd), int(b_box.ymin * ht), \
                        int(b_box.width * wd), int(b_box.height * ht)
                    cv2.rectangle(image, b_boxD, (0, 0, 255), 2)
                    cv2.putText(image, f'{int(detection.score[0] * 100)}%', (b_boxD[0], b_boxD[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
        cv2.imshow('Face Detector', image)
        if cv2.waitKey(3000) & 0xFF == ord('q'):
            break