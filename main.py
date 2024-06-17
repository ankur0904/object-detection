from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)  # webcam
# cap.set(3, 1280)  # width
# cap.set(4, 720)  # height


cap = cv2.VideoCapture("./Videos/7.mp4")
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)


model = YOLO("./fine-tuned/final-bottle-model.pt")

classNames = {0: 'bottle'}

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        print(r)
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 233), 4)

            # Bounding box using 2nd method
            # x1, y1, w, h = box.xywh[0]
            # bbox = int(x1), int(y1), int(w), int(h)
            # cvzone.cornerRect(img, bbox)

            # Confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])
            if conf > 0.8:
                cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=3, thickness=3)
    # imS = cv2.resize(img, (1280, 720))
    cv2.imshow("Image", img)
    cv2.waitKey(1)

