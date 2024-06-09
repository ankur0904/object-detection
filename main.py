from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0)  # webcam
# cap.set(3, 1280)  # width
# cap.set(4, 720)  # height


cap = cv2.VideoCapture("../Videos/6.mp4")
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)


model = YOLO("../yolo-weights/yolov8l.pt")

classNames = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

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
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=3, thickness=3)
    imS = cv2.resize(img, (1280, 720))
    cv2.imshow("Image", imS)
    cv2.waitKey(0)
