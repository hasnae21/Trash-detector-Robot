import argparse
import cv2
from ultralytics import YOLO

SUPER_CLASSES = ['Plastic', 'Metal', 'Glass', 'Paper', 'Other']

# Color per super-class (BGR)
COLORS = {
    0: (0, 165, 255),   # Plastic  → Orange
    1: (0, 0, 255),     # Metal    → Red
    2: (255, 255, 0),   # Glass    → Cyan
    3: (0, 255, 0),     # Paper    → Green
    4: (128, 0, 128),   # Other    → Purple
}

CLASS_MAP = {
    # Plastic → 0
    3: 0, 4: 0, 5: 0, 7: 0, 21: 0, 22: 0, 24: 0, 27: 0,
    29: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0,
    43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 54: 0,
    55: 0, 57: 0,
    # Metal → 1
    0: 1, 1: 1, 2: 1, 8: 1, 10: 1, 11: 1, 12: 1, 28: 1, 50: 1, 52: 1,
    # Glass → 2
    6: 2, 9: 2, 23: 2, 26: 2,
    # Paper / Cardboard → 3
    13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3,
    30: 3, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3, 56: 3,
    # Other → 4
    25: 4, 51: 4, 53: 4, 58: 4, 59: 4,
}

parser = argparse.ArgumentParser(description="Yolov8 inference script")
parser.add_argument(
    "--model",
    type=str,
    default="runs/detect/train/yolov8s_100epochs/weights/best.pt",
    help="path to yolo weights"
)
parser.add_argument(
    "--source",
    type=str,
    default="assets/litter.mp4",
    help="path to data to infer on"
)
parser.add_argument(
    "--conf",
    type=float,
    default=0.3,
    help="confidence threshold (default: 0.3)"
)
parser.add_argument(
    "--save",
    action="store_true",
    help="save predictions"
)

if __name__ == "__main__":
    args = parser.parse_args()

    model = YOLO(args.model)

    for result in model.predict(source=args.source, stream=True, conf=args.conf):
        frame = result.orig_img.copy()

        for box in result.boxes:
            original_cls = int(box.cls)
            super_cls = CLASS_MAP.get(original_cls, 4)
            conf = float(box.conf)
            label = f"{SUPER_CLASSES[super_cls]} {conf:.2f}"
            color = COLORS[super_cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Litter Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
