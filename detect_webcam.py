import cv2
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="runs/detect/train/yolov8n_100epochs/weights/best.pt")
parser.add_argument("--conf", type=float, default=0.4)
args = parser.parse_args()

model = YOLO(args.model)
cap = cv2.VideoCapture(0)

print(f"Using model: {args.model}")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, conf=args.conf, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Litter Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()