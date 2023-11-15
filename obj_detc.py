import cv2
import time
import torch
from sys import argv
from ultralytics import YOLO


def main():
    if len(argv) < 2:
        print("Usage: python obj_detc.py <video_path>")
        exit(1)

    # Set video source
    WIDTH = 1280
    HEIGHT = 720

    cap = cv2.VideoCapture(argv[1])
    if not cap.isOpened():
        print("Error opening video file")
        exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Load model
    torch.cuda.set_device(0)
    path = "yolo_models/yolov8s.pt"
    model = YOLO(path).cuda()

    # Run
    pTime = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess frame
        frame = cv2.flip(frame, 1)

        # Run detection
        results = model(frame, verbose=False, stream=True)

        # Draw bounding boxes
        for res in results:
            for box in res.boxes:
                if box.conf[0] < 0.5:
                    continue

                x1, y1, x2, y2 = list(map(int, box.xyxy[0]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{model.names[int(box.cls)]} {box.conf[0]:.2f}",
                    (max(20, x1), max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (50, 155, 255),
                    2,
                )

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("YOLOv8n", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
