import cv2
import time
from sys import argv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing


class HandTracking(mp_hands.Hands):
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        super().__init__(
            static_image_mode,
            max_num_hands,
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def process(self, image):
        results = super().process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

    def draw_landmarks(self, image, results, connections=False):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS if connections else None,
                )

    def draw_circle(self, image, results, idx=0, radius=5, color=(0, 0, 255)):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark[idx]
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), radius, color, -1)


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

    # Create model
    model = HandTracking()

    # Run
    pTime = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess frame
        frame = cv2.flip(frame, 1)

        # Run detection
        results = model.process(frame)

        # Draw landmarks
        model.draw_landmarks(frame, results, True)

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

        cv2.imshow("hand tracking", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
