import cv2
import time
from sys import argv
from colour import Color
import logging
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

logging.basicConfig(level=logging.CRITICAL + 1)


def distance(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2


class GripperTracking(mp_hands.Hands):
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
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

        colors = Color("violet").range_to(Color("red"), 21)
        self.drawing_spec = {
            i: mp_drawing.DrawingSpec(color=tuple(map(lambda x: int(x * 255), c.rgb)))
            for i, c in zip(range(21), colors)
        }

        self.reps = 0
        self.in_rep = False

    def process(self, image):
        results = super().process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

    def draw_landmarks(self, image, results, connections=False):
        if not results.multi_hand_landmarks:
            return

        mp_drawing.draw_landmarks(
            image,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS if connections else None,
            landmark_drawing_spec=self.drawing_spec,
        )

    def count_reps(self, results):
        if not results.multi_hand_landmarks:
            return

        middle_finger_tip = results.multi_hand_landmarks[0].landmark[
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]
        middle_finger_mcp = results.multi_hand_landmarks[0].landmark[
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        ]
        ring_finger_tip = results.multi_hand_landmarks[0].landmark[
            mp_hands.HandLandmark.RING_FINGER_TIP
        ]
        ring_finger_mcp = results.multi_hand_landmarks[0].landmark[
            mp_hands.HandLandmark.RING_FINGER_MCP
        ]
        wrist = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]

        middletip_to_wrist = distance(middle_finger_tip, wrist)
        middlemcp_to_wrist = distance(middle_finger_mcp, wrist)
        ringtip_to_wrist = distance(ring_finger_tip, wrist)
        ringmcp_to_wrist = distance(ring_finger_mcp, wrist)

        if (
            middletip_to_wrist < middlemcp_to_wrist
            and ringtip_to_wrist < ringmcp_to_wrist
            and not self.in_rep
        ):
            self.reps += 1
            self.in_rep = True
            logging.info("*" * 10)
            logging.info(f"Reps: {self.reps}")
            logging.info("*" * 10)

        self.in_rep = (
            middletip_to_wrist < middlemcp_to_wrist
            and ringtip_to_wrist < ringmcp_to_wrist
        )


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

    # Set video writer
    out = cv2.VideoWriter(
        "vods/output.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (WIDTH, HEIGHT),
    )

    # Create model
    model = GripperTracking()
    draw = False

    # Run
    pTime = 0
    pause = True
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Preprocess frame
        frame = cv2.flip(frame, 1)

        # Run detection
        results = model.process(frame)

        # Draw landmarks
        if draw:
            model.draw_landmarks(frame, results, True)

        # Draw reps
        if not pause:
            model.count_reps(results)
        cv2.putText(
            frame,
            f"Reps: {model.reps}",
            (620, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 128, 128),
            2,
        )

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 220, 0),
            2,
        )

        # Pause
        if pause:
            cv2.putText(
                frame,
                "Press 'p' to start",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 220, 0),
                2,
            )

        cv2.imshow("Gripper Counter", frame)
        out.write(frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("p"):
            pause = not pause
        if key == ord("r"):
            model.reps = 0
        if key == ord("d"):
            draw = not draw

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
