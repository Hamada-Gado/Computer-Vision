import cv2
import time
import random
import logging
from sys import argv
from colour import Color
from collections import namedtuple as NamedTuple
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

logging.basicConfig(level=logging.DEBUG)


def get_distance(p1, p2):
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2


def get_center(vectors):
    if len(vectors) == 0:
        return None

    x = sum([v.x for v in vectors]) / len(vectors)
    y = sum([v.y for v in vectors]) / len(vectors)
    z = sum([v.z for v in vectors]) / len(vectors)

    return NamedTuple("Point", ["x", "y", "z"])(x, y, z)


class FingersTracking(mp_hands.Hands):
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

    def process(self, image):
        results = super().process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

    def draw_landmarks(self, image, results, connections=False):
        if not results.multi_hand_landmarks:
            return

        for hand in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS if connections else None,
                landmark_drawing_spec=self.drawing_spec,
            )

            # Draw center
            center = get_center(
                [
                    hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value],
                    hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP.value],
                    hand.landmark[mp_hands.HandLandmark.PINKY_MCP.value],
                    hand.landmark[mp_hands.HandLandmark.WRIST.value],
                ]
            )
            assert center is not None

            circle_border_radius = max(
                self.drawing_spec[0].circle_radius + 1,
                int(self.drawing_spec[0].circle_radius * 1.2),
            )
            cv2.circle(
                image,
                (int(center.x * image.shape[1]), int(center.y * image.shape[0])),
                circle_border_radius,
                (255, 255, 255),
                self.drawing_spec[0].thickness,
            )
            cv2.circle(
                image,
                (int(center.x * image.shape[1]), int(center.y * image.shape[0])),
                self.drawing_spec[0].circle_radius,
                (255, 0, 255),
                self.drawing_spec[0].thickness,
            )

    def draw_count(self, image, results, hand_no):
        if not results.multi_hand_landmarks:
            return

        # Get center
        hand = results.multi_hand_landmarks[hand_no]
        center = get_center([hand.landmark[i] for i in range(21)])
        assert center is not None

        # Draw score
        cv2.putText(
            image,
            f"{self.getFingerCount(results, hand_no)}",
            (int(center.x * image.shape[1]), int(center.y * image.shape[0])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 220, 0),
            2,
        )

    def isOpen(self, results, hand_no, finger_no):
        hand = results.multi_hand_landmarks[hand_no]
        d1 = get_distance(
            hand.landmark[finger_no],
            hand.landmark[0],
        )
        d2 = get_distance(
            hand.landmark[finger_no - 2],
            hand.landmark[0],
        )

        return d1 > d2

    def isThumbOpen(self, results, hand_no):
        hand = results.multi_hand_landmarks[hand_no]

        center = get_center(
            [
                hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value],
                hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP.value],
                hand.landmark[mp_hands.HandLandmark.PINKY_MCP.value],
                hand.landmark[mp_hands.HandLandmark.WRIST.value],
            ]
        )
        assert center is not None

        dist_1 = get_distance(
            hand.landmark[mp_hands.HandLandmark.THUMB_TIP.value], center
        )
        dist_2 = get_distance(
            hand.landmark[mp_hands.HandLandmark.THUMB_MCP.value], center
        )

        return dist_1 > dist_2

    def getFingerCount(self, results, hand_no):
        if not results.multi_hand_landmarks or hand_no >= len(
            results.multi_hand_landmarks
        ):
            return None

        # Thumb
        thumb = self.isThumbOpen(results, hand_no)

        # Index
        index = self.isOpen(
            results, hand_no, mp_hands.HandLandmark.INDEX_FINGER_TIP.value
        )

        # Middle
        middle = self.isOpen(
            results, hand_no, mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value
        )

        # Ring
        ring = self.isOpen(
            results, hand_no, mp_hands.HandLandmark.RING_FINGER_TIP.value
        )

        # Pinky
        pinky = self.isOpen(results, hand_no, mp_hands.HandLandmark.PINKY_TIP.value)

        return thumb + index + middle + ring + pinky


class Game:
    WIDTH = 1280
    HEIGHT = 720

    def __init__(self, players_num):
        self.players_num = players_num
        self.players_score = [0] * players_num

        self.num = -1
        self.fps = -1
        self.paused = True
        self.draw_fps = False
        self.draw_landmakrs = False

        # Create model
        self.model = FingersTracking(
            max_num_hands=players_num,
            min_tracking_confidence=0.4,
        )

    def calc_score(self, results):
        if not results.multi_hand_landmarks:
            return

        for i in range(self.players_num):
            # Get number of fingers
            fingers = self.model.getFingerCount(results, i)
            logging.debug(f"Player {i} has {fingers} fingers")

            if fingers is None:
                continue

            # Check if correct
            if fingers == self.num:
                self.players_score[i] += 1
                self.num = -1

    def draw_score(self, image, results, hand_no):
        if not results.multi_hand_landmarks or hand_no >= len(
            results.multi_hand_landmarks
        ):
            return

        # Get center
        hand = results.multi_hand_landmarks[hand_no]
        center = get_center([hand.landmark[i] for i in range(21)])
        assert center is not None

        # Draw score
        cv2.putText(
            image,
            f"{self.players_score[hand_no]}",
            (int(center.x * image.shape[1]), int(center.y * image.shape[0])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50, 220, 220),
            2,
        )

    def draw(self, frame, results):
        # Draw number
        if self.num != -1:
            cv2.putText(
                frame,
                f"{self.num}",
                (Game.WIDTH // 2, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (100, 120, 220),
                3,
            )

        # Draw paused
        if self.paused:
            cv2.putText(
                frame,
                "PAUSED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (20, 20, 220),
                3,
            )

        # Draw landmarks
        if self.draw_landmakrs:
            self.model.draw_landmarks(frame, results, True)

        # Draw score
        for i in range(self.players_num):
            self.draw_score(frame, results, i)

        # Display FPS
        if self.draw_fps:
            cv2.putText(
                frame,
                f"FPS: {self.fps:.0f}",
                (Game.WIDTH - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 220, 0),
                2,
            )

    def run(self):
        # Set video source
        cap = cv2.VideoCapture(argv[1])
        if not cap.isOpened():
            print("Error opening video file")
            exit(1)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Game.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Game.HEIGHT)

        # Set video writer
        out = cv2.VideoWriter(
            "vods/output.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (Game.WIDTH, Game.HEIGHT),
        )

        pTime = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Get random number
            if self.num == -1:
                self.num = random.randint(0, 5)

            # Preprocess frame
            frame = cv2.flip(frame, 1)

            # Run detection
            results = self.model.process(frame)

            # Calculate score
            if not self.paused:
                self.calc_score(results)

            # Calculate FPS
            cTime = time.time()
            self.fps = 1 / (cTime - pTime)
            pTime = cTime

            # Draw
            self.draw(frame, results)
            cv2.imshow("Gripper Counter", frame)
            out.write(frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("p"):
                self.paused = not self.paused
            if key == ord("d"):
                self.draw_landmakrs = not self.draw_landmakrs
            if key == ord("f"):
                self.draw_fps = not self.draw_fps

        cap.release()
        out.release()
        cv2.destroyAllWindows()


def main():
    if len(argv) < 2:
        print("Usage: python obj_detc.py <video_path>")
        exit(1)

    game = Game(players_num=1 if len(argv) < 3 else int(argv[2]))
    game.run()


if __name__ == "__main__":
    main()
