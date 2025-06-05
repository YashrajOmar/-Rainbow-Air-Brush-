import cv2
import numpy as np
import mediapipe as mp
from enum import Enum
import math

class Tools(Enum):
    PEN = 1
    ERASER = 2

class DrawingCanvas:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.thickness = 15
        self.eraser_thickness = 125
        self.current_tool = Tools.PEN
        self.drawing = False
        self.start_point = None
        self.hue = 0  # For rainbow color cycling

    def draw(self, point):
        if not self.drawing or self.start_point is None:
            return

        if self.current_tool == Tools.PEN:
            bgr_colour = self.hsv_to_bgr(self.hue, 1, 1)
            cv2.line(self.canvas, self.start_point, point, bgr_colour, self.thickness)
            self.hue = (self.hue + 2) % 180
        elif self.current_tool == Tools.ERASER:
            cv2.line(self.canvas, self.start_point, point, (0, 0, 0), self.eraser_thickness)

        self.start_point = point

    def start_drawing(self, point):
        self.drawing = True
        self.start_point = point

    def stop_drawing(self):
        self.drawing = False
        self.start_point = None

    def set_tool(self, tool: Tools):
        self.current_tool = tool

    def get_display(self):
        return self.canvas.copy()

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    @staticmethod
    def hsv_to_bgr(h, s, v):
        hsv_pixel = np.uint8([[[h, int(s*255), int(v*255)]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr_pixel)

class UIManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def draw_text(self, frame, text, x, y, font_scale=1, color=(255, 255, 255), thickness=2):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA, False)

    def draw(self, frame, last_audio_command="None"):
        self.draw_text(frame, f"Last audio command: {last_audio_command}", x=60, y=40)

class HandTracker:
    def __init__(self, max_hands=1, min_det_conf=0.7, min_track_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_hand_position(self, frame, hand_number=0):
        landmark_list = []
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_number:
                hand = self.results.multi_hand_landmarks[hand_number]
                for id, landmark in enumerate(hand.landmark):
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append((id, cx, cy))
        return landmark_list

class GestureType(Enum):
    NONE = "none"
    DRAW = "draw"
    ERASE = "erase"
    SELECT = "select"

class GestureRecogniser:
    def __init__(self, pinch_threshold=75):
        self.pinch_threshold = pinch_threshold

    def recognise_gesture(self, landmark_list):
        if not landmark_list:
            return GestureType.NONE

        landmarks = dict([(id, (x, y)) for id, x, y in landmark_list])
        pinch_dist = self._distance(landmarks[4], landmarks[8])
        fingers_extended = self._fingers_extended(landmarks)

        if pinch_dist < self.pinch_threshold:
            return GestureType.DRAW
        if all(fingers_extended):
            return GestureType.ERASE
        if self._is_select_gesture(landmarks, fingers_extended):
            return GestureType.SELECT

        return GestureType.NONE

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _fingers_extended(self, landmarks):
        palm_x = sum(landmarks[i][0] for i in [0, 5, 9, 13, 17]) / 5
        thumb_extended = landmarks[4][0] < palm_x
        fingers = []
        for tip, mid, base in [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]:
            fingers.append(landmarks[tip][1] < landmarks[mid][1] < landmarks[base][1])
        return [thumb_extended] + fingers

    def _is_select_gesture(self, landmarks, fingers_extended):
        index_extended = fingers_extended[1]
        others_curled = not any(fingers_extended[2:])
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        dx = index_tip[0] - index_pip[0]
        dy = index_tip[1] - index_pip[1]
        angle = abs(math.degrees(math.atan2(dx, -dy)))
        is_vertical = angle < 30
        index_tip_y = index_tip[1]
        other_tips_y = [landmarks[i][1] for i in [12, 16, 20]]
        is_highest = all(index_tip_y < y for y in other_tips_y)
        return index_extended and others_curled and is_vertical and is_highest

def main():
    width, height = 1280, 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    canvas = DrawingCanvas(width, height)
    ui = UIManager(width, height)
    hand_tracker = HandTracker()
    gesture_recogniser = GestureRecogniser()
    drawing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hand_tracker.find_hands(frame, draw=True)
        landmarks = hand_tracker.get_hand_position(frame)

        index_finger_tip = next(((x, y) for id, x, y in landmarks if id == 8), None)
        gesture = gesture_recogniser.recognise_gesture(landmarks)

        if gesture == GestureType.DRAW:
            canvas.set_tool(Tools.PEN)
            if index_finger_tip:
                if not drawing:
                    canvas.start_drawing(index_finger_tip)
                    drawing = True
                else:
                    canvas.draw(index_finger_tip)
        elif gesture == GestureType.ERASE:
            canvas.set_tool(Tools.ERASER)
            if index_finger_tip:
                if not drawing:
                    canvas.start_drawing(index_finger_tip)
                    drawing = True
                else:
                    canvas.draw(index_finger_tip)
        else:
            if drawing:
                canvas.stop_drawing()
                drawing = False

        canvas_img = canvas.get_display()
        combined = cv2.addWeighted(frame, 1, canvas_img, 0.7, 0)
        ui.draw(combined)
        cv2.imshow("AirCanvas - Gesture Drawing", combined)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break
        elif key & 0xFF == ord("c"):
            canvas.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
