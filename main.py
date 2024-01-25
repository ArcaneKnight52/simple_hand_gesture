import cv2
import mediapipe as mp

class HandGestureDetector:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.hand_detector = mp.solutions.hands.Hands()

    def detect_gesture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = self.get_gesture(hand_landmarks)
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                    )
            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def get_gesture(self, landmarks):
        thumb_tip_y = landmarks.landmark[4].y
        index_tip_y = landmarks.landmark[8].y
        thumbs_up_threshold = 0.1
        if thumb_tip_y < index_tip_y - thumbs_up_threshold:
            return "Thumbs Up"
        elif thumb_tip_y > index_tip_y and thumb_tip_y > 0:  
            return "Thumbs Down"
        else:
            return "Undefined"

if __name__ == "__main__":
    gesture_detector = HandGestureDetector()
    gesture_detector.detect_gesture()