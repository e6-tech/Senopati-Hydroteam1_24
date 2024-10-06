import cv2
from ultralytics import YOLO
import time
import math


class BallDetector:
    def __init__(self, model_path, video_path,  max_lr, frame_width=640, frame_height=640):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x_frame = frame_width // 2
        self.left_x_frame = max_lr
        self.right_x_frame = frame_width-max_lr
        self.pivot_point = (frame_width // 2, frame_height)
        self.bottom_pivot = 340
        self.class_names = {0: 'green_ball', 1: 'red_ball'}
        # Hijau untuk green_ball, merah untuk red_ball
        self.class_colors = [(0, 255, 0), (0, 0, 255)]

    def draw_center_line(self, frame, fps, green, red, mid):
        """Menggambar garis tengah pada frame video."""
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if green is not None:
            green_center_x, green_center_y = green[4]
            cv2.putText(frame, f'Green: ({green_center_x:.0f}, {green_center_y:.0f})', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if red is not None:
            red_center_x, red_center_y = red[4]
            cv2.putText(frame, f'Red: ({red_center_x:.0f}, {red_center_y:.0f})', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, f'Mid: ({mid[0]:.0f}, {mid[1]:.0f})', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.line(frame, (self.center_x_frame, 0),
                 (self.center_x_frame, self.frame_height), (255, 255, 255), 2)
        cv2.line(frame, (self.left_x_frame, 0),
                 (self.left_x_frame, self.frame_height), (255, 255, 255), 2)
        cv2.line(frame, (self.right_x_frame, 0),
                 (self.right_x_frame, self.frame_height), (255, 255, 255), 2)
        cv2.line(frame, (0, self.bottom_pivot),
                 (self.frame_width, self.bottom_pivot), (255, 255, 255), 2)

    def calculate_distance(self, point1, point2):
        """Menghitung jarak Euclidean antara dua titik."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_mid_point(self, point1, point2):
        """Menghitung titik tengah antara dua titik."""
        return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    def detect(self, set_device='0', confident=0.6, iou_thres=0.7):
        global mid_point
        mid_point = 320, 320
        """Fungsi utama untuk menjalankan deteksi."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(
                frame, (self.frame_width, self.frame_height))

            start_time = time.time()
            results = self.model.predict(
                source=resized_frame, imgsz=640, conf=confident, iou=iou_thres, max_det=8, device=set_device)
            end_time = time.time()

            fps = 1 / (end_time - start_time)
            closest_green_ball = None
            closest_red_ball = None
            closest_green_distance = float('inf')
            closest_red_distance = float('inf')

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    center_box = ((x1 + x2) / 2, (y1 + y2) / 2)
                    color = self.class_colors[cls] if cls < len(
                        self.class_colors) else (255, 255, 255)
                    detected_class_name = self.class_names.get(cls, 'Unknown')

                    distance_to_pivot = self.calculate_distance(
                        center_box, self.pivot_point)

                    if center_box[1] < self.bottom_pivot:
                        if cls == 0 and distance_to_pivot < closest_green_distance:
                            closest_green_distance = distance_to_pivot
                            closest_green_ball = (
                                x1, y1, x2, y2, center_box, color)
                        if cls == 1 and distance_to_pivot < closest_red_distance:
                            closest_red_distance = distance_to_pivot
                            closest_red_ball = (
                                x1, y1, x2, y2, center_box, color)

                    cv2.rectangle(resized_frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), color, 2)
                    label = f'{detected_class_name} {conf:.2f}'
                    cv2.putText(resized_frame, label, (int(x1), int(
                        y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if closest_green_ball and closest_red_ball:
                green_center = closest_green_ball[4]
                red_center = closest_red_ball[4]

                cv2.circle(resized_frame, (int(green_center[0]), int(
                    green_center[1])), 5, (0, 255, 0), -1)
                cv2.circle(resized_frame, (int(red_center[0]), int(
                    red_center[1])), 5, (0, 0, 255), -1)

                mid_point = self.calculate_mid_point(green_center, red_center)
                cv2.line(resized_frame, (int(green_center[0]), int(green_center[1])),
                         (int(red_center[0]), int(red_center[1])), (0, 255, 255), 2)
                cv2.line(resized_frame, (int(mid_point[0]), int(
                    mid_point[1])), self.pivot_point, (0, 255, 255), 2)
                cv2.circle(resized_frame, (int(mid_point[0]), int(
                    mid_point[1])), 5, (255, 0, 0), -1)
                return mid_point

            self.draw_center_line(resized_frame, fps, closest_green_ball,
                                  closest_red_ball, (self.frame_width // 2, 320))
            cv2.imshow('ASV-KKI 2024', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def navigation_data(self):
        return mid_point[0]
