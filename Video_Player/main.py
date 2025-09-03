import os
import time
import sys
import cv2
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QSizePolicy, QPushButton, QHBoxLayout, QSlider
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class VideoPlayer(QWidget):
    def __init__(self, video_path):
        super().__init__()

        self.setWindowTitle("Deepfake Video Player")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 150)

        # Timer label
        self.timer_label = QLabel(self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 14px; padding: 5px;")

        # Seek bar
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderReleased.connect(self.seek_position)
        self.slider.sliderPressed.connect(lambda: self.timer.stop())

        # Play/Pause button
        self.playing = True
        self.detection_enabled = False
        self.detection_result = ""
        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_play_pause)

        self.detect_button = QPushButton("Enable Detection")
        self.detect_button.setCheckable(True)
        self.detect_button.clicked.connect(self.toggle_detection)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.slider)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.play_button)
        info_layout.addWidget(self.detect_button)
        info_layout.addStretch()
        info_layout.addWidget(self.timer_label)
        info_layout.addStretch()

        layout.addLayout(info_layout)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError("Cannot open video: " + video_path)

        # Extract frame count and FPS
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.fps if self.fps > 1 else 24
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_time_sec = self.total_frames / self.fps
        self.current_frame = 0
        self.slider.setRange(0, self.total_frames)

        self.predicted_results = {}  # timestamp -> (label, confidence)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_next_frame)
        self.timer.start(int(1000 / self.fps))

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02}:{secs:02}"

    def display_next_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            elapsed_time = self.current_frame / self.fps

            frame = self.resize_and_crop(frame, self.video_label.width(), self.video_label.height())

            timestamp = round(self.current_frame / self.fps, 2)

            # Show result if it's available
            if self.detection_enabled and timestamp in self.predicted_results:
                label, confidence = self.predicted_results[timestamp]
                self.detection_result = f"{label} ({confidence:.2f}) at {self.format_time(timestamp)}"

            # Predict for next second
            next_timestamp = round(timestamp + 1, 2)
            if self.detection_enabled and next_timestamp not in self.predicted_results and self.current_frame % int(self.fps) == 0:
                # Prepare copy of frame for async processing
                detection_frame = frame.copy()
                threading.Thread(
                    target=self.async_predict_frame,
                    args=(detection_frame, next_timestamp),
                    daemon=True
                ).start()

            if self.detection_result:
                label_text = self.detection_result
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                box_coords = ((10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 20))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

                color = (0, 0, 255) if "Deepfake" in label_text else (0, 255, 0)
                cv2.putText(frame, label_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, color, 2, cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

            self.timer_label.setText(
                f"{self.format_time(elapsed_time)} / {self.format_time(self.total_time_sec)}"
            )
            self.slider.setValue(self.current_frame)
        else:
            self.timer.stop()
            self.playing = False
            self.play_button.setText("Play")

    def toggle_play_pause(self):
        if self.playing:
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(int(1000 / self.fps))
            self.play_button.setText("Pause")
        self.playing = not self.playing

    def resize_and_crop(self, frame, target_width, target_height):
        return cv2.resize(frame, (target_width, target_height))

    def seek_position(self):
        frame_number = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        self.timer.start(int(1000 / self.fps))

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        if self.detection_enabled:
            self.detect_button.setText("Disable Detection")
        else:
            self.detect_button.setText("Enable Detection")
        self.detection_result = ""

    def classify_frame_with_api(self, frame):
        try:
            import requests
            import base64
            from PIL import Image
            from io import BytesIO

            # Resize to 256x256 to match model input
            resized = cv2.resize(frame, (256, 256))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            response = requests.post("http://127.0.0.1:8000/predict/", json={"frame": encoded})

            if response.status_code == 200:
                result = response.json()
                label = "Deepfake" if result["confidence"] >= 0.80 else "Real"
                return label, result["confidence"]
            else:
                return "Error", 0.0
        except Exception as e:
            return f"Error", 0.0

    def async_predict_frame(self, frame, timestamp):
        label, confidence = self.classify_frame_with_api(frame)
        self.predicted_results[timestamp] = (label, confidence)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer("R4.mp4")
    player.show()
    sys.exit(app.exec_())