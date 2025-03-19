import sys
import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication, 
    QLabel, 
    QWidget, 
    QVBoxLayout,  
    QHBoxLayout,
    QPushButton
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import time
import struct
import zlib
import websocket
import threading
from collections import deque
import cv2

class WebSocketClient(QThread):
    image_signal = pyqtSignal(QPixmap, float, int, int, int)

    def __init__(self, ws_url, number):
        super().__init__()
        self.ws_url = ws_url
        self.running = True
        self.ws = None
        self.number = number

        self.last_time = time.time()
        self.audio_data = None
        self.waiting_for_response = False
        self.frame_count = 0
        self.send_time = 0
        self.reception_time = 0

        self.request_id = 0

    def run(self):
        def on_message(ws, message):
            if not self.running:
                return

            try:
                frame_number = struct.unpack('>I', message[:4])[0]
                image_data = message[4:]

                if frame_number == 0xFFFFFFFF:  # End of frames
                    self.waiting_for_response = False
                    end_time = time.time()
                    total_time = end_time - self.send_time
                    avg_time_per_frame = total_time / self.frame_count
                    # print(f"Complete! Time Report:\n  Total time: {total_time:.2f}s\n  Frames: {self.frame_count}\n  Avg time per frame: {avg_time_per_frame:.4f}s\n  First frame reception time: {self.reception_time}s\n")
                elif frame_number == 0xFFFFFFFE:  # Error
                    print(" ! Error occurred during frame generation")
                    self.waiting_for_response = False
                else:
                    if self.frame_count == 0:
                        self.reception_time = time.time() - self.send_time

                    image = QImage.fromData(image_data)
                    if not image.isNull():
                        pixmap = QPixmap.fromImage(image)
                        current_time = time.time()
                        fps = 1 / (current_time - self.last_time)
                        self.last_time = current_time
                        
                        self.image_signal.emit(pixmap, fps, self.request_id, frame_number, self.number)
                    self.frame_count += 1
            except Exception as e:
                print(f"      <> Error processing message: {e}")
        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed: {close_status_code} - {close_msg}")

        def on_open(ws):
            print("WebSocket connection opened")

        self.ws = websocket.WebSocketApp(self.ws_url,
                                         on_message=on_message,
                                         on_error=on_error,
                                         on_close=on_close,
                                         on_open=on_open)

        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

        while self.running:
            if self.audio_data is not None:
                compressed_audio = zlib.compress(self.audio_data.tobytes())
                data_to_send = struct.pack('>I', self.request_id) + compressed_audio
                
                audio_duration_ms = len(self.audio_data) / 16000 * 1000
                print(f"Sending REQ #{self.request_id} | DURATION: {audio_duration_ms:.2f} ms | {time.time() - self.send_time:.2f}s | Client: {self.number}")

                self.ws.send_bytes(data_to_send)
                self.send_time = time.time()
                self.frame_count = 0
                self.audio_data = None
                self.waiting_for_response = True
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

    def set_audio_data(self, data):
        self.audio_data = data

class AudioRecorder(QThread):
    audio_data_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        def callback(indata, frames, time_info, status):
            audio_data = indata.copy()
            self.audio_data_signal.emit(audio_data)

        with sd.InputStream(callback=callback, dtype='int16'):
            while self.running:
                sd.sleep(100)

    def stop(self):
        self.running = False

class CameraFeed(QThread):
    frame_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(450, 450, Qt.KeepAspectRatio)
                self.frame_signal.emit(p)
        cap.release()

    def stop(self):
        self.running = False

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio to Image Streamer")
        main_layout = QVBoxLayout()  # Changed to QVBoxLayout

        # Video feeds layout
        video_layout = QHBoxLayout()

        # Left side: Camera feed
        left_layout = QVBoxLayout()
        self.camera_label = QLabel(self)
        placeholder = QPixmap(500, 400)
        placeholder.fill()
        self.camera_label.setPixmap(placeholder)
        left_title = QLabel("My Cam")
        left_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.camera_label)
        video_layout.addLayout(left_layout)

        # Right side: Image and stats
        right_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        placeholder = QPixmap(400, 300)
        placeholder.fill()
        self.image_label.setPixmap(placeholder)
        right_title = QLabel("Real-time Lipsync")
        right_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_title)
        right_layout.addWidget(self.image_label)
        video_layout.addLayout(right_layout)

        main_layout.addLayout(video_layout)

        # Add spacer to push content to the top
        main_layout.addStretch(1)

        # Add start and stop buttons below both videos
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Variables
        self.audio_recorder = None 
        self.camera_feed = None
        self.num_clients = 3
        self.audio_buffer = []
        self.audio_energy_threshold = 10
        self.sample_rate = 16000
        self.time = 0
        self.request_id = 0
        self.last_displayed_request_id = 0

        # Websocket clients
        self.websocket_clients = []
        self.active_client = 1

        # Start the camera feed
        if not self.camera_feed:
            self.camera_feed = CameraFeed()
            self.camera_feed.frame_signal.connect(self.update_camera_feed)
        self.camera_feed.start()

    def start(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Start the audio recorder
        if not self.audio_recorder:
            self.audio_recorder = AudioRecorder()
            self.audio_recorder.audio_data_signal.connect(self.process_audio_data)
        self.audio_recorder.start()
        
        self.init_websocket_clients()
        self.init_timers_and_variables()

    def stop(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.send_timer.stop()

        # Stop the audio recorder
        if self.audio_recorder:
            self.audio_recorder.stop()
            self.audio_recorder.wait()  # Wait for the thread to finish
            self.audio_recorder = None
        
        # Stop the camera feed
        if self.camera_feed:
            self.camera_feed.stop()
            self.camera_feed.wait()
            self.camera_feed = None

        # Clear audio buffer
        self.audio_buffer.clear()

        # Create video from recordings
        # self.create_video()
    
    def update_camera_feed(self, image):
        self.camera_label.setPixmap(QPixmap.fromImage(image))

    def init_websocket_clients(self):
        for i in range(self.num_clients):
            client = WebSocketClient("wss://theindexx--z-muse-lipsync-endpoint-dev.modal.run/ws", i)
            client.image_signal.connect(self.queue_image)
            self.websocket_clients.append(client)
            client.start()
            time.sleep(0.3)

    def init_timers_and_variables(self):
        self.websocket_queue = {
            0: deque(),
            1: deque(),
            2: deque(),
        }
        
        # Timers
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.display_frame)
        self.display_timer.start(33)  # ~30 fps

        self.send_timer = QTimer()
        self.send_timer.timeout.connect(self.send_audio)
        self.send_timer.start(300)  # Send every 300 ms

    def process_audio_data(self, audio_data):
        self.audio_buffer.append(audio_data)

    def send_audio(self):
        client = self.websocket_clients[self.active_client]
        if self.audio_buffer and not client.waiting_for_response:
            combined_audio = np.concatenate(self.audio_buffer)
            samples_to_send = min(int(0.3 * self.sample_rate), len(combined_audio))
            audio_to_send = combined_audio[:samples_to_send]
            
            client.request_start_time = time.time()
            self.request_id += 1
            client.request_id = self.request_id
            client.set_audio_data(audio_to_send)
        elif client.waiting_for_response:
            print(f"Client {self.active_client} is waiting for request {client.request_id}. Length of queue: {len(self.websocket_queue[self.active_client])}")
        
        # print(f"\nActive client: {self.active_client} | Waiting: {client.waiting_for_response} | Switched after {time.time()-self.time:.2f}s")
        # self.time = time.time()
        self.switch_clients()
        self.audio_buffer = []

    def queue_image(self, pixmap, fps, request_id, frame_number, number):
        if frame_number == 1:  # if first frame, clear the queue
            self.websocket_queue[number].clear()
        self.websocket_queue[number].append((pixmap, fps, request_id, frame_number))

    def display_frame(self):
        if len(self.websocket_queue[self.active_client]) > 0: # if there are frames to display
            pixmap, fps, request_id, frame_number = self.websocket_queue[self.active_client][0]
            
            if request_id >= self.last_displayed_request_id:
                self.websocket_queue[self.active_client].popleft()
                print(f">  Displaying {frame_number} | Client: {self.active_client} | Length: {len(self.websocket_queue[self.active_client]) + 1} | {time.time()-self.time:.2f}s | Request: {request_id}")
                self.time = time.time()
                
                self.image_label.setPixmap(pixmap)
                self.last_displayed_request_id = request_id
            else:
                removed_count = 0
                while self.websocket_queue[self.active_client] and self.websocket_queue[self.active_client][0][2] < self.last_displayed_request_id:
                    _, _, old_request_id, old_frame_number = self.websocket_queue[self.active_client].popleft()
                    removed_count += 1
                print(f"X  Discarded {removed_count} frame(s) | Client: {self.active_client} | Request: {old_request_id} (older than {self.last_displayed_request_id})")

                if len(self.websocket_queue[self.active_client]) > 0:
                    self.display_frame()
        else:
            pass

    def handle_error(self, error):
        print(f"Media player error: {error}")

    def switch_clients(self):
        self.active_client += 1
        if self.active_client > self.num_clients - 1:
            self.active_client = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
