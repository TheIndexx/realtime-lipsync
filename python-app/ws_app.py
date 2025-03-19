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
from collections import deque, defaultdict
import cv2

class WebSocketClient(QThread):
    image_signal = pyqtSignal(QPixmap, int, int)
    request_sent_signal = pyqtSignal(int, float)
    ready_signal = pyqtSignal()

    def __init__(self, ws_url):
        super().__init__()
        self.ws_url = ws_url
        self.running = True
        self.ws = None
        self.is_ready = False
        
        self.audio_data = None
        self.pending_requests = {}
        self.frame_count = 0
        self.send_time = 0
        self.reception_time = 0
        
        self.request_id = 0
    
    def run(self):
        def on_message(ws, message):
            if not self.running:
                return
            
            if message == b"READY":
                self.is_ready = True
                self.ready_signal.emit()
                return
            
            request_id, frame_number = struct.unpack('>II', message[:8])
            image_data = message[8:]
            
            if frame_number == 0xFFFFFFFF:
                self.pending_requests.pop(request_id)
                # end_time = time.time()
                # total_time = end_time - self.send_time
                # avg_time_per_frame = total_time / self.frame_count
                # print(f"Complete! Time Report:\n  Total time: {total_time:.2f}s\n  Frames: {self.frame_count}\n  Avg time per frame: {avg_time_per_frame:.4f}s\n  First frame reception time: {self.reception_time}s\n")
            elif frame_number == 0xFFFFFFFE:
                print(" ! Error occurred during frame generation")
            else:
                if self.frame_count == 0:
                    self.reception_time = time.time() - self.send_time

                image = QImage.fromData(image_data)
                if not image.isNull():
                    pixmap = QPixmap.fromImage(image)
                    self.image_signal.emit(pixmap, request_id, frame_number)
                
                self.frame_count += 1

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
                self.request_id += 1
                audio_duration_ms = len(self.audio_data) / 16000 * 1000
                print(f"Sending REQ #{self.request_id} | DURATION: {audio_duration_ms:.2f} ms | {time.time() - self.send_time:.2f}s")

                self.pending_requests[self.request_id] = {'start_time': time.time(),}
                
                compressed_audio = zlib.compress(self.audio_data.tobytes())
                current_time = time.time()
                data_to_send = struct.pack('>Id', self.request_id, current_time) + compressed_audio
                self.ws.send_bytes(data_to_send)
                
                self.send_time = time.time()
                self.frame_count = 0
                self.audio_data = None

                self.request_sent_signal.emit(self.request_id, self.send_time)
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
        self.audio_buffer = []
        self.sample_rate = 16000
        self.last_displayed_request_id = 0
        self.cleaned_up_requests = set()

        # Websocket clients
        self.websocket_client = None
        self.websocket_ready = False

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
    
    def init_websocket_clients(self):
        client = WebSocketClient("wss://theindexx--z-muse-lipsync-endpoint-dev.modal.run/ws")
        client.image_signal.connect(self.queue_image)
        client.request_sent_signal.connect(self.track_request_send_time)
        client.ready_signal.connect(self.on_websocket_ready)
        self.websocket_client = client
        client.start()
    
    def on_websocket_ready(self):
        print("WebSocket is ready. Starting to send audio data.")
        self.websocket_ready = True
        self.init_timers_and_variables()
    
    def init_timers_and_variables(self):
        # Initialize separate queues
        self.websocket_queues = defaultdict(deque)
        self.current_request_id = 1
        self.expected_frame_number = {}
        self.request_send_times = {}
        
        # Timers
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.display_frame)
        self.display_timer.start(33)  # ~30 fps
        
        self.send_timer = QTimer()
        self.send_timer.timeout.connect(self.send_audio)
        self.send_timer.start(300)  # Send every 300 ms
    
    def track_request_send_time(self, request_id, send_time):
        self.request_send_times[request_id] = send_time
    
    def process_audio_data(self, audio_data):
        self.audio_buffer.append(audio_data)
    
    def send_audio(self):
        if self.websocket_ready and self.audio_buffer:
            combined_audio = np.concatenate(self.audio_buffer)
            samples_to_send = min(int(0.3 * self.sample_rate), len(combined_audio))
            audio_to_send = combined_audio[:samples_to_send]
            
            self.websocket_client.set_audio_data(audio_to_send)
        self.audio_buffer = []
    
    def queue_image(self, pixmap, request_id, frame_number):
        if request_id in self.cleaned_up_requests:
            # print(f"Ignoring frame for cleaned-up request ID: {request_id}")
            return
        if request_id not in self.websocket_queues:
            self.expected_frame_number[request_id] = 1
        self.websocket_queues[request_id].append((frame_number, pixmap))
    
    def display_frame(self):
        if self.current_request_id not in self.expected_frame_number:
            return
        
        queue = self.websocket_queues[self.current_request_id] # Get the queue for the current request
        send_time = self.request_send_times.get(self.current_request_id, None) # Get the send time for the current request
        current_time = time.time() # Get the current time
        
        if send_time is None:
            print(f"Send time not found for Request ID: {self.current_request_id}. Skipping processing.")
            return
        
        elapsed_time = current_time - send_time
        
        if 0.6 <= elapsed_time <= 0.9:
            # Time to display frames
            if queue:
                frame_number, pixmap = queue.popleft()
                expected = self.expected_frame_number[self.current_request_id]
                
                if frame_number == expected:
                    # Display the frame
                    self.image_label.setPixmap(pixmap)
                    self.last_displayed_request_id = self.current_request_id
                    self.expected_frame_number[self.current_request_id] += 1
                    print(f">  {self.current_request_id}: Displaying Frame {frame_number}")
                elif frame_number > expected:
                    # Missing frames, skip to current frame
                    print(f">  {self.current_request_id}: Displaying Frame {frame_number} (Skipped {frame_number - expected} frames)")
                    self.image_label.setPixmap(pixmap)
                    self.last_displayed_request_id = self.current_request_id
                    self.expected_frame_number[self.current_request_id] = frame_number + 1
                else:
                    # Duplicate or out-of-order frame, discard
                    print(f">  {self.current_request_id}: Discarded Frame {frame_number} (Expected {expected})")
            else:
                # No frames to display yet; possibly waiting for frames
                pass
        elif elapsed_time > 0.9:
            # Elapsed time > 0.9 seconds, stop processing this request
            print(f"Request ID: {self.current_request_id} exceeded time window ({elapsed_time:.2f}s). Discarding frames.")
            if not (self.current_request_id in self.cleaned_up_requests):
                self.cleanup_request(self.current_request_id)
                self.current_request_id += 1
            else:
                self.cleanup_request(self.current_request_id)

            self.expected_frame_number[self.current_request_id] = 1
            print(f"\n>  Processing Request ID: {self.current_request_id}")
    
    def cleanup_request(self, request_id):
        """Clean up data related to a specific request_id."""
        if request_id in self.websocket_queues:
            del self.websocket_queues[request_id]
        if request_id in self.expected_frame_number:
            del self.expected_frame_number[request_id]
        if request_id in self.request_send_times:
            del self.request_send_times[request_id]
                
        # Add request_id to cleaned_up_requests set
        self.cleaned_up_requests.add(request_id)
    
    def stop(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.send_timer.stop()
        self.display_timer.stop()
        
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
        
        # Stop the websocket client
        if self.websocket_client:
            self.websocket_client.stop()
            self.websocket_client.wait()
            self.websocket_client = None
        
        # Clear audio buffer
        self.audio_buffer.clear()
        
        # Clear websocket queues
        self.websocket_queues.clear()
        self.current_request_id = None
        self.expected_frame_number.clear()
        self.request_send_times.clear()
        self.cleaned_up_requests.clear()
    
    def update_camera_feed(self, image):
        self.camera_label.setPixmap(QPixmap.fromImage(image))
    
    def handle_error(self, error):
        print(f"Media player error: {error}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
