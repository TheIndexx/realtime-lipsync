import modal
import modal.gpu

app = modal.App("rt_lipsync")
volume = modal.Volume.from_name("lipsync-vol")
image = modal.Image.debian_slim().apt_install(
    "libgl1-mesa-glx", 
    "libglib2.0-0", 
    "portaudio19-dev",
).pip_install(
    "setuptools",
    "SpeechRecognition",
    "torch",
    "numpy",
    "openai-whisper",
    "pyaudio",
    "fastapi",
    "websockets",
)

with image.imports():
    from queue import Queue
    from datetime import datetime, timedelta, UTC
    import time

@app.cls(image=image, gpu=modal.gpu.A10G(count=1), volumes={
    "/vol": volume,
})
class Transcription:
    def __init__(self):
        self.phrase_time = None
        self.phrase_timeout = 3
        
    
    @modal.enter()
    def load(self):
        import whisper

        print("\n\nLoading model...")
        self.audio_model = whisper.load_model("/vol/whisper/tiny.pt")
        print("\n\nModel loaded.")

    @modal.method()
    def transcribe(self, audio_data):
        import torch
        import numpy as np
        print("Starting transcription...")

        # try:
        #     if len(audio_data) % 16 != 0:
        #         trim_size = len(audio_data) - (len(audio_data) // 16 * 16)
        #         audio_data = audio_data[:-trim_size]
        #         print(f"Trimmed audio data size: {len(audio_data)}")
        #     audio_np = np.frombuffer(audio_data, dtype=np.int16)
        # except ValueError:
        #     print("Failed to load audio data as int16, trying uint8...")
        #     # If that fails, try uint8
        #     audio_np = np.frombuffer(audio_data, dtype=np.uint8)
        #     # Convert uint8 to int16
        #     audio_np = (audio_np.astype(np.int16) - 128) * 256

        # # Convert to float32 and normalize
        # audio_np = audio_np.astype(np.float32) / 32768.0
        # print(f"Audio shape: {audio_np.shape}")

        if len(audio_data) % 32 != 0:
            trim_size = len(audio_data) - (len(audio_data) // 32 * 32)
            audio_data = audio_data[:-trim_size]
            print(f"Trimmed audio data size: {len(audio_data)}")
        
        audio_np = np.frombuffer(audio_data, dtype=np.int32)
        print("Example audio data:", audio_np[:10])
        audio_np = audio_np.astype(np.float32) / (2 ** 31 - 1)
        print(f"Example audio data ({audio_np.dtype}):", audio_np[:10])

        result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        print(f"Transcription: {result['text'].strip()}")
        return result['text'].strip()
    
    @modal.method()
    def process_audio(self, queue_data):
        print("Processing audio, data length:", len(queue_data))
        now = datetime.now(UTC)
        phrase_complete = False
        
        if queue_data:
            if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                phrase_complete = True
            self.phrase_time = now

            print("Phrase complete:", phrase_complete)

            audio_data = b''.join(queue_data)

            text = self.transcribe.remote(audio_data)
            return text, phrase_complete
        
        return None, False

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, WebSocket

    web_app = FastAPI()

    @web_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        t = Transcription()
        data_queue = Queue()
        last_process_time = datetime.now(UTC)
        
        try:
            while True:
                data = await websocket.receive_bytes()
                print(f"\nReceived audio data: {len(data)} bytes")
                
                data_queue.put(data)
                print(f"Queue size: {data_queue.qsize()}")
                
                now = datetime.now(UTC)
                if now - last_process_time > timedelta(seconds=3):
                    queue_data = list(data_queue.queue)
                    data_queue.queue.clear()
                    text, phrase_complete = t.process_audio.remote(queue_data)
                    if text:
                        await websocket.send_text(text)
                        if phrase_complete:
                            await websocket.send_text("[PHRASE_COMPLETE]")
                    last_process_time = now
                else:
                    time.sleep(0.1)

        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await websocket.close()


    return web_app