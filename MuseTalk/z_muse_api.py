"""HTTP-based API implementation, depreciated"""

import modal
import modal.gpu
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

app = modal.App("z_muse_lipsync")
volume = modal.Volume.from_name("lipsync-vol")
image = modal.Image.debian_slim().apt_install(
    "libgl1-mesa-glx", 
    "libglib2.0-0", 
    "ffmpeg"
).apt_install(
    "portaudio19-dev",
).pip_install_from_requirements(
    "requirements.txt"
).pip_install(
    "openmim",
    "boto3",
    "fastapi",
    "websockets",
    "setuptools",
    "SpeechRecognition",
    "pyaudio",
).run_commands(
    "mim install mmengine", 
    "mim install mmcv>=2.0.1",
    "mim install mmdet>=3.1.0",
    "mim install mmpose>=1.1.0",
    "export FFMPEG_PATH=/vol/ffmpeg",
)

s3_bucket_name = "bucketwithsamename"  # Bucket name not ARN.
s3_access_credentials = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "AKIAVOHZJ3MLJK5UGRTJ",
    "AWS_SECRET_ACCESS_KEY": "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",
    "AWS_REGION": "us-east-1"
})

with image.imports():
    import configs
    import scripts
    import results
    import data

    from scripts.realtime_inference import main, init
    import base64
    import numpy as np
    import json
    import time

    import os
    print("Current working directory:", os.getcwd())
    print("Folders and files in current working directory:")
    print(os.listdir())


@app.cls(image=image, gpu=modal.gpu.A10G(count=1), volumes={"/vol": volume}, timeout=1200)
class AvatarProcessor:
    @modal.enter()
    def start(self):
        print("\nInitializing AvatarProcessor")
        self.avatar = init()
        print("AvatarProcessor initialized\n")
    
    @modal.method()
    def get_binary_image(self, audio_data):
        print("\nGetting binary image from audio data")
        for frame in main(audio_data, self.avatar):
            yield frame
        print(" ! Get Binary Image Function Call Complete!!\n")


class Input(BaseModel):
    audio_data: str

@app.function(image=image, 
              volumes={"/vol": volume}, 
              timeout=1200, 
              container_idle_timeout=500, 
              concurrency_limit=1)
@modal.web_endpoint(method="POST")
def start(input: Input):
    print("\n\nProcessing audio data...")
    processor = AvatarProcessor()

    audio_data = base64.b64decode(input.audio_data)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    sample_rate = 44100
    duration_ms = len(audio_np) / sample_rate * 1000
    print(f"Audio duration: {duration_ms:.2f} milliseconds\n")
    
    def generate_frames():
        start_time = time.time()
        frame_count = 0
        try:
            for frame_number, binary_image in enumerate(processor.get_binary_image.remote_gen(audio_np), 1):
                if binary_image == b"DONE":
                    print(" ! All frames processed")
                    yield json.dumps({"done": True})
                    break
                elif binary_image == b"ERROR":
                    print(" ! Error occurred during frame generation")
                    yield json.dumps({"error": True})
                    return
                
                image_base64 = base64.b64encode(binary_image).decode('utf-8')
                print(f"\nFRAME {frame_number}")
                print(f"   > B64 ENCODED | SIZE: {len(image_base64)} | TYPE: {type(image_base64)}")
                print(f"   > IMAGE: ..{image_base64[-30:]}")
                yield json.dumps({
                    "frame": frame_number,
                    "image": image_base64
                })
                frame_count += 1
        except Exception as e:
            print(f" ! Error in generate_frames: {str(e)}")
            yield json.dumps({"error": str(e)})
        finally:
            end_time = time.time()
            total_time = end_time - start_time
            frames_per_second = frame_count / total_time if total_time > 0 else 0
            print(" ! Frame generation complete")
            print(f" ! Frames sent per second: {frames_per_second:.2f}")
    return StreamingResponse(generate_frames(), media_type="text/event-stream")