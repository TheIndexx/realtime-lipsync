import modal
import modal.gpu

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

s3_bucket_name = "bucketwithsamename"
s3_access_credentials = modal.Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "AKIAVOHZJ3MLJK5UGRTJ",
    "AWS_SECRET_ACCESS_KEY": "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",
    "AWS_REGION": "us-east-1"
})

with image.imports():
    # import configs
    import scripts
    import data

    from scripts.realtime_inference import main, init
    from fastapi import FastAPI, WebSocket
    import struct
    import zlib
    import time
    import numpy as np
    import random


@app.cls(
    image=image, 
    gpu=modal.gpu.A10G(), 
    volumes={"/vol": volume, "/mount": modal.CloudBucketMount(s3_bucket_name, secret=s3_access_credentials)}, 
    timeout=1200,
    keep_warm=3,
    concurrency_limit=3
)
class AvatarProcessor:
    @modal.enter()
    def start(self):
        print("\nInitializing AvatarProcessor")
        self.avatar = init()
        print("AvatarProcessor initialized\n")
    
    @modal.method()
    def get_binary_image(self, audio_data, request_id=1):
        for frame in main(audio_data, self.avatar, request_id):
            yield frame


@app.function(
    image=image, 
    volumes={
        "/vol": volume,
        "/mount": modal.CloudBucketMount(s3_bucket_name, secret=s3_access_credentials)
    }
)
@modal.asgi_app()
def endpoint():
    web_app = FastAPI()
    random_num = random.randint(1, 1000)
    
    @web_app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        
        print("WebSocket connection established.")
        processor = AvatarProcessor()
        count = 0
        sample_rate = 16000
        
        # try:
        while True:
            data = await websocket.receive_bytes()
            start_time = time.time()
            time_to_first_frame = 0
            count += 1

            request_id = struct.unpack('>I', data[:4])[0]
            
            compressed_audio = data[4:]
            data = zlib.decompress(compressed_audio)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            image_size = []
            for frame_number, binary_image in enumerate(processor.get_binary_image.remote_gen(audio_np, request_id), 1):
                if frame_number == 1:
                    time_to_first_frame = time.time() - start_time
                if binary_image == b"DONE":
                    await websocket.send_bytes(struct.pack('>I', 0xFFFFFFFF)) # Send 4 bytes of 0xFFFFFFFF to indicate end of frames 
                    break
                elif binary_image == b"ERROR":
                    print(" ! Error occurred during frame generation")
                    await websocket.send_bytes(struct.pack('>I', 0xFFFFFFFE))  # Special frame number for "error"
                    return
                
                
                # print(f"   > FRAME {frame_number} | ORIGINAL: {len(binary_image)} | COMPRESSED: {len(compressed_binary_image)}")
                
                image_size.append(len(binary_image))
                await websocket.send_bytes(struct.pack('>I', frame_number) + binary_image)
            # print("Complete! Report:")
            # print(f"  First frame: {time_to_first_frame}s")
            # print(f"  Last frame: {time.time() - start_time}s")
            # # print(f"  Time per frame: {(time.time() - start_time) / frame_number}s")
            # # print(f"  ID: {random_num} | Request number: {count}")
            # print(f"  Average size: {sum(image_size) / len(image_size)}")
            # print(f"  Request ID: {request_id}\n")
            # # print(f"  Audio duration: {len(audio_np) / sample_rate * 1000:.2f} milliseconds\n\n")
        # except Exception as e:
        #     print(f" ! Error in generate_frames: {str(e)}")
        #     await websocket.send_bytes(struct.pack('>I', 0xFFFFFFFE))  # Special frame number for "error"
        
    return web_app