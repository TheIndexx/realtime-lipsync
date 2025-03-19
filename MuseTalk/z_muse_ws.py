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
    "AWS_ACCESS_KEY_ID": modal.Secret.from_name("AWS_ACCESS_KEY_ID").get(),
    "AWS_SECRET_ACCESS_KEY": modal.Secret.from_name("AWS_SECRET_ACCESS_KEY").get(),
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
    import asyncio


class AvatarProcessor:
    def __init__(self) -> None:
        print("\nInitializing AvatarProcessor")
        self.last_completion_time = 0
        self.avatar = init()
        print("AvatarProcessor initialized\n")
    
    async def get_binary_image(self, audio_data, request_id=1):
        for frame in main(audio_data, self.avatar, request_id):
            yield frame

current_requests = 0

async def process_request(websocket, data, processor):
    global current_requests
    
    current_requests += 1
    time_to_first_frame = 0
    request_id, request_time = struct.unpack('>Id', data[:12])

    start_time = time.time()
    print(f"#{request_id} | {start_time - request_time}s to process | {current_requests} requests")
    if start_time - request_time > 1:
        print(f" ! Cancelling\n")
        current_requests -= 1
        await websocket.send_bytes(struct.pack('>II', request_id, 0xFFFFFFFD))
        return
    
    compressed_audio = data[12:]
    data = zlib.decompress(compressed_audio)
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        frame_number = 1
        async for binary_image in processor.get_binary_image(audio_np, request_id):
            if frame_number == 1:
                time_to_first_frame = time.time() - start_time
            
            if binary_image == b"DONE":
                await websocket.send_bytes(struct.pack('>II', request_id, 0xFFFFFFFF))
                break
            elif binary_image == b"ERROR":
                print("Error during frame generation")
                await websocket.send_bytes(struct.pack('>II', request_id, 0xFFFFFFFE))
                return
            
            await websocket.send_bytes(struct.pack('>II', request_id, frame_number) + binary_image)
            frame_number += 1
        
        print(f"Completed {request_id}! Report:")
        print(f"  First frame: {time_to_first_frame}s")
        print(f"  Last frame: {time.time() - start_time}s")
        print(f"  Request ID: {request_id}")
        # print(f"  Time per frame: {(time.time() - start_time) / frame_number}s for {frame_number} frames")
        # print(f"  Average size: {sum(image_size) / len(image_size)}")
        # print(f"  Audio duration: {len(audio_np) / 16000 * 1000:.2f} milliseconds")
        
        current_completion_time = time.time()
        if processor.last_completion_time > 0:
            time_since_last_completion = current_completion_time - processor.last_completion_time
            print(f"  Time since last completion: {time_since_last_completion}s")
        processor.last_completion_time = current_completion_time
        
        print()
    except Exception as e:
        print(f"Error in process_request: {str(e)}")
        await websocket.send_bytes(struct.pack('>I', 0xFFFFFFFE))
    finally:
        current_requests -= 1


@app.function(image=image, gpu=modal.gpu.A10G(count=1), volumes={
    "/vol": volume,
    "/mount": modal.CloudBucketMount(s3_bucket_name, secret=s3_access_credentials)
}, timeout=600)
@modal.asgi_app()
def endpoint():
    web_app = FastAPI()
    MAX_CONCURRENT_REQUESTS = 3
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    @web_app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        
        print("WebSocket connection established.")
        processor = AvatarProcessor()
        
        await websocket.send_bytes(b"READY")

        async def process_request_wrapper(websocket, data, processor):
            async with semaphore:
                await process_request(websocket, data, processor)
        
        while True:
            data = await websocket.receive_bytes()
            asyncio.create_task(process_request_wrapper(websocket, data, processor))
        # except Exception as e:
        #     print(f" ! Error in generate_frames: {str(e)}")
        #     await websocket.send_bytes(struct.pack('>I', 0xFFFFFFFE))  # Special frame number for "error"
        
    return web_app