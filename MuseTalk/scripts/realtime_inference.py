import os
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image_prepare_material,get_image_blending, get_image_blending_v2
from musetalk.utils.utils import load_all_model
import shutil
import queue
import time
import boto3
import concurrent.futures

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    """
    Extracts frames from a video file and saves them as individual images.

    Args:
        vid_path (str): Path to the input video file.
        save_path (str): Directory path to save the extracted frames.
        ext (str, optional): File extension for saved images. Defaults to '.png'.
        cut_frame (int, optional): Maximum number of frames to extract. Defaults to 10000000.

    This function reads a video file frame by frame, converts each frame to an image,
    and saves it in the specified directory. It stops when either all frames are processed
    or the number of extracted frames reaches the cut_frame limit.

    The saved images are named sequentially with 8-digit zero-padded numbers (e.g., 00000001.png, 00000002.png, etc.).

    Note: Ensure that the save_path directory exists before calling this function.
    """
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def read_aws_imgs(img_list):
    s3 = boto3.client(
        's3', 
        aws_access_key_id="AKIAVOHZJ3MLJK5UGRTJ",
        aws_secret_access_key="PF+cCjG+6lDbebGr4pyO9ugPMW8NsIdtoguoxmwY"
    )
    bucket_name = "bucketwithsamename"

    def fetch_and_decode_image(img_path):
        key = img_path.replace("/mount/", "")
        response = s3.get_object(Bucket=bucket_name, Key=key)
        img_data = response['Body'].read()
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print('Reading images...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        frames = list(tqdm(executor.map(fetch_and_decode_image, img_list), total=len(img_list)))
    
    return frames


@torch.no_grad() 
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"/mount/results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path= f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path =f"{self.avatar_path}/mask"
        self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id":avatar_id,
            "video_path":video_path,
            "bbox_shift":bbox_shift   
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()
        
    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_aws_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_aws_imgs(input_mask_list)

                print(f"\n\nCoord list cycle: {self.coord_list_cycle[0]}")
                print(f"Frame list cycle: {len(self.frame_list_cycle)}")
                print(f"Mask coords list cycle: {len(self.mask_coords_list_cycle)}")
                print(f"Mask list cycle: {len(self.mask_list_cycle)}\n\n")
            else:
                print("*********************************")
                print(f"  creating avatar: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
                self.prepare_material()
        else: 
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)
                
            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:  
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)
    
    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
        
        print("extracting frames...")
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext = 'png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1]=="png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient 
        coord_placeholder = (0.0,0.0,0.0,0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i,frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png",frame)
            
            face_box = self.coord_list_cycle[i]
            mask,crop_box = get_image_prepare_material(frame,face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png",mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)
        
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
            
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path)) 
        
    def process_frames(self, 
                       res_frame_queue,
                       video_len,
                       skip_save_images):
        print(video_len)
        while True:
            if self.idx>=video_len-1:
                break
            try:
                # start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            bbox = self.coord_list_cycle[self.idx%(len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx%(len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx%(len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx%(len(self.mask_coords_list_cycle))]
            #combine_frame = get_image(ori_frame,res_frame,bbox)
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png",combine_frame)
            self.idx = self.idx + 1
    
    def process_frames_v2(self, res_frame):
        # start_time = time.time()
        
        bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
        ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % len(self.frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        
        
        # blending_start_time = time.time()
        mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
        mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
        combine_frame = get_image_blending_v2(ori_frame, res_frame, bbox, mask, mask_crop_box)
        # blending_time = time.time() - blending_start_time

        aspect_ratio = combine_frame.shape[1] / combine_frame.shape[0]
        combine_frame = cv2.resize(combine_frame, (int(400 * aspect_ratio), 400)) # width is 256 * aspect ratio
        self.idx += 1

        # total_time = time.time() - start_time
        # print(f">   Total time: {total_time:.4f}s | Blending time: {blending_time:.4f}s")

        return combine_frame # returns a numpy array

    def inference(self, 
                  audio_data, 
                  fps,
                  request_id):
        start_time = time.time()
        
        # Get RMS energy per frame
        sr = 16000
        samples_per_frame = int(sr / fps)
        
        num_frames = 9
        rms_energy_per_frame = []
        last_valid_rms = 0
        for i in range(num_frames):
            frame_audio = audio_data[i * samples_per_frame: (i + 1) * samples_per_frame] * 32768
            rms = np.sqrt(np.mean(frame_audio ** 2))

            if np.isnan(rms):
                rms = last_valid_rms if i > 0 else 0
            else:
                last_valid_rms = rms
            
            rms_energy_per_frame.append(rms)
        print(f"Time 1: {time.time() - start_time}")
        
        whisper_feature = audio_processor.audio2feat(audio_data)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        
        self.idx = request_id * 9 - 9
        # print(f"Using frames {self.idx} to {self.idx + 9}")
        
        cycled_latents = self.input_latent_list_cycle[self.idx % len(self.frame_list_cycle):] + self.input_latent_list_cycle[:self.idx % len(self.frame_list_cycle)]
        gen = datagen(whisper_chunks,
                      cycled_latents, 
                      self.batch_size)
        print(f"Time 2: {time.time() - start_time}")
        
        frame_count = 0
        for i, (whisper_batch,latent_batch) in enumerate(gen):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            
            pred_latents = unet.model(latent_batch, 
                                      timesteps, 
                                      encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            print(f"Time {i + 3}: {time.time() - start_time}")
            
            for res_frame in recon:
                if frame_count < len(rms_energy_per_frame):
                    rms = rms_energy_per_frame[frame_count]
                else:
                    rms = 0
                
                if rms > 3.5:
                    full_frame = self.process_frames_v2(res_frame)
                else:
                    full_frame = self.frame_list_cycle[self.idx % len(self.frame_list_cycle)]
                    aspect_ratio = full_frame.shape[1] / full_frame.shape[0]
                    full_frame = cv2.resize(full_frame, (int(400 * aspect_ratio), 400))
                    self.idx += 1
                
                _, combined_buffer = cv2.imencode('.webp', full_frame, [cv2.IMWRITE_WEBP_QUALITY, 80])
                binary_image = combined_buffer.tobytes()
                
                print(f"Time {i + 4}: {time.time() - start_time}")
                yield binary_image
                frame_count += 1

def init(video_path="data/video/rishi_formal.mp4", name="rishi_1"):
    batch_size = 4
    bbox_shift = 5
    avatar = Avatar(
        avatar_id=name,
        video_path=video_path, 
        bbox_shift=bbox_shift, 
        batch_size=batch_size,
        preparation=True)
    
    return avatar


def main(audio_data, avatar, request_id):
    '''
    This script is used to simulate online chatting and applies necessary pre-processing 
    such as face detection and face parsing in advance. During online chatting, only UNet
    and the VAE decoder are involved, which makes MuseTalk real-time.
    '''
    
    frame_generator = avatar.inference(
        audio_data=audio_data, 
        fps=25,
        request_id=request_id
    )
    
    frame_count = 0
    for frame in frame_generator:
        frame_count += 1
        yield frame
    
    yield b"DONE"


if __name__ == "__main__":
    avatar = init()
    # main(audio_data, avatar)