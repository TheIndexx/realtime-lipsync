from PIL import Image
import numpy as np
import cv2
from face_parsing import FaceParsing

fp = FaceParsing()

def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s

def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image

def get_image(image,face,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    #print(image.shape)
    #print(face.shape)
    
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box 
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    face_position = (x, y)

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    mask_image = Image.fromarray(mask_array)
    
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]

def get_image_prepare_material(image,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array,crop_box

def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = Image.fromarray(image[:,:,::-1]) # BGR to RGB
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)

    mask_image = Image.fromarray(mask_array)
    mask_image = mask_image.convert("L")
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]

def get_image_blending_v2(image, face, face_box, mask_array, crop_box):
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box

    # Resize face to match the dimensions defined by face_box
    face_width = x1 - x
    face_height = y1 - y
    face = cv2.resize(face, (face_width, face_height))

    # Resize mask_array to match face dimensions
    mask_array_resized = cv2.resize(mask_array, (face_width, face_height))

    # Ensure mask_array_resized is single-channel (grayscale)
    if mask_array_resized.ndim == 3 and mask_array_resized.shape[2] == 3:
        mask_array_resized = cv2.cvtColor(mask_array_resized, cv2.COLOR_BGR2GRAY)

    # Crop the region of interest from the image
    face_large = image[y_s:y_e, x_s:x_e].copy()

    # Calculate the position where the face will be pasted
    x_offset = x - x_s
    y_offset = y - y_s

    # Ensure the face image fits within the cropped region
    h_large, w_large = face_large.shape[:2]
    h_face, w_face = face.shape[:2]
    x_end = min(x_offset + w_face, w_large)
    y_end = min(y_offset + h_face, h_large)

    # Paste the resized face onto the cropped region
    face_large[y_offset:y_end, x_offset:x_end] = face[0:y_end - y_offset, 0:x_end - x_offset]

    # Create a mask of the same size as face_large
    mask = np.zeros((h_large, w_large), dtype=np.float32)

    # Place the resized mask_array onto the mask at the correct position
    mask[y_offset:y_end, x_offset:x_end] = mask_array_resized[0:y_end - y_offset, 0:x_end - x_offset].astype(np.float32) / 255.0

    # Ensure mask has three channels
    if mask.ndim == 2:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Blend the images using the mask
    blended_region = (face_large.astype(np.float32) * mask +
                      image[y_s:y_e, x_s:x_e].astype(np.float32) * (1 - mask)).astype(np.uint8)
    image[y_s:y_e, x_s:x_e] = blended_region

    return image
