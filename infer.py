import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

import torch
from torch.amp import autocast
import torch.nn.functional as F

from network.line_extractor import LineExtractor

def is_file(path):
    return os.path.splitext(os.path.basename(path))[1]  # TODO: a better way to check if path is file

def is_image(path):
    fname = os.path.basename(path)
    return os.path.splitext(fname)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif', '.tga']

def is_video(path):
    fname = os.path.basename(path)
    return os.path.splitext(fname)[1].lower() in ['.mp4', '.avi', '.mkv', '.mov']

def increase_sharpness(img, factor=6.0):
    image = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(image)
    return np.array(enhancer.enhance(factor))

def load_model(args):
    if args.mode == 'basic':
        model = LineExtractor(3, 1, True).to(args.device)
    elif args.mode == 'detail':
        model = LineExtractor(2, 1, True).to(args.device)
    else:
        raise ValueError('Mode must be either basic or detail')
    
    path_model = os.path.join('weights', f'{args.mode}.pth')
    model.load_state_dict(torch.load(path_model, map_location=torch.device(args.device), weights_only=True))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    return model

def process_image(path_in, path_out, **kwargs):
    img = cv2.cvtColor(np.array(Image.open(path_in)), cv2.COLOR_RGB2BGR)
    img = inference(img, **kwargs)
    img = Image.fromarray(img)
    img.save(path_out)
    return img
    
def process_video(path_in, path_out, fourcc='mp4v', **kwargs):
    video = cv2.VideoCapture(path_in)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    video_out = cv2.VideoWriter(path_out, fourcc, fps, (width, height))
    
    for _ in tqdm(range(total_frames), desc='Processing Video'):
        ret, frame = video.read()
        if not ret:
            break
        
        img = inference(frame, **kwargs)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        video_out.write(img)
        
    video.release()
    video_out.release()

def inference(img: np.ndarray, model, args):
    if args.mode == 'basic':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = increase_sharpness(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.
        x_in = img
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(args.device) / 255.
        sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(args.device) / 255.
    
        x_in = torch.cat([img, sobel], dim=1)

    B, C, H, W = x_in.shape
    pad_h = 8 - (H % 8)
    pad_w = 8 - (W % 8)
    x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad(), autocast(enabled=args.fp16, device_type='cuda'):
        pred = model(x_in)
    pred = pred[:, :, :H, :W]
    if args.binarize != -1:
        pred = (pred > args.binarize).float()
    
    return np.clip((pred[0, 0].cpu().numpy() * 255) + 0.5, 0, 255).astype(np.uint8)

def do_inference(path_in, path_out, model, args):
    fname = os.path.basename(path_in)
    if is_image(fname):
        process_image(path_in, path_out, model=model, args=args)
    elif is_video(fname):
        process_video(path_in, path_out, fourcc='mp4v', model=model, args=args)
    else:
        raise ValueError(f'Unsupported file: {path_in}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', type=str, default='./input', help='input directory or file')
    parser.add_argument('--dir_out', type=str, default='./output', help='output directory or file')
    parser.add_argument('--mode', type=str, default='detail', help='basic or detail')
    parser.add_argument('--fp16', type=bool, default=True, help='use mixed precision to speed up')
    parser.add_argument('--binarize', type=float, default=-1, help='set to [0, 1] to binarize the output')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    args = parser.parse_args()

    model = load_model(args)

    flist = [args.dir_in] if os.path.isfile(args.dir_in) else os.listdir(args.dir_in)
    
    for filename in tqdm(flist, desc='Processing'):       
        path_in = filename if is_file(args.dir_in) else os.path.join(args.dir_in, filename)
        path_out = args.dir_out if is_file(args.dir_out) else os.path.join(args.dir_out, os.path.basename(filename))
        if not is_file(args.dir_out):
            os.makedirs(args.dir_out, exist_ok=True)
        
        do_inference(path_in, path_out, model, args)
        
if __name__ == '__main__':
    main()