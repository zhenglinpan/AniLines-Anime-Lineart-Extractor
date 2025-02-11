import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance

import torch
from torch.amp import autocast
import torch.nn.functional as F
from torchvision.utils import save_image

from network.line_extractor import LineExtractor

def increase_sharpness(img, factor=6.0):
    image = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(image)
    return np.array(enhancer.enhance(factor))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', type=str, default='./input', help='input directory or file')
    parser.add_argument('--dir_out', type=str, default='./output', help='output directory or file')
    parser.add_argument('--mode', type=str, default='basic', help='basic or detail')
    parser.add_argument('--fp16', type=bool, default=True, help='use mixed precision to speed up')
    parser.add_argument('--binarize', type=float, default=-1, help='set to [0, 1] to binarize the output')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    args = parser.parse_args()

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

    if args.dir_in.endswith(('png', 'jpg', 'jpeg')):
        flist = [args.dir_in]
    else:
        flist = os.listdir(args.dir_in)

    os.makedirs(args.dir_out, exist_ok=True)    
    for filename in tqdm(flist, desc='Processing'):
        if args.dir_out.endswith(('png', 'jpg', 'jpeg')):
            path_out = args.dir_out
        else:
            path_out = os.path.join(args.dir_out, os.path.basename(filename))
        
        if args.dir_in.endswith(('png', 'jpg', 'jpeg')):
            img = cv2.imread(filename)
        else:
            img = cv2.imread(os.path.join(args.dir_in, filename))
        
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
        save_image(pred, path_out)           
        
if __name__ == '__main__':
    main()