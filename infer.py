import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageEnhance
import queue
import threading

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
    model = kwargs.get('model')
    args = kwargs.get('args')
    
    device_type = getattr(args, 'device', 'cuda')
    batch_size = getattr(args, 'batch_size', 1)

    video = cv2.VideoCapture(path_in)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_path = path_out + '.tmp.mp4'
    fourcc_val = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(temp_path, fourcc_val, fps, (width, height))
    
    read_queue = queue.Queue(maxsize=batch_size * 4)
    write_queue = queue.Queue(maxsize=batch_size * 4)
    
    thread_exceptions = []
    stop_event = threading.Event()
    
    # 1. Reader Thread
    def reader_worker():
        try:
            for _ in range(total_frames):
                if stop_event.is_set() or getattr(args, 'stop_processing', False):
                    stop_event.set()
                    break
                ret, frame = video.read()
                if not ret:
                    break
                
                # Preprocess on CPU in reader thread
                if args.mode == 'basic':
                    img_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_proc = increase_sharpness(img_proc)
                    tensor = torch.from_numpy(img_proc).permute(2, 0, 1).float() / 255.
                else:
                    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
                    sobel = cv2.magnitude(sobelx, sobely)
                    sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
                
                    t_img = torch.from_numpy(img_gray).unsqueeze(0).float() / 255.
                    t_sobel = torch.from_numpy(sobel).unsqueeze(0).float() / 255.
                    tensor = torch.cat([t_img, t_sobel], dim=0)
                
                # Put with timeout to allow cancellation check
                while not stop_event.is_set() and not getattr(args, 'stop_processing', False):
                    try:
                        read_queue.put(tensor, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            thread_exceptions.append(e)
            stop_event.set()
        finally:
            try:
                read_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            video.release()

    # 2. Writer Thread
    def writer_worker():
        pbar = tqdm(total=total_frames, desc='Processing Video')
        try:
            while True:
                if stop_event.is_set() or getattr(args, 'stop_processing', False):
                    stop_event.set()
                    break
                try:
                    item = write_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    write_queue.task_done()
                    break
                
                pred_numpy = item.numpy()
                img_out = np.clip((pred_numpy * 255) + 0.5, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
                
                video_out.write(img_bgr)
                pbar.update(1)
                write_queue.task_done()
        except Exception as e:
            thread_exceptions.append(e)
            stop_event.set()
        finally:
            video_out.release()
            pbar.close()

    # Start threads
    reader_thread = threading.Thread(target=reader_worker, daemon=True)
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    
    reader_thread.start()
    writer_thread.start()
    
    # 3. Main/Inference Loop on calling thread
    try:
        active = True
        device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
        autocast_enabled = args.fp16 if device_type == 'cuda' else False
        while active and not stop_event.is_set():
            if getattr(args, 'stop_processing', False):
                stop_event.set()
                break
            # Gather batch
            batch_tensors = []
            for _ in range(batch_size):
                if stop_event.is_set() or getattr(args, 'stop_processing', False):
                    active = False
                    break
                
                # Get with timeout to check cancellation flag
                tensor = None
                while not stop_event.is_set() and not getattr(args, 'stop_processing', False):
                    try:
                        tensor = read_queue.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue
                
                if tensor is None:
                    active = False
                    break
                
                batch_tensors.append(tensor)
                read_queue.task_done()
            
            if not batch_tensors:
                break
                
            # Perform inference on batch
            x_in = torch.stack(batch_tensors, dim=0).to(args.device)
            B, C, H, W = x_in.shape
            pad_h = 8 - (H % 8)
            pad_w = 8 - (W % 8)
            x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')
            
            with torch.no_grad(), autocast(enabled=autocast_enabled, device_type=device_type):
                pred = model(x_in)
            
            pred = pred[:, :, :H, :W]
            
            if args.binarize != -1:
                pred = (pred > args.binarize).float()
                
            pred_cpu = pred.cpu()
            
            for i in range(B):
                while not stop_event.is_set() and not getattr(args, 'stop_processing', False):
                    try:
                        write_queue.put(pred_cpu[i, 0], timeout=0.1)
                        break
                    except queue.Full:
                        continue
                
    except Exception as e:
        thread_exceptions.append(e)
        stop_event.set()
        
    reader_thread.join()
    try:
        write_queue.put(None, timeout=1.0)
    except queue.Full:
        pass
    writer_thread.join()
    
    if thread_exceptions or getattr(args, 'stop_processing', False):
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        if getattr(args, 'stop_processing', False):
            raise RuntimeError("Processing cancelled by user.")
        raise thread_exceptions[0]
        
    # Convert the temp file to the final destination using ffmpeg
    import subprocess
    is_webm = path_out.lower().endswith('.webm')
    
    if is_webm:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_path,
            '-c:v', 'libvpx-vp9',
            '-crf', '30',
            '-b:v', '0',
            path_out
        ]
    else:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            path_out
        ]
        
    ffmpeg_success = False
    try:
        res = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            ffmpeg_success = True
    except FileNotFoundError:
        pass
        
    if ffmpeg_success:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        print("FFmpeg encoding failed. Falling back to raw OpenCV video output.")
        if os.path.exists(path_out):
            os.remove(path_out)
        os.rename(temp_path, path_out)

def inference(img: np.ndarray, model, args):
    if args.mode == 'basic':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = increase_sharpness(img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.
        x_in = img
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
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