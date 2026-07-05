import os
import cv2
import time
import argparse
import gradio as gr
import numpy as np
from threading import Thread
from infer import inference, load_model, do_inference
from infer import process_video as _process_video

# Flag to stop processing
stop_processing = False
active_video_args = None

def process_image(image, mode, binarize, threshold, fp16, cpu_checkbox):
    if image is None:
        return None
    binarize_value = threshold if binarize else -1
    device = "cpu" if cpu_checkbox else "cuda"
    args = argparse.Namespace(mode=mode, binarize=binarize_value, fp16=fp16, device=device)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    model = load_model(args)
    
    return inference(image, model, args)

def process_video(video_path, mode, binarize, threshold, fp16, cpu_checkbox, batch_size):
    global active_video_args, stop_processing
    stop_processing = False
    if video_path is None:
        return None
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found.")
    
    binarize_value = threshold if binarize else -1
    device = "cpu" if cpu_checkbox else "cuda"
    actual_batch_size = 1 if cpu_checkbox else int(batch_size)
    args = argparse.Namespace(mode=mode, binarize=binarize_value, fp16=fp16, device=device, batch_size=actual_batch_size, stop_processing=False)
    active_video_args = args
    output_path = 'output.mp4'  # HACK: temp output
    model = load_model(args)
    try:
        _process_video(video_path, output_path, fourcc='mp4v', model=model, args=args)
    except RuntimeError as e:
        if "cancelled" in str(e):
            print("Video processing was cancelled.")
            return None
        raise e
    finally:
        active_video_args = None
        if 'model' in locals():
            del model
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return os.path.abspath(output_path)

def process_folder(input_folder, output_folder, mode, binarize, threshold, fp16, cpu_checkbox, batch_size):
    global stop_processing
    binarize_value = threshold if binarize else -1
    device = "cpu" if cpu_checkbox else "cuda"
    actual_batch_size = 1 if cpu_checkbox else int(batch_size)
    args = argparse.Namespace(mode=mode, binarize=binarize_value, fp16=fp16, device=device, batch_size=actual_batch_size)
    model = load_model(args)

    if not os.path.exists(input_folder):
        yield f"Input folder {input_folder} does not exist!"
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)
    total_files = len(files)

    if total_files == 0:
        yield f"No files found in the input folder {input_folder}."
        return

    start_time = time.time()

    try:
        for i, file in enumerate(files):
            if stop_processing:
                yield "Processing stopped."
                break

            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            file_start = time.time() 
            do_inference(input_path, output_path, model, args)
            file_end = time.time()

            elapsed_time = file_end - start_time
            avg_time_per_file = elapsed_time / (i + 1)
            remaining_files = total_files - (i + 1)
            eta = remaining_files * avg_time_per_file

            eta_minutes = int(eta // 60)
            eta_seconds = int(eta % 60)

            yield f"{i + 1} / {total_files}\tETA: {eta_minutes}m {eta_seconds}s"
    finally:
        if 'model' in locals():
            del model
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    yield "All Done."

def stop_processing_function():
    global stop_processing, active_video_args
    stop_processing = True
    if active_video_args is not None:
        active_video_args.stop_processing = True

with gr.Blocks() as demo:
    gr.Markdown("# AniLines - Anime Line Extractior")

    with gr.Tabs():
        with gr.Tab("Image Processing"):
            gr.Markdown("## Process Images")
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image")
                image_output = gr.Image(label="Processed Output")

            mode_dropdown = gr.Radio(["basic", "detail"], value="detail", label="Processing Mode")
            with gr.Row():
                fp16_checkbox = gr.Checkbox(label="Use FP16", value=True)
                cpu_checkbox = gr.Checkbox(label="Use CPU", value=False)
            binarize_checkbox = gr.Checkbox(label="Binarize", value=False)
            binarize_slider = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.75, label="Binarization Threshold (-1 for auto)", visible=False)
            binarize_checkbox.change(lambda binarize: gr.update(visible=binarize), inputs=binarize_checkbox, outputs=binarize_slider)

            process_button = gr.Button("Process")
            process_button.click(process_image, 
                                 inputs=[image_input, mode_dropdown, binarize_checkbox, binarize_slider, fp16_checkbox, cpu_checkbox], 
                                 outputs=image_output)

        with gr.Tab("Video Processing"):
            gr.Markdown("## Process Videos")
            with gr.Row():
                video_input = gr.Video(label="Upload Video")
                video_output = gr.Video(label="Processed Output")

            mode_dropdown = gr.Radio(["basic", "detail"], value="detail", label="Processing Mode")
            with gr.Row():
                fp16_checkbox = gr.Checkbox(label="Use FP16", value=True)
                cpu_checkbox = gr.Checkbox(label="Use CPU", value=False)
                batch_size_number_video = gr.Number(value=1, minimum=1, maximum=16, precision=0, label="Batch Size (GPU only)")
            binarize_checkbox = gr.Checkbox(label="Binarize", value=False)
            binarize_slider = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.75, label="Binarization Threshold (-1 for auto)", visible=False)
            binarize_checkbox.change(lambda binarize: gr.update(visible=binarize), inputs=binarize_checkbox, outputs=binarize_slider)

            with gr.Row():
                process_button = gr.Button("Process")
                cancel_button = gr.Button("Cancel the process", variant="stop")
            process_button.click(process_video, 
                                 inputs=[video_input, mode_dropdown, binarize_checkbox, binarize_slider, fp16_checkbox, cpu_checkbox, batch_size_number_video], 
                                 outputs=video_output)
            cancel_button.click(stop_processing_function, inputs=None, outputs=None)

        with gr.Tab("Folder Processing"):
            gr.Markdown("## Process Folder")
            input_folder = gr.Textbox(label="Input Folder Path")
            output_folder = gr.Textbox(label="Output Folder Path")
            progress = gr.Markdown("Click 'Process Folder' to start.", visible=True)

            mode_dropdown = gr.Radio(["basic", "detail"], value="detail", label="Processing Mode")
            with gr.Row():
                fp16_checkbox = gr.Checkbox(label="Use FP16", value=True)
                cpu_checkbox = gr.Checkbox(label="Use CPU", value=False)
                batch_size_number_folder = gr.Number(value=1, minimum=1, maximum=16, precision=0, label="Batch Size (GPU only)")
            binarize_checkbox = gr.Checkbox(label="Binarize", value=False)
            binarize_slider = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.75, label="Binarization Threshold (-1 for auto)", visible=False)
            binarize_checkbox.change(lambda binarize: gr.update(visible=binarize), inputs=binarize_checkbox, outputs=binarize_slider)

            process_button_folder = gr.Button("Process Folder")
            stop_button_folder = gr.Button("Kill the process")
            process_button_folder.click(process_folder, 
                                        inputs=[input_folder, output_folder, mode_dropdown, binarize_checkbox, binarize_slider, fp16_checkbox, cpu_checkbox, batch_size_number_folder], 
                                        outputs=progress, 
                                        show_progress=True)
            stop_button_folder.click(stop_processing_function, inputs=None, outputs=None)

if __name__ == "__main__":
    demo.launch()
