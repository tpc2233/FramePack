# === Necessary Imports ===
from diffusers_helper.hf_login import login
import os
import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time # For timing checks
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb # Note: gpu might be just 'cuda:0' initially
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# --- Multiprocessing Setup ---
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True) # Crucial for CUDA multiprocessing
    print("Multiprocessing start method set to 'spawn'.")
except RuntimeError:
    print("Multiprocessing start method already set or failed to set.")

# === Argument Parsing (Keep as is) ===
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()
print(args)

# === Global Settings ===
outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)
# Define resolution choices
RESOLUTION_CHOICES = [512, 640, 768, 1024]
DEFAULT_RESOLUTION = 640

# === Model Loading Function (to be called inside workers) ===
# Avoid loading models globally in the main process when using spawn
def load_models(model_names, device):
    """Loads specified models onto the target device."""
    models = {}
    print(f"[{os.getpid()}/GPU{device}] Loading models: {model_names}...")
    base_dtype = torch.float16 # Common dtype
    transformer_dtype = torch.bfloat16

    if 'text_encoder' in model_names:
        models['text_encoder'] = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=base_dtype).to(device).eval()
    if 'text_encoder_2' in model_names:
        models['text_encoder_2'] = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=base_dtype).to(device).eval()
    if 'tokenizer' in model_names:
        models['tokenizer'] = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    if 'tokenizer_2' in model_names:
        models['tokenizer_2'] = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    if 'vae' in model_names:
        models['vae'] = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=base_dtype).to(device).eval()
        # models['vae'].enable_slicing() # Enable if needed for decoder GPUs
        # models['vae'].enable_tiling()
    if 'feature_extractor' in model_names:
        models['feature_extractor'] = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    if 'image_encoder' in model_names:
        models['image_encoder'] = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=base_dtype).to(device).eval()
    if 'transformer' in model_names:
        models['transformer'] = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=transformer_dtype).to(device).eval()
        models['transformer'].high_quality_fp32_output_for_inference = True

    print(f"[{os.getpid()}/GPU{device}] Models loaded.")
    return models

# === Sampler Worker (GPU 0) ===
def sampler_worker(gpu_id, sampler_input_queue, sampler_output_queue, main_update_queue):
    device = f'cuda:{gpu_id}'
    print(f"[Sampler Worker {os.getpid()}] Starting on {device}")
    torch.cuda.set_device(device)

    models = {}
    try:
        # Load models needed for sampling + initial encodes
        models = load_models(['transformer', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'image_encoder', 'feature_extractor', 'vae'], device)
        transformer = models['transformer']
        text_encoder = models['text_encoder']
        text_encoder_2 = models['text_encoder_2']
        tokenizer = models['tokenizer']
        tokenizer_2 = models['tokenizer_2']
        image_encoder = models['image_encoder']
        feature_extractor = models['feature_extractor']
        vae = models['vae'] # Needed only for initial encode here

        while True:
            job_data = sampler_input_queue.get()
            if job_data == 'STOP':
                print(f"[Sampler Worker {os.getpid()}] Received STOP signal.")
                break

            job_id = job_data['job_id']
            params = job_data['params']
            print(f"[Sampler Worker {os.getpid()}] Received job {job_id}")

            # --- Initial Setup ---
            input_image_np = params['input_image_np']
            prompt = params['prompt']
            n_prompt = params['n_prompt']
            seed = params['seed']
            latent_window_size = params['latent_window_size']
            steps = params['steps']
            cfg = params['cfg']
            gs = params['gs']
            rs = params['rs']
            use_teacache = params['use_teacache']
            selected_resolution = params['selected_resolution']
            total_latent_sections = params['total_latent_sections']

            # --- Initial Encodings (on GPU 0) ---
            main_update_queue.put({'job_id': job_id, 'type': 'progress', 'data': (None, 'Initial Encodings...', make_progress_bar_html(0, 'Text/Image Encoding...'))})

            # Text
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Image Preprocessing
            H, W, C = input_image_np.shape
            height, width = find_nearest_bucket(H, W, resolution=selected_resolution)
            input_image_resized_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)
            input_image_pt = torch.from_numpy(input_image_resized_np).float() / 127.5 - 1.0
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None].to(device, dtype=vae.dtype)

            # VAE Encode
            start_latent = vae_encode(input_image_pt, vae)

            # CLIP Vision
            image_encoder_output = hf_clip_vision_encode(input_image_resized_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtypes for Transformer
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
            start_latent_tf = start_latent.to(transformer.dtype) # Keep original VAE dtype too if needed?

            # Send initial data back to main process
            sampler_output_queue.put({
                'job_id': job_id,
                'type': 'initial_data',
                'start_latent': start_latent.cpu(), # Send start latent on CPU
                'height': height,
                'width': width
            })

            # --- Sampling Loop ---
            rnd = torch.Generator(device).manual_seed(seed) # Generator on target GPU
            num_frames = latent_window_size * 4 - 3
            latent_height = height // 8
            latent_width = width // 8

            # History latents managed by the *main* process, sampler just receives what it needs
            latent_paddings = list(reversed(range(total_latent_sections)))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            for section_index, latent_padding in enumerate(latent_paddings):
                print(f"[Sampler Worker {os.getpid()}] Waiting for section {section_index} command...")
                section_command = sampler_input_queue.get()
                if section_command == 'STOP' or section_command.get('job_id') != job_id:
                     print(f"[Sampler Worker {os.getpid()}] Received STOP or wrong job ID during sections.")
                     # Maybe send an error/stop confirmation back?
                     break # Stop processing sections for this job

                if section_command.get('type') == 'start_section':
                    current_section_index = section_command['section_index']
                    history_latents_for_cond = section_command['history_latents_for_cond'].to(device, dtype=transformer.dtype) # Needs B, C, 19, H, W
                    print(f"[Sampler Worker {os.getpid()}] Starting section {current_section_index}")

                    is_last_section = latent_padding == 0
                    latent_padding_size = latent_padding * latent_window_size

                    # Prepare conditioning latents
                    clean_latents_pre = start_latent_tf # From initial encode
                    clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents_for_cond.split([1, 2, 16], dim=2)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                    # Indices calculation (as before)
                    total_indices = sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
                    indices = torch.arange(0, total_indices, device=device).unsqueeze(0)
                    clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                        indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                    # Configure TeaCache
                    if use_teacache:
                        transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                    else:
                        transformer.initialize_teacache(enable_teacache=False)

                    # --- K-Diffusion Sampling Callback ---
                    def callback_sampler(d):
                        try:
                            # Preview generation (use fake VAE decode for speed)
                            # Need VAE on this GPU if doing fake decode here, or send latents to main queue
                            # Sending latents is safer if VAE isn't loaded here.
                            preview_latents = d['denoised'] # B, C, T, H, W

                            # Update progress bar and description
                            current_step = d['i'] + 1
                            percentage = int(100.0 * current_step / steps)
                            hint = f'Sampling Step {current_step}/{steps}'
                            # Description needs total generated frame count from main process - tricky
                            # Send basic progress, main process can add more context.
                            main_update_queue.put({
                                'job_id': job_id,
                                'type': 'progress',
                                'data': (None, f"Section {current_section_index+1}/{len(latent_paddings)}", make_progress_bar_html(percentage, hint)),
                                'preview_latent': preview_latents.cpu() # Send preview latent for main process VAE decode if needed
                            })
                        except Exception as e:
                            print(f"[Sampler Worker {os.getpid()}] Error in callback: {e}")
                        return

                    # --- Actual Sampling Call ---
                    print(f"[Sampler Worker {os.getpid()}] Running sample_hunyuan for section {current_section_index}...")
                    generated_latents = sample_hunyuan(
                        transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames,
                        real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                        num_inference_steps=steps, generator=rnd,
                        prompt_embeds=llama_vec.to(device), prompt_embeds_mask=llama_attention_mask.to(device), prompt_poolers=clip_l_pooler.to(device),
                        negative_prompt_embeds=llama_vec_n.to(device), negative_prompt_embeds_mask=llama_attention_mask_n.to(device), negative_prompt_poolers=clip_l_pooler_n.to(device),
                        image_embeddings=image_encoder_last_hidden_state.to(device),
                        latent_indices=latent_indices, clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                        device=device, dtype=transformer.dtype, callback=callback_sampler, callback_steps=5 # Callback less often
                    )
                    print(f"[Sampler Worker {os.getpid()}] Finished sample_hunyuan for section {current_section_index}.")

                    # Send result back to main process
                    sampler_output_queue.put({
                        'job_id': job_id,
                        'type': 'section_result',
                        'section_index': current_section_index,
                        'generated_latents': generated_latents.cpu(), # Send latents on CPU
                        'is_last_section': is_last_section
                    })

            print(f"[Sampler Worker {os.getpid()}] Finished all sections for job {job_id}")
            # Signal job completion? Optional. Main process knows when it receives the last section.

    except Exception as e:
        print(f"[Sampler Worker {os.getpid()}] Error: {e}")
        traceback.print_exc()
        # Send error to main process if possible
        main_update_queue.put({'job_id': job_id, 'type': 'error', 'data': str(e)})
    finally:
        print(f"[Sampler Worker {os.getpid()}] Exiting.")
        # Clean up models? Or let the process exit handle it.
        del models
        torch.cuda.empty_cache()


# === Decoder Worker (GPUs 1, 2, ...) ===
def decoder_worker(gpu_id, decoder_input_queue, decoder_output_queue, main_update_queue):
    device = f'cuda:{gpu_id}'
    print(f"[Decoder Worker {os.getpid()}] Starting on {device}")
    torch.cuda.set_device(device)
    models = {}
    try:
        models = load_models(['vae'], device)
        vae = models['vae']
        vae_dtype = vae.dtype

        while True:
            decode_task = decoder_input_queue.get()
            if decode_task == 'STOP':
                print(f"[Decoder Worker {os.getpid()} on {device}] Received STOP signal.")
                break

            job_id = decode_task['job_id']
            section_index = decode_task['section_index']
            latent_chunk = decode_task['latent_chunk'] # Should be on CPU
            print(f"[Decoder Worker {os.getpid()} on {device}] Received decode task for job {job_id} section {section_index}")

            # Decode
            with torch.no_grad():
                 start_decode_time = time.time()
                 pixel_chunk = vae_decode(latent_chunk.to(device, dtype=vae_dtype), vae)
                 end_decode_time = time.time()
                 print(f"[Decoder Worker {os.getpid()} on {device}] Decoded section {section_index} in {end_decode_time - start_decode_time:.2f}s")


            # Send result back to main process on CPU
            decoder_output_queue.put({
                'job_id': job_id,
                'type': 'decode_result',
                'section_index': section_index,
                'pixel_chunk': pixel_chunk.cpu()
            })
            del pixel_chunk # Free GPU memory
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[Decoder Worker {os.getpid()} on {device}] Error: {e}")
        traceback.print_exc()
        # Send error to main process if possible
        main_update_queue.put({'job_id': job_id, 'type': 'error', 'data': str(e)})
    finally:
        print(f"[Decoder Worker {os.getpid()} on {device}] Exiting.")
        del models
        torch.cuda.empty_cache()


# === Gradio UI and Main Orchestration Logic ===

# Global variables to hold worker processes and queues (managed by the main process)
sampler_proc = None
decoder_procs = []
sampler_input_queue = None
sampler_output_queue = None
decoder_input_queue = None
decoder_output_queue = None
main_update_queue = None

# Store VAE locally for fake decoding previews if needed
local_vae_for_preview = None

def start_workers():
    """Initializes and starts the worker processes and queues."""
    global sampler_proc, decoder_procs, sampler_input_queue, sampler_output_queue, \
           decoder_input_queue, decoder_output_queue, main_update_queue, local_vae_for_preview

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError("Multi-GPU pipeline requires at least 2 GPUs.")

    print(f"Starting multi-GPU pipeline with {num_gpus} GPUs.")
    print("GPU 0: Sampler")
    for i in range(1, num_gpus):
        print(f"GPU {i}: Decoder")

    # Create Queues
    sampler_input_queue = mp.Queue()
    sampler_output_queue = mp.Queue()
    decoder_input_queue = mp.Queue()
    decoder_output_queue = mp.Queue()
    main_update_queue = mp.Queue()

    # Start Sampler Worker (GPU 0)
    sampler_proc = mp.Process(target=sampler_worker, args=(0, sampler_input_queue, sampler_output_queue, main_update_queue), daemon=True)
    sampler_proc.start()

    # Start Decoder Workers (GPUs 1 to N-1)
    decoder_procs = []
    for i in range(1, num_gpus):
        p = mp.Process(target=decoder_worker, args=(i, decoder_input_queue, decoder_output_queue, main_update_queue), daemon=True)
        decoder_procs.append(p)
        p.start()

    # Load VAE in main process *only* if preview decode needed here
    try:
        # Using fake decode instead to avoid loading VAE here
        # models = load_models(['vae'], 'cpu') # Load on CPU
        # local_vae_for_preview = models['vae']
        print("Using fake VAE decode for previews in main process.")
    except Exception as e:
        print(f"Warning: Could not load VAE for preview in main process: {e}")


def stop_workers():
    """Signals workers to stop and joins the processes."""
    global sampler_proc, decoder_procs, sampler_input_queue, sampler_output_queue, \
           decoder_input_queue, decoder_output_queue, main_update_queue
    print("Stopping worker processes...")
    try:
        if sampler_input_queue: sampler_input_queue.put('STOP')
        if decoder_input_queue:
            for _ in decoder_procs: decoder_input_queue.put('STOP') # Send stop signal for each worker

        if sampler_proc and sampler_proc.is_alive(): sampler_proc.join(timeout=10)
        for p in decoder_procs:
            if p.is_alive(): p.join(timeout=10)

        # Close queues
        if sampler_input_queue: sampler_input_queue.close()
        if sampler_output_queue: sampler_output_queue.close()
        if decoder_input_queue: decoder_input_queue.close()
        if decoder_output_queue: decoder_output_queue.close()
        if main_update_queue: main_update_queue.close()

    except Exception as e:
        print(f"Error during worker shutdown: {e}")
    finally:
        sampler_proc = None
        decoder_procs = []
        sampler_input_queue = sampler_output_queue = decoder_input_queue = decoder_output_queue = main_update_queue = None
        print("Workers stopped.")


# New Gradio callback using multi-GPU pipeline
def process_multi_gpu(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation_ignored, use_teacache, mp4_crf, selected_resolution):
    """Gradio button click handler for multi-GPU pipeline."""
    if input_image is None:
        gr.Warning("Please provide an input image.")
        yield None, gr.update(visible=False), "", "", gr.update(interactive=True), gr.update(interactive=False)
        return

    # --- Start Workers if not already running ---
    # This basic implementation restarts workers for each job.
    # A more advanced version could keep them running.
    stop_workers() # Ensure clean state
    start_workers()

    job_id = generate_timestamp()
    print(f"\n--- Starting Job {job_id} ---")

    # --- Initial UI Update ---
    yield None, gr.update(visible=False), "Initializing...", make_progress_bar_html(0, 'Initializing...'), gr.update(interactive=False), gr.update(interactive=True)

    # --- Prepare Job Data ---
    total_latent_sections = int(max(round((total_second_length * 30) / (latent_window_size * 4)), 1))
    params = {
        'input_image_np': input_image, 'prompt': prompt, 'n_prompt': n_prompt, 'seed': int(seed),
        'total_second_length': total_second_length, 'latent_window_size': latent_window_size,
        'steps': steps, 'cfg': cfg, 'gs': gs, 'rs': rs, 'use_teacache': use_teacache,
        'mp4_crf': mp4_crf, 'selected_resolution': selected_resolution,
        'total_latent_sections': total_latent_sections
    }
    sampler_input_queue.put({'job_id': job_id, 'params': params})

    # --- Main Orchestration Loop ---
    start_latent = None
    height, width = 0, 0
    history_latents_cpu = None # Store history on CPU
    history_pixels = None
    output_filename = None
    last_saved_section = -1
    decoded_pixel_chunks = {} # Store received pixel chunks: {section_index: pixel_chunk}
    processed_decode_sections = set() # Track sections whose pixels have been stitched

    total_generated_latent_frames = 0 # Tracked here in main process

    # --- Wait for Initial Data from Sampler ---
    initial_data_received = False
    while not initial_data_received:
        try:
            update = main_update_queue.get(timeout=1) # Check for progress updates too
            if update.get('job_id') == job_id and update.get('type') == 'progress':
                 preview, desc, html = update['data']
                 yield output_filename, gr.update(visible=bool(preview), value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

            output = sampler_output_queue.get(timeout=1) # Check for initial data
            if output.get('job_id') == job_id and output.get('type') == 'initial_data':
                start_latent = output['start_latent'] # Already on CPU
                height = output['height']
                width = output['width']
                initial_data_received = True
                print(f"[Main {os.getpid()}] Received initial data for job {job_id}. Target size: {width}x{height}")
                # Initialize history latent template (CPU)
                latent_height, latent_width = height // 8, width // 8
                history_latents_cpu = torch.zeros((1, 16, 1 + 2 + 16, latent_height, latent_width), dtype=start_latent.dtype) # Match VAE dtype initially
            elif output.get('job_id') != job_id:
                 print(f"[Main {os.getpid()}] Warning: Received output for wrong job ID ({output.get('job_id')})")

        except mp.queues.Empty:
            print(f"[Main {os.getpid()}] Waiting for initial data...")
            time.sleep(0.5)
        except Exception as e:
            print(f"[Main {os.getpid()}] Error waiting for initial data: {e}")
            stop_workers()
            yield output_filename, gr.update(visible=False), f"Error: {e}", "", gr.update(interactive=True), gr.update(interactive=False)
            return


    # --- Section Processing Loop ---
    active_decode_section = -1 # Track which section's pixels we expect next
    sampler_finished_section = -1 # Track last section sampler finished

    while sampler_finished_section < total_latent_sections - 1 or active_decode_section < total_latent_sections - 1:
        try:
            # --- 1. Check for Updates from Workers (Non-blocking) ---
            while not main_update_queue.empty():
                 update = main_update_queue.get()
                 if update.get('job_id') != job_id: continue

                 if update.get('type') == 'progress':
                     preview_latent = update.get('preview_latent')
                     preview_np = None
                     if preview_latent is not None:
                         # Use fake decode for preview
                         try:
                             preview_pixels = vae_decode_fake(preview_latent.to(start_latent.dtype)) # Use fake decode
                             preview_pixels = (preview_pixels * 127.5 + 127.5).clamp(0, 255)
                             preview_np = preview_pixels.detach().cpu().numpy().astype(np.uint8)
                             preview_np = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')
                         except Exception as e:
                             print(f"[Main {os.getpid()}] Error generating preview: {e}")

                     _p, desc, html = update['data']
                     # Add more context to description
                     total_pixel_frames_so_far = max(0, total_generated_latent_frames * 4 - 3)
                     current_video_len_sec = total_pixel_frames_so_far / 30.0
                     full_desc = f'{desc} | Total Frames: {total_pixel_frames_so_far}, Vid Len: {current_video_len_sec:.2f}s'
                     yield output_filename, gr.update(visible=bool(preview_np), value=preview_np), full_desc, html, gr.update(interactive=False), gr.update(interactive=True)

                 elif update.get('type') == 'error':
                     error_msg = update.get('data', 'Unknown worker error')
                     gr.Error(f"Worker Error: {error_msg}")
                     stop_workers()
                     yield output_filename, gr.update(visible=False), f"Error: {error_msg}", "", gr.update(interactive=True), gr.update(interactive=False)
                     return

            # --- 2. Check for Finished Latents from Sampler ---
            while not sampler_output_queue.empty():
                 output = sampler_output_queue.get()
                 if output.get('job_id') != job_id: continue

                 if output.get('type') == 'section_result':
                     section_index = output['section_index']
                     generated_latents = output['generated_latents'] # On CPU
                     is_last_section = output['is_last_section']
                     sampler_finished_section = section_index
                     print(f"[Main {os.getpid()}] Received latents for section {section_index}")

                     # Append start frame if last section
                     if is_last_section:
                          generated_latents = torch.cat([start_latent, generated_latents], dim=2)

                     # Send to decoder queue
                     decoder_input_queue.put({
                         'job_id': job_id,
                         'section_index': section_index,
                         'latent_chunk': generated_latents # Send whole chunk
                     })
                     active_decode_section = max(active_decode_section, section_index) # Expect decode result for this section

                     # Update history latents *after* sending chunk for decode
                     num_new_latents = generated_latents.shape[2]
                     total_generated_latent_frames += num_new_latents

                     # Prepend new latents to history (CPU)
                     if section_index == 0:
                         history_latents_cpu = generated_latents
                     else:
                         # Ensure history_latents_cpu exists and has correct dtype
                         if history_latents_cpu is None or history_latents_cpu.shape[1]!=generated_latents.shape[1]:
                             print("Warning: history_latents_cpu incorrect shape/type, reinitializing")
                              history_latents_cpu = generated_latents # Fallback
                         else:
                             history_latents_cpu = torch.cat([generated_latents.to(history_latents_cpu.dtype), history_latents_cpu], dim=2)


            # --- 3. Check for Finished Decoded Pixels ---
            while not decoder_output_queue.empty():
                 output = decoder_output_queue.get()
                 if output.get('job_id') != job_id: continue

                 if output.get('type') == 'decode_result':
                     section_index = output['section_index']
                     pixel_chunk = output['pixel_chunk'] # On CPU
                     print(f"[Main {os.getpid()}] Received decoded pixels for section {section_index}")
                     decoded_pixel_chunks[section_index] = pixel_chunk

            # --- 4. Process Decoded Pixels (Stitching) in Order ---
            # Check if the *next expected* decoded section is ready
            next_expected_section = min(decoded_pixel_chunks.keys()) if decoded_pixel_chunks else -1
            can_process_section = -1
            if decoded_pixel_chunks:
                 if history_pixels is None and 0 in decoded_pixel_chunks: # First section ready
                     can_process_section = 0
                 elif history_pixels is not None and last_saved_section + 1 in decoded_pixel_chunks: # Next consecutive section ready
                     can_process_section = last_saved_section + 1

            if can_process_section != -1:
                section_index = can_process_section
                pixel_chunk = decoded_pixel_chunks.pop(section_index) # Process and remove
                print(f"[Main {os.getpid()}] Processing/Stitching pixels for section {section_index}")

                if history_pixels is None:
                    history_pixels = pixel_chunk
                else:
                    overlapped_frames = latent_window_size * 4 - 3
                    history_pixels = soft_append_bcthw(pixel_chunk, history_pixels, overlap_frames=overlapped_frames)

                # Save video
                output_filename = os.path.join(outputs_folder, f'{job_id}_section_{section_index}.mp4')
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
                last_saved_section = section_index
                processed_decode_sections.add(section_index)
                print(f"[Main {os.getpid()}] Saved video up to section {section_index}. Total pixel shape: {history_pixels.shape}")

                # Yield file update
                yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)


            # --- 5. Send Command for Next Section to Sampler ---
            # If sampler is idle (finished section N) and we haven't started N+1
            next_section_to_sample = sampler_finished_section + 1
            if next_section_to_sample < total_latent_sections:
                # Prepare conditioning history for sampler
                # Needs the latest 1+2+16 frames
                hist_frames_needed = 1 + 2 + 16
                hist_available = history_latents_cpu.shape[2] if history_latents_cpu is not None else 0
                hist_start_idx = max(0, hist_available - hist_frames_needed)

                if hist_available >= hist_frames_needed:
                    cond_latents = history_latents_cpu[:, :, hist_start_idx:, :, :]
                else: # Pad if not enough history (should only happen early on)
                    cond_latents_partial = history_latents_cpu[:, :, hist_start_idx:, :, :] if history_latents_cpu is not None else torch.zeros((1,16,0,latent_height, latent_width), dtype=start_latent.dtype)
                    padding_needed = hist_frames_needed - cond_latents_partial.shape[2]
                    padding = torch.zeros((1, 16, padding_needed, latent_height, latent_width), dtype=start_latent.dtype)
                    cond_latents = torch.cat([padding, cond_latents_partial], dim=2)

                # Send command to sampler
                print(f"[Main {os.getpid()}] Sending command to start section {next_section_to_sample}")
                sampler_input_queue.put({
                    'job_id': job_id,
                    'type': 'start_section',
                    'section_index': next_section_to_sample,
                    'history_latents_for_cond': cond_latents # Send on CPU, worker moves to GPU
                })
                # Prevent sending command twice for the same section
                sampler_finished_section = next_section_to_sample # Assume sampler starts it

            # Small sleep if queues were empty to prevent busy-waiting
            time.sleep(0.05)

        except Exception as e:
            print(f"[Main {os.getpid()}] Error in orchestration loop: {e}")
            traceback.print_exc()
            gr.Error(f"Orchestration Error: {e}")
            stop_workers()
            yield output_filename, gr.update(visible=False), f"Error: {e}", "", gr.update(interactive=True), gr.update(interactive=False)
            return

    # --- End of Loop ---
    print(f"[Main {os.getpid()}] Finished processing all sections for job {job_id}.")
    stop_workers()
    yield output_filename, gr.update(visible=False), "Finished.", "", gr.update(interactive=True), gr.update(interactive=False)


# === Gradio Interface Definition (Mostly Unchanged) ===
css = make_progress_bar_css()
block = gr.Blocks(css=css, theme=gr.themes.Soft()).queue()

with block:
    # ... (Gradio layout definition remains exactly the same as in your previous complete code) ...
    gr.Markdown("# FramePack Image-to-Video (Multi-GPU Pipeline)")
    # ... Input Column ...
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(sources='upload', type="numpy", label="Input Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='A cinematic shot of a woman walking on the street.', lines=2)
            # ... (Quick prompts dataset) ...
            quick_prompts = [
                ['The girl dances gracefully, with clear movements, full of charm.'],
                ['A character doing some simple body movements.'],
                ['A close-up shot of an eye blinking.'],
                ['Ocean waves crashing on the shore.'],
                ['A steaming cup of coffee on a table.'],
                ['Leaves falling from a tree in autumn.'],
                ['A cat slowly stretching.'],
            ]
            example_quick_prompts = gr.Dataset(
                samples=quick_prompts, label='Quick Prompt List', samples_per_page=10, components=[prompt], type="index"
            )
            example_quick_prompts.click(fn=None, inputs=[example_quick_prompts], outputs=[prompt], _js=f"(i) => {quick_prompts}[i][0]")

            with gr.Row():
                start_button = gr.Button("Start Generation", variant="primary")
                # Stop button functionality is complex with multiprocessing, removed for now
                # end_button = gr.Button("Stop Generation", interactive=False)

            with gr.Accordion("Generation Settings", open=True): # Open by default
                with gr.Group():
                    selected_resolution = gr.Dropdown(
                        label="Target Resolution (Approx.)", choices=RESOLUTION_CHOICES, value=DEFAULT_RESOLUTION,
                        info="Higher resolutions increase VAE decode time, potentially benefiting more from multi-GPU."
                    )
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1.0, maximum=20.0, value=4.0, step=0.1)
                    seed = gr.Number(label="Seed", value=1234, precision=0, interactive=True)
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Potentially faster speed, but may slightly affect fine details like fingers.')
                    mp4_crf = gr.Slider(label="MP4 CRF (Quality)", minimum=0, maximum=51, value=18, step=1, info="Lower value = better quality. 18-28 typical.")

            with gr.Accordion("Advanced Settings", open=False):
                 with gr.Group():
                    steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=25, step=1, info='Default: 25.')
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.1, info='Default: 10.0.')
                    # gpu_memory_preservation = gr.Slider(...) # Removed as High VRAM assumed
                    # Hidden/Unused parameters
                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)

        # ... Output Column ...
        with gr.Column(scale=1):
            gr.Markdown("### Output")
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, interactive=False)
            preview_image = gr.Image(label="Live Preview (Latent Approximation)", height=256, visible=False, interactive=False)
            gr.Markdown("ℹ️ *Multi-GPU pipeline active. VAE decoding runs in parallel with next section sampling.*")
            progress_desc = gr.Markdown('', elem_classes='generating-info')
            progress_bar = gr.HTML('', elem_classes='generating-progress')

    gr.Markdown("---")
    gr.HTML('<div style="text-align:center; margin-top:10px;">Model Credits: HunyuanVideo Community | FramePack Weights by lllyasviel | UI based on FramePack Gradio</div>')


    # --- Event Listeners ---
    # Define inputs list matching the 'process_multi_gpu' function signature
    ips = [
        input_image, prompt, n_prompt, seed, total_second_length, latent_window_size,
        steps, cfg, gs, rs, # gpu_memory_preservation is ignored but keep placeholder if needed
        steps, # Placeholder for gpu_memory_preservation index
        use_teacache, mp4_crf, selected_resolution
    ]
    # Define outputs list matching the 'yield' statements in 'process_multi_gpu'
    # Need dummy output for the start button state if end button removed
    dummy_interactive_state = gr.update(interactive=True)
    ops = [
        result_video, preview_image, progress_desc, progress_bar, start_button, dummy_interactive_state # start_button, end_button
    ]

    # Use the new multi-GPU processing function
    start_button.click(fn=process_multi_gpu, inputs=ips, outputs=ops)
    # End button functionality needs careful implementation with multiprocessing (sending signals to main process to stop workers) - Omitted for now.
    # end_button.click(fn=end_process_multi_gpu) # Needs implementation

# === Launch App ===
if __name__ == "__main__": # Protect multiprocessing code
    # Clean up any previous workers on script exit
    import atexit
    atexit.register(stop_workers)

    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        prevent_thread_lock=True # May not be strictly needed with multiprocessing, but doesn't hurt
    )
