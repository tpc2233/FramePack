# !pip install gradio diffusers transformers accelerate bitsandbytes safetensors torch einops omegaconf
# !pip install git+https://github.com/huggingface/diffusers

from diffusers_helper.hf_login import login

import os

# Set HF_HOME before importing transformers/diffusers if needed
# Adjust the path './hf_download' as necessary
hf_home_path = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__) if '__file__' in locals() else '.', './hf_download')))
os.environ['HF_HOME'] = hf_home_path
os.makedirs(hf_home_path, exist_ok=True)
print(f"HF_HOME set to: {hf_home_path}")


import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

# Check for CUDA availability early
if not torch.cuda.is_available():
    print("CUDA not available. Exiting.")
    exit()

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60 # Adjust threshold if needed

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- Model Loading ---
print("Loading models...")
try:
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure you have the necessary model files downloaded and accessible.")
    print("Check network connection or HF_HOME path.")
    exit()
print("Models loaded.")

# --- Model Configuration ---
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Set dtypes
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

# No gradients needed
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# Move to GPU / Setup Offloading
if not high_vram:
    print("Setting up dynamic swapping (offloading) for low VRAM.")
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but potentially faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    # Other models will be loaded/offloaded as needed
else:
    print("Moving models to GPU for high VRAM mode.")
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Define resolution choices
RESOLUTION_CHOICES = [512, 640, 768, 1024] # Start conservative, add higher later if stable
DEFAULT_RESOLUTION = 640


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, selected_resolution): # Added selected_resolution
    """The main generation worker thread."""
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4) # 30 FPS assumed
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # --- Initial Cleanup (Low VRAM Mode) ---
        if not high_vram:
            print("Unloading models before starting...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            torch.cuda.empty_cache() # Aggressively clear cache

        # --- Text encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        if not high_vram:
            print("Loading text encoders...")
            # Use fake device context for potentially faster single load than offload->load
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        else: # Ensure they are on GPU if high_vram mode
             text_encoder.to(gpu)
             text_encoder_2.to(gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1: # If CFG is 1, uncond is zeros (optimization)
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            # Only encode negative if needed
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        # Pad/crop embeddings
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        if not high_vram: # Offload text encoders after use
             offload_model_from_device_for_memory_preservation(text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
             offload_model_from_device_for_memory_preservation(text_encoder_2, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # --- Processing input image ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        # Use the selected resolution here!
        print(f"Input image dimensions: {W}x{H}")
        print(f"Finding nearest bucket for target resolution: {selected_resolution}")
        height, width = find_nearest_bucket(H, W, resolution=selected_resolution)
        print(f"Selected target dimensions for generation: {width}x{height}")

        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Save input for reference
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_input.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1.0 # Normalize to [-1, 1]
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # Add Batch, Channel, Time dims (B, C, T, H, W)

        # --- VAE encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        if not high_vram:
            print("Loading VAE...")
            load_model_as_complete(vae, target_device=gpu)
        else:
            vae.to(gpu)

        start_latent = vae_encode(input_image_pt, vae) # Latent dims depend on height, width
        print(f"Encoded start latent shape: {start_latent.shape}") # B, C, T, H/8, W/8

        if not high_vram: # Offload VAE after use
             offload_model_from_device_for_memory_preservation(vae, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # --- CLIP Vision encoding ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        if not high_vram:
            print("Loading Image Encoder...")
            load_model_as_complete(image_encoder, target_device=gpu)
        else:
            image_encoder.to(gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        if not high_vram: # Offload Image Encoder after use
             offload_model_from_device_for_memory_preservation(image_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # --- Dtype Conversion for Transformer ---
        # Transformer expects bfloat16 for optimal performance/memory
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
        start_latent = start_latent.to(transformer.dtype) # Also convert start latent

        # --- Sampling Loop ---
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed) # Use CPU generator for reproducibility across devices
        num_frames = latent_window_size * 4 - 3 # Number of frames in a single sampling window (FramePack specific)

        # Initialize history_latents with correct dimensions based on chosen H/W
        # VAE typically downsamples by 8x. Transformer operates on latent space.
        latent_height = height // 8
        latent_width = width // 8
        # History needs space for clean latents used by FramePack (1+2+16 = 19 frames)
        # Shape: B, C=16 (latent channels), T (max history), H/8, W/8
        history_latents_template = torch.zeros(size=(1, 16, 1 + 2 + 16, latent_height, latent_width), dtype=start_latent.dtype)
        print(f"Initializing history latents template shape: {history_latents_template.shape}")

        history_pixels = None # Stores the decoded pixel frames
        total_generated_latent_frames = 0 # Track total frames generated so far

        # FramePack specific padding logic for smoother generation
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
            # Trick for longer sequences (might need adjustment based on results)
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for i, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            # Check for user interruption
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                print("Generation interrupted by user.")
                return

            print(f'\n--- Section {i+1}/{len(latent_paddings)} ---')
            print(f'Latent Padding Size: {latent_padding_size}, Is Last Section: {is_last_section}')

            # --- FramePack Specific Indexing ---
            # These indices define which parts of the input tensor to the transformer are
            # noise, clean latents (conditioning), etc.
            total_indices = sum([1, latent_padding_size, latent_window_size, 1, 2, 16]) # Total time dimension for transformer input
            indices = torch.arange(0, total_indices).unsqueeze(0) # B=1, T
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = \
                indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            # --- Prepare Conditioning Latents for FramePack ---
            # Needs the start latent and potentially previous generated latents if available
            clean_latents_pre = start_latent.to(history_latents_template.device, dtype=history_latents_template.dtype) # Ensure start latent matches history dtype/device

            # Get conditioning frames from history (if available) or zeros
            # History shape: B, C, T_hist, H, W
            # Needed: B, C, T_cond, H, W where T_cond = 1 + 2 + 16
            if total_generated_latent_frames > 0:
                 # Take the most recent 1+2+16 frames from history
                 hist_frames_needed = 1 + 2 + 16
                 hist_available = history_latents.shape[2]
                 hist_start_idx = max(0, hist_available - hist_frames_needed)
                 cond_latents_from_hist = history_latents[:, :, hist_start_idx:, :, :]
                 # Pad if not enough history frames yet (shouldn't happen often after first iter)
                 if cond_latents_from_hist.shape[2] < hist_frames_needed:
                     padding_needed = hist_frames_needed - cond_latents_from_hist.shape[2]
                     padding = torch.zeros(
                         (1, 16, padding_needed, latent_height, latent_width),
                         dtype=cond_latents_from_hist.dtype,
                         device=cond_latents_from_hist.device
                     )
                     cond_latents_from_hist = torch.cat([padding, cond_latents_from_hist], dim=2)
            else:
                 # Use zeros if no history yet
                 cond_latents_from_hist = torch.zeros_like(history_latents_template) # Use template for shape/dtype/device

            cond_latents_from_hist = cond_latents_from_hist.to(history_latents_template.device, dtype=history_latents_template.dtype)
            clean_latents_post, clean_latents_2x, clean_latents_4x = cond_latents_from_hist.split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2) # Combine start frame + history frame

            # --- Load Transformer (if needed) ---
            if not high_vram:
                print("Loading Transformer...")
                unload_complete_models() # Unload anything else first
                torch.cuda.empty_cache()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
            else:
                transformer.to(gpu) # Ensure it's on GPU

            # --- Configure TeaCache ---
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            # --- K-Diffusion Sampling Callback ---
            def callback(d):
                """ Progress update callback during sampling. """
                try:
                    # Preview generation (use fake VAE decode for speed)
                    preview_latents = d['denoised'].to(vae.dtype) # Match VAE dtype
                    preview_pixels = vae_decode_fake(preview_latents) # Fast approx decode

                    # Convert preview to displayable numpy array
                    preview_pixels = (preview_pixels * 127.5 + 127.5).clamp(0, 255) # Denormalize [0, 255]
                    preview_np = preview_pixels.detach().cpu().numpy().astype(np.uint8)
                    # Rearrange B C T H W -> (B H) (T W) C for display
                    preview_display = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')

                    # Check for user interruption inside callback
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User ended the task.') # Stop sampler

                    # Update progress bar and description
                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f'Sampling Step {current_step}/{steps}'
                    # Calculate approx video length based on *total* generated frames so far
                    total_pixel_frames_so_far = max(0, total_generated_latent_frames * 4 - 3) # FramePack conversion
                    current_video_len_sec = total_pixel_frames_so_far / 30.0
                    desc = f'Total Generated Frames: {total_pixel_frames_so_far}, Video Length: {current_video_len_sec:.2f}s / {total_second_length:.1f}s (Target)'
                    stream.output_queue.push(('progress', (preview_display, desc, make_progress_bar_html(percentage, hint))))
                except Exception as e:
                    print(f"Error in callback: {e}") # Log errors but don't crash generation
                return

            print(f"Starting K-Sampler for {num_frames} latent frames...")
            # --- Actual Sampling Call ---
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc', # Or other K-diffusion samplers
                width=width,     # Pass the calculated width
                height=height,   # Pass the calculated height
                frames=num_frames, # Number of frames per window
                real_guidance_scale=cfg, # CFG scale (often 1.0 for FramePack)
                distilled_guidance_scale=gs, # Distilled CFG
                guidance_rescale=rs, # Guidance rescale (often 0.0)
                num_inference_steps=steps,
                generator=rnd,
                # --- Prompts & Embeddings ---
                prompt_embeds=llama_vec.to(gpu, dtype=transformer.dtype),
                prompt_embeds_mask=llama_attention_mask.to(gpu),
                prompt_poolers=clip_l_pooler.to(gpu, dtype=transformer.dtype),
                negative_prompt_embeds=llama_vec_n.to(gpu, dtype=transformer.dtype),
                negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu),
                negative_prompt_poolers=clip_l_pooler_n.to(gpu, dtype=transformer.dtype),
                image_embeddings=image_encoder_last_hidden_state.to(gpu, dtype=transformer.dtype),
                # --- FramePack Conditioning Latents ---
                latent_indices=latent_indices.to(gpu), # Indices defining noise area
                clean_latents=clean_latents.to(gpu, dtype=transformer.dtype),
                clean_latent_indices=clean_latent_indices.to(gpu),
                clean_latents_2x=clean_latents_2x.to(gpu, dtype=transformer.dtype),
                clean_latent_2x_indices=clean_latent_2x_indices.to(gpu),
                clean_latents_4x=clean_latents_4x.to(gpu, dtype=transformer.dtype),
                clean_latent_4x_indices=clean_latent_4x_indices.to(gpu),
                # --- Other ---
                device=gpu,
                dtype=transformer.dtype, # Use bfloat16 for transformer
                callback=callback,
                callback_steps=1 # Call callback every step
            )
            print(f"Sampling finished. Generated latent shape: {generated_latents.shape}") # B, C, T_window, H/8, W/8

            # --- VAE Decoding Section ---
            stream.output_queue.push(('progress', (None, 'Decoding frames...', make_progress_bar_html(100, 'VAE Decoding...'))))

            if not high_vram:
                print("Offloading Transformer, Loading VAE...")
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8) # Keep some VRAM free
                load_model_as_complete(vae, target_device=gpu)
            else:
                vae.to(gpu) # Ensure VAE is on GPU

            # --- Append Start Frame (if last section) ---
            # FramePack generates conditioning->content, so start frame is added at the end
            if is_last_section:
                print("Prepending start latent for the final section.")
                # Ensure start_latent matches dtype and device of generated_latents
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            # --- Update History & Decode ---
            current_latent_chunk = generated_latents.to(vae.device, dtype=vae.dtype) # Move to VAE device/dtype

            # Update total frame count *before* potentially modifying history_latents
            num_new_latents = current_latent_chunk.shape[2]
            total_generated_latent_frames += num_new_latents

            # Update history latents (on CPU to save VRAM, convert back if needed later)
            # Prepend new latents to history for correct temporal order (0 is oldest)
            if i == 0: # First iteration, just use the generated latents
                history_latents = current_latent_chunk.cpu()
            else:
                history_latents = torch.cat([current_latent_chunk.cpu(), history_latents], dim=2)

            print(f"Decoding {num_new_latents} new latent frames...")
            # Decode only the *newly generated* latent chunk for efficiency
            current_pixels = vae_decode(current_latent_chunk, vae).cpu() # Decode on GPU, move to CPU
            print(f"Decoded pixel shape for current chunk: {current_pixels.shape}") # B, C, T_new, H, W

            # --- Stitch Video Frames ---
            # Use soft append to blend overlapping frames smoothly
            if history_pixels is None: # First chunk decoded
                history_pixels = current_pixels
            else:
                # Overlap calculation from original FramePack logic
                # Corresponds to the overlap between consecutive sampling windows in *pixel* frames
                overlapped_frames = latent_window_size * 4 - 3
                print(f"Appending pixels with overlap: {overlapped_frames} frames")
                # Append the *new* pixels to the *existing* history
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap_frames=overlapped_frames)

            print(f"Total history pixel shape: {history_pixels.shape}")

            # --- Save Intermediate/Final Video ---
            output_filename = os.path.join(outputs_folder, f'{job_id}_progress_{i+1}.mp4')
            print(f"Saving video progress to: {output_filename}")
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            stream.output_queue.push(('file', output_filename)) # Send file path to Gradio UI

            # --- Cleanup for next iteration (Low VRAM) ---
            if not high_vram:
                print("Offloading VAE after decoding...")
                offload_model_from_device_for_memory_preservation(vae, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                torch.cuda.empty_cache()

            if is_last_section:
                print("Last section processed.")
                break # Exit loop after the final section

    except KeyboardInterrupt:
        print("Generation process interrupted by user (KeyboardInterrupt).")
        stream.output_queue.push(('end', None)) # Signal clean end to UI
    except Exception as e:
        print("\n--- ERROR DURING GENERATION ---")
        traceback.print_exc()
        stream.output_queue.push(('error', str(e))) # Send error message to UI if possible
    finally:
        # --- Final Cleanup ---
        print("Generation finished or stopped. Cleaning up...")
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            torch.cuda.empty_cache()
        print("Worker finished.")
        # Ensure the 'end' signal is sent if not already done
        if stream.output_queue.queue and stream.output_queue.queue[-1] != ('end', None):
             stream.output_queue.push(('end', None))


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, selected_resolution): # Added selected_resolution
    """Gradio button click handler."""
    global stream
    if input_image is None:
        gr.Warning("Please provide an input image.")
        # Return updates for all outputs to reset state
        yield None, gr.update(visible=False), "", "", gr.update(interactive=True), gr.update(interactive=False)
        return

    # Reset UI state and start worker
    yield None, gr.update(visible=False), "", make_progress_bar_html(0, 'Initializing...'), gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream() # Recreate stream for new job

    # Run the worker function asynchronously
    async_run(worker, input_image, prompt, n_prompt, int(seed), total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, selected_resolution) # Pass selected_resolution

    output_filename = None # Track the latest generated file

    # --- Stream Processing Loop ---
    # Read results from the worker thread's output queue
    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data # Update the latest video file path
            # Update video player, keep other elements as they are
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'progress':
            preview, desc, html = data
            # Update preview image, description text, and progress bar
            yield output_filename, gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        elif flag == 'error':
            gr.Error(f"An error occurred: {data}")
            # Reset UI to interactive state on error
            yield output_filename, gr.update(visible=False), "Error occurred.", "", gr.update(interactive=True), gr.update(interactive=False)
            break # Stop processing this job

        elif flag == 'end':
            print("Received end signal from worker.")
            # Final update: show the last video, hide preview, clear progress, enable start button
            yield output_filename, gr.update(visible=False), "Finished.", "", gr.update(interactive=True), gr.update(interactive=False)
            break # Exit the loop


def end_process():
    """Callback for the 'End Generation' button."""
    print("Stop button clicked. Sending 'end' signal to worker.")
    if 'stream' in globals() and stream:
        stream.input_queue.push('end')
    # Note: UI update (disabling stop button) happens in the main process loop when 'end' is received


# --- Gradio UI Definition ---

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
    'A close-up shot of an eye blinking.',
    'Ocean waves crashing on the shore.',
    'A steaming cup of coffee on a table.',
    'Leaves falling from a tree in autumn.',
    'A cat slowly stretching.',
]
quick_prompts = [[x] for x in quick_prompts] # Format for gr.Dataset


css = make_progress_bar_css() # Get CSS for the progress bar
block = gr.Blocks(css=css, theme=gr.themes.Soft()).queue() # Use queue for handling concurrent users

with block:
    gr.Markdown("# FramePack Image-to-Video")
    gr.Markdown("Generate short videos from a starting image and a text prompt using the FramePack method with Hunyuan-DiT.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(sources='upload', type="numpy", label="Input Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='A cinematic shot of a woman walking on the street.', lines=2)

            gr.Markdown("#### Quick Prompts")
            example_quick_prompts = gr.Dataset(
                samples=quick_prompts,
                label='Quick Prompt List',
                samples_per_page=10,
                components=[prompt],
                type="index" # Use index to avoid component mismatch errors
            )
            # Use js to set value directly, safer than lambda with Dataset index
            example_quick_prompts.click(
                fn=None, inputs=[example_quick_prompts], outputs=[prompt],
                 _js=f"(i) => {quick_prompts}[i][0]"
            )


            with gr.Row():
                start_button = gr.Button("Start Generation", variant="primary")
                end_button = gr.Button("Stop Generation", interactive=False)

            with gr.Accordion("Generation Settings", open=False):
                with gr.Group():
                    selected_resolution = gr.Dropdown(
                        label="Target Resolution (Approx.)",
                        choices=RESOLUTION_CHOICES,
                        value=DEFAULT_RESOLUTION,
                        info="Select target resolution. Final dimensions depend on input aspect ratio and model buckets. Higher resolutions require significantly more VRAM and time."
                    )
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1.0, maximum=20.0, value=4.0, step=0.1)
                    seed = gr.Number(label="Seed", value=1234, precision=0, interactive=True)
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Potentially faster speed, but may slightly affect fine details like fingers.')
                    mp4_crf = gr.Slider(label="MP4 CRF (Quality)", minimum=0, maximum=51, value=18, step=1, info="Lower value means better quality (larger file size). 0 is lossless. 18-28 is typical range.")

            with gr.Accordion("Advanced Settings", open=False):
                 with gr.Group():
                    steps = gr.Slider(label="Steps", minimum=10, maximum=100, value=25, step=1, info='Default: 25. Changing this value is generally not recommended for FramePack.')
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.1, info='Guidance scale for the distilled model. Default: 10.0. Changing is generally not recommended.')
                    gpu_memory_preservation = gr.Slider(label="GPU Preserved Memory (GB, Low VRAM Mode)", minimum=2, maximum=24, value=6 if not high_vram else 8, step=0.5, info="Memory (GB) to keep free when offloading models. Increase if OOM occurs, decrease for potentially faster speed. Only active in low VRAM mode.")
                    # Hidden/Unused parameters from original code - keep structure but hide
                    n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)
                    latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change


        with gr.Column(scale=1):
            gr.Markdown("### Output")
            result_video = gr.Video(label="Generated Video", autoplay=True, show_share_button=False, height=512, interactive=False)
            preview_image = gr.Image(label="Live Preview (Latent Approximation)", height=256, visible=False, interactive=False)
            gr.Markdown("ℹ️ *Note: Due to the generation method, the end of the video is generated first. The full sequence appears gradually.*")
            progress_desc = gr.Markdown('', elem_classes='generating-info') # Placeholder for text like 'Sampling 5/20'
            progress_bar = gr.HTML('', elem_classes='generating-progress') # Placeholder for the HTML progress bar

    gr.Markdown("---")
    gr.HTML('<div style="text-align:center; margin-top:10px;">Model Credits: <a href="https://huggingface.co/hunyuanvideo-community/HunyuanVideo" target="_blank">HunyuanVideo Community</a> | <a href="https://huggingface.co/lllyasviel/FramePackI2V_HY" target="_blank">FramePack Weights by lllyasviel</a> | UI based on FramePack Gradio</div>')


    # --- Event Listeners ---
    # Define inputs list matching the 'process' function signature
    ips = [
        input_image, prompt, n_prompt, seed, total_second_length, latent_window_size,
        steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, selected_resolution # Added selected_resolution
    ]
    # Define outputs list matching the 'yield' statements in 'process'
    ops = [
        result_video, preview_image, progress_desc, progress_bar, start_button, end_button
    ]

    start_button.click(fn=process, inputs=ips, outputs=ops)
    end_button.click(fn=end_process, inputs=None, outputs=None, cancels=[start_button.click]) # Allow stop button to cancel the generation


# --- Launch App ---
try:
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        prevent_thread_lock=True # Important for async operations
    )
except Exception as e:
     print(f"Error launching Gradio: {e}")
     print("If you get a port conflict, try specifying a different port using --port <number>")
