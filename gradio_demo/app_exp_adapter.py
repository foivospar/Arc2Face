import sys
sys.path.append('./')

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)

from arc2face import CLIPTextModelWrapper, project_face_embs, image_align, ReferenceAdapter

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random

import gradio as gr

from arc2face.exp_utils import ExpressionEncoder, run_smirk

import face_alignment

# global variable
MAX_SEED = np.iinfo(np.int32).max
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# Load face detection and recognition package
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))

# Load landmark detector
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

# Load pipeline
base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=dtype
)
unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=dtype
)
pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None
    )
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)

# load Expression Adapter
pipeline.load_ip_adapter(
    "models", 
    subfolder="exp_adapter", 
    weight_name="exp_adapter.bin", 
    image_encoder_folder=None
)

# load LoRA weights for Reference Adapter
pipeline.load_lora_weights(
    "models/ref_adapter", 
    weight_name="pytorch_lora_weights.safetensors",
    adapter_name="ref"
)

# load Reference UNet
ref_unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=dtype
).to(device)

# Create (optional) Reference Adapter
ref_adapter_w = ReferenceAdapter(
    ref_unet,
    mode="write",
)
ref_adapter_r = ReferenceAdapter(
    pipeline.unet,
    mode="read",
    cfg=True,
)

# load SMIRK expression encoder (predicts FLAME parameters from images)
smirk_encoder = ExpressionEncoder(n_exp=50).to(device)
checkpoint = torch.load('models/smirk/SMIRK_em1.pt')
checkpoint_encoder = {k.replace('smirk_encoder.expression_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder.expression_encoder.' in k}
smirk_encoder.load_state_dict(checkpoint_encoder)
smirk_encoder.eval()


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_example():
    case = [
        [
            './assets/examples/freddie.png',
            './assets/examples/exp1.png',
        ],
        [
            './assets/examples/lily.png',
            './assets/examples/exp2.png',
        ],
        [
            './assets/examples/freeman.jpg',
            './assets/examples/exp3.png',
        ],
        [
            './assets/examples/hepburn.png',
            './assets/examples/exp4.png',
        ],
    ]
    return case

def generate_image(image_path, exp_image_path, use_ref_adapter, lora_ref_scale, num_steps, guidance_scale, num_images, exp_adapter_scale, seed, progress=gr.Progress(track_tqdm=True)):

    pipeline.set_ip_adapter_scale(exp_adapter_scale)
    pipeline.set_adapters("ref", lora_ref_scale if use_ref_adapter else 0.0) 

    if image_path is None:
        raise gr.Error("Cannot find any input face image! Please upload a face image.")
    
    if exp_image_path is None:
        raise gr.Error("Cannot find any expression image! Please upload an image.")
    
    img = Image.open(image_path)

    if use_ref_adapter:
        # Align input image to FFHQ template
        face_landmarks, _, bboxes = fa.get_landmarks(np.array(img), return_bboxes=True)
        if face_landmarks is None:
            raise gr.Error("Face detection failed! Please try with another input face image.")
        else:
            if len(face_landmarks)>1:   # keep the largest face
                sizes = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
                idx = np.argmax(sizes)
                lmks = face_landmarks[idx]
            else:
                lmks = face_landmarks[0]
        img = image_align(img, lmks, output_size=512)
    
    # Face detection and ID-embedding extraction
    faces = app.get(np.array(img)[:,:,::-1])
    
    if len(faces) == 0:
        raise gr.Error("Face detection failed! Please try with another input face image.")
    
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    id_emb = torch.tensor(faces['embedding'], dtype=dtype)[None].to(device)
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

    # Get FLAME expression params (Smirk) from the expression image
    exp_image = np.array(Image.open(exp_image_path))    
    outputs = run_smirk(smirk_encoder, exp_image, device=device)
    if outputs is None:
        raise gr.Error(f"Face detection failed! Please try with another expression image.")
    exp_embs = torch.cat([outputs['expression_params'], outputs['eyelid_params'], outputs['jaw_params']], dim=1).to(dtype=dtype)
    exp_adapter_embeds = torch.cat([torch.zeros_like(exp_embs[:,None,:]), exp_embs[:,None,:]], dim=0)
    exp_adapter_embeds = exp_adapter_embeds.repeat_interleave(repeats=num_images, dim=0)
    
    generator = torch.Generator(device=device).manual_seed(seed)

    if use_ref_adapter:   # run a forward pass through the Reference UNet to update the Reference Adapter
        ref_img = (torch.tensor(np.array(img), dtype=dtype).to(device).permute(2,0,1)/255)*2-1
        ref_img = torch.stack([ref_img, ref_img]).repeat_interleave(repeats=num_images, dim=0)
        ref_img = pipeline.vae.encode(ref_img).latent_dist.sample()
        ref_img = ref_img * pipeline.vae.config.scaling_factor
        encoder_hidden_states = torch.cat([id_emb, id_emb], dim=0).repeat_interleave(repeats=num_images, dim=0)

        ref_unet(
            ref_img,
            torch.zeros(ref_img.size(0), device=ref_img.device).long(),
            encoder_hidden_states,
            return_dict=False,
        )
        ref_adapter_r.update(ref_adapter_w)

    print("Start inference...")
    images = pipeline(
        prompt_embeds=id_emb.repeat_interleave(repeats=num_images, dim=0),
        ip_adapter_image_embeds=[exp_adapter_embeds],
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale, 
        generator=generator
    ).images

    ref_adapter_r.clear()
    ref_adapter_w.clear()

    return images

### Description
title = r"""
<h1>Arc2Face with Expression Adapter</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='http://arxiv.org/abs/2510.04706'><b>ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion</b></a>.<br>
This demo integrates an <b>Expression Adapter</b> into <a href='https://arc2face.github.io/'>Arc2Face</a> for accurate and ID-consistent facial expression transfer.<br>
The target expression parameters (FLAME blendshapes) are extracted from the target expression image using the state-of-the-art <a href='https://github.com/georgeretsi/smirk'>SMIRK</a> method.<br>
Optionally, a <b>Reference Adapter</b> can be enabled to preserve the appearance and background of the input image.<br><br>
Steps:<br>
1. Upload an image with a face.
2. Upload a target image showing the desired expression.
3. Click:<br>
&emsp;• <b>Submit (unconditional)</b> to generate new images of the input subject with the target expression, or<br>
&emsp;• <b>Submit (reference-guided)</b> to additionally preserve the appearance and background of the input image while editing the expression.<br><br>

💡 <b>Hint:</b> When using the reference-guided mode, the Reference Adapter tends to “copy-paste” the input image, causing deviations from the target expression. You can mitigate this by reducing the <i>Reference Adapter scale factor</i>, which trades off pose and background consistency for improved expression fidelity. In practice, a value around <b>0.8</b> often provides a good balance.
"""

Footer = r"""
---
📝 **Citation**
<br>
If you find our model helpful for your research, please consider citing our papers:
```bibtex
@inproceedings{paraperas2025arc2face_exp,
      title={ID-Consistent, Precise Expression Generation with Blendshape-Guided Diffusion}, 
      author={Paraperas Papantoniou, Foivos and Zafeiriou, Stefanos},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
      year={2025}
}
```
and<br>
```bibtex
@inproceedings{paraperas2024arc2face,
      title={Arc2Face: A Foundation Model for ID-Consistent Human Faces}, 
      author={Paraperas Papantoniou, Foivos and Lattas, Alexandros and Moschoglou, Stylianos and Deng, Jiankang and Kainz, Bernhard and Zafeiriou, Stefanos},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2024}
}
```
"""

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:

    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                # upload face image
                img_file = gr.Image(label="Upload a photo with a face", type="filepath")

                # upload expression image
                exp_img_file = gr.Image(label="Upload a photo with the desired expression", type="filepath")
            
            with gr.Row():
                submit_btn = gr.Button("Submit (unconditional)", variant="primary")
                submit_ref_btn = gr.Button("Submit (reference-guided)", variant="primary")

            lora_ref_scale = gr.Slider(
                    label="Reference Adapter scale",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.8,
                )

            with gr.Accordion(open=False, label="Advanced Options"):
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=25,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1.1,
                    maximum=10.0,
                    step=0.1,
                    value=3.0,
                )
                num_images = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                )
                exp_adapter_scale = gr.Slider(
                    label="Expression Adapter scale",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=1.0,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")

        submit_btn.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[img_file, exp_img_file, gr.State(False), lora_ref_scale, num_steps, guidance_scale, num_images, exp_adapter_scale, seed],
            outputs=[gallery]
        )

        submit_ref_btn.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[img_file, exp_img_file, gr.State(True), lora_ref_scale, num_steps, guidance_scale, num_images, exp_adapter_scale, seed],
            outputs=[gallery]
        )
    
    gr.Examples(
        label = "Examples",
        examples=get_example(),
        inputs=[img_file, exp_img_file],
    )

    gr.Markdown(Footer)
    
demo.launch(share=True)