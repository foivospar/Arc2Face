import sys
sys.path.append('./')

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)

from arc2face import CLIPTextModelWrapper, project_face_embs, image_align

from gdl.utils.FaceDetector import FAN
from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import preprocess_for_emoca

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random

import gradio as gr

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
app.prepare(ctx_id=0, det_size=(640, 640))

# Load pipeline
base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=dtype
)
unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=dtype
)
controlnet = ControlNetModel.from_pretrained(
    'models', subfolder="controlnet", torch_dtype=dtype
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)

# load and disable LCM
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipeline.disable_lora()

def toggle_lcm_ui(value):
    if value:
        return (
            gr.update(minimum=1, maximum=20, step=1, value=3),
            gr.update(minimum=0.1, maximum=10.0, step=0.1, value=1.0),
        )
    else:
        return (
            gr.update(minimum=1, maximum=100, step=1, value=25),
            gr.update(minimum=0.1, maximum=10.0, step=0.1, value=3.0),
        )

# Load Emoca
face_detector = FAN()
path_to_models = "external/emoca/assets/EMOCA/models"
model_name = 'EMOCA_v2_lr_mse_20'
mode = 'detail'
emoca_model, conf = load_model(path_to_models, model_name, mode)
emoca_model.to(device)
emoca_model.eval()

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_example():
    case = [
        [
            './assets/examples/freddie.png',
            './assets/examples/pose1.png',
        ],
        [
            './assets/examples/lily.png',
            './assets/examples/pose2.png',
        ],
        [
            './assets/examples/freeman.jpg',
            './assets/examples/pose3.jpg',
        ],
    ]
    return case

def run_example(img_file, ref_img_file):
    return generate_image(img_file, ref_img_file, 25, 3, 23, 2, False)

def run_emoca(img, ref_img):

    img_dict = preprocess_for_emoca(img, face_detector)
    img_dict['image'] = img_dict['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        codedict = emoca_model.encode(img_dict, training=False)
    
    bbox, bbox_type, landmarks = face_detector.run(np.array(ref_img.convert('RGB')), with_landmarks=True)
    if len(bbox) == 0:
        raise gr.Error(f"Face detection failed in reference image! Please try with another reference image.")
    if len(bbox)>1:   # select largest face
        sizes = [(b[2]-b[0])*(b[3]-b[1]) for b in bbox]
        idx = np.argmax(sizes)
        lmks = landmarks[idx]
    else:
        lmks = landmarks[0]
    ref_img_aligned = image_align(ref_img.copy(), lmks, output_size=512)
    ref_img_dict = preprocess_for_emoca(ref_img_aligned, face_detector)
    ref_img_dict['image'] = ref_img_dict['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        ref_codedict = emoca_model.encode(ref_img_dict, training=False)
        ref_codedict['shapecode'] = codedict['shapecode'].clone()
        ref_codedict['detailcode'] = codedict['detailcode'].clone()
        tform = ref_img_dict['tform'].unsqueeze(0).to(device)
        tform = torch.inverse(tform).transpose(1, 2)
        visdict = emoca_model.decode(ref_codedict, training=False, render_orig=True, original_image=ref_img_dict['original_image'].unsqueeze(0).to(device), tform=tform)

    cond_img = Image.fromarray(((visdict['normal_images'][0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    
    return ref_img_aligned, cond_img

def generate_image(image_path, ref_image_path, num_steps, guidance_scale, seed, num_images, use_lcm, progress=gr.Progress(track_tqdm=True)):

    if use_lcm:
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_lora()
    else:
        pipeline.disable_lora()
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    if image_path is None:
        raise gr.Error(f"Cannot find any input face image! Please upload a face image.")
    
    if ref_image_path is None:
        raise gr.Error(f"Cannot find any reference image! Please upload a reference image.")
    
    img = np.array(Image.open(image_path))[:,:,::-1]

    # Face detection and ID-embedding extraction
    faces = app.get(img)
    
    if len(faces) == 0:
        raise gr.Error(f"Face detection failed! Please try with another input face image.")
    
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    id_emb = torch.tensor(faces['embedding'], dtype=dtype)[None].to(device)
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(pipeline, id_emb)    # pass throught the encoder

    # pose extraction with EMOCA
    ref_img_a, cond_img = run_emoca(Image.open(image_path), Image.open(ref_image_path))
                    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Start inference...")        
    images = pipeline(
        image=cond_img,
        prompt_embeds=id_emb,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale, 
        num_images_per_prompt=num_images,
        generator=generator
    ).images

    return [ref_img_a, cond_img] + images

### Description
title = r"""
<h1>Arc2Face: A Foundation Model for ID-Consistent Human Faces</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://arc2face.github.io/' target='_blank'><b>Arc2Face: A Foundation Model for ID-Consistent Human Faces</b></a>.<br>
This demo uses Arc2Face with ControlNet to generate images of a subject with the pose extracted from a reference image.

Steps:<br>
1. Upload an image with a face. If multiple faces are detected, we use the largest one. For images with already tightly cropped faces, detection may fail, try images with a larger margin.
2. Upload a reference image for the pose (can be a different subject). We align this image to the FFHQ template for pose extraction (so, the final generated images correspond to the aligned pose). Again, if multiple faces are detected, we use the largest one.
2. Click <b>Submit</b> to generate new images of the input subject with the reference pose.

Note: For pose extraction and conditioning, we use a 3D reconstruction <a href='https://github.com/radekd91/emoca'>method</a> based on the FLAME model (we use only the rendered mesh normals as conditioning image, which is visualized alongside the output images). Reconstruction may fail in extreme poses (normals will correspond to random poses instead of the given one).
"""

Footer = r"""
---
📝 **Citation**
<br>
If you find Arc2Face helpful for your research, please consider citing our paper:
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

                # upload reference image
                ref_img_file = gr.Image(label="Upload a reference photo for the pose", type="filepath")
            
            submit = gr.Button("Submit", variant="primary")

            use_lcm = gr.Checkbox(
                label="Use LCM-LoRA to accelerate sampling", value=False,
                info="Reduces sampling steps significantly, but may decrease quality.",
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
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=3.0,
                )
                num_images = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
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
            gallery = gr.Gallery(label="Extracted Pose + Generated Images")

        submit.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[img_file, ref_img_file, num_steps, guidance_scale, seed, num_images, use_lcm],
            outputs=[gallery]
        )

    use_lcm.input(
            fn=toggle_lcm_ui,
            inputs=[use_lcm],
            outputs=[num_steps, guidance_scale],
            queue=False,
        )
    
    gr.Examples(
        examples=get_example(),
        inputs=[img_file, ref_img_file],
        run_on_click=True,
        fn=run_example,
        outputs=[gallery],
    )
    
    gr.Markdown(Footer)

demo.launch()