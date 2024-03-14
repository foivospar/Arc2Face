<div align="center">

# Arc2Face: A Foundation Model of Human Faces

[Foivos Paraperas Papantoniou](https://foivospar.github.io/) &emsp; [Alexandros Lattas](https://alexlattas.com/) &emsp; [Stylianos Moschoglou](https://moschoglou.com/)   

[Jiankang Deng](https://jiankangdeng.github.io/) &emsp; [Bernhard Kainz](https://bernhard-kainz.com/) &emsp; [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)  

Imperial College London, UK

<a href='https://arc2face.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href=''><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/FoivosPar/Arc2Face'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>

</div>

This is the official implementation of **[Arc2Face](https://arc2face.github.io/)**, an ID-conditioned face model:

&emsp;✅ that generates high-quality images of any subject given only its ArcFace embedding, within a few seconds<br>
&emsp;✅ trained on the large-scale WebFace42M dataset offers superior ID similarity compared to existing models<br>
&emsp;✅ built on top of Stable Diffusion, can be extended to different input modalities, e.g. with ControlNet<br>

<img src='assets/teaser.gif'>

# News/Updates
- [2024/03/14] 🔥 We release Arc2Face.

# Installation
```bash
conda create -n arc2face python=3.10
conda activate arc2face

# Install requirements
pip install -r requirements.txt
```

# Download Models
The models can be downloaded manually from [HuggingFace](https://huggingface.co/FoivosPar/Arc2Face) or using python:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arc2face/diffusion_pytorch_model.safetensors", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/config.json", local_dir="./models")
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="encoder/pytorch_model.bin", local_dir="./models")
```
For face detection and ID-embedding extraction, download the [antelopev2](https://github.com/deepinsight/insightface/tree/master/python-package) package and place the checkpoints under `models/antelopev2`. We use an ArcFace trained on WebFace42M. Download `arcface.onnx` from [HuggingFace](https://huggingface.co/FoivosPar/Arc2Face) and put it in `models/antelopev2` or using python:
```python
hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir="./models/antelopev2")
```
and **delete** `glintr100.onnx` (the default backbone from insightface). The `models` folder structure should finally be:

```
  . ── models ──┌── antelopev2
                ├── arc2face
                └── encoder
```

# Usage

Load pipeline using [diffusers](https://huggingface.co/docs/diffusers/index):
```python
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

base_model = 'runwayml/stable-diffusion-v1-5'

encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None
    )
```
You can use use any SD-compatible schedulers and steps, just like with Stable Diffusion. By default, we use `DPMSolverMultistepScheduler` with 25 steps, which produces very good results in just a few seconds.
```python
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')
```
Pick an image and extract the ID-embedding:
```python
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

img = np.array(Image.open('assets/examples/joacquin.png'))[:,:,::-1]

faces = app.get(img)
faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
id_emb = project_face_embs(pipeline, id_emb)    # pass throught the encoder
```

<div align="center">
<img src='assets/examples/joacquin.png' style='width:25%;'>
</div>

Generate images:
```python
num_images = 4
images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
```
<div align="center">
<img src='assets/samples.jpg'>
</div>

## Start a local gradio demo
You can start a local demo for inference by running:
```python
python gradio_demo/app.py
```

## TODOs
- Release inference code for pose-controlled Arc2Face.
- Release training dataset.

## Citation
If you find Arc2Face useful for your research, please consider citing us:

```bibtex
```
