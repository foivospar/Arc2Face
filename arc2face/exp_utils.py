''' 
FLAME regression code borrowed from SMIRK:
    - https://github.com/georgeretsi/smirk/blob/main/utils/mediapipe_utils.py
    - https://github.com/georgeretsi/smirk/blob/main/src/smirk_encoder.py
    - https://github.com/georgeretsi/smirk/blob/main/demo.py
'''

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp

import torch
import torch.nn.functional as F
from torch import nn
import timm

base_options = python.BaseOptions(model_asset_path='models/smirk/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    output_facial_transformation_matrixes=True,
                                    num_faces=1,
                                    min_face_detection_confidence=0.1,
                                    min_face_presence_confidence=0.1
                                    )
detector = vision.FaceLandmarker.create_from_options(options)

def run_mediapipe(image):
    # print(image.shape)    
    image_numpy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_numpy)


    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    if len (detection_result.face_landmarks) == 0:
        print('No face detected')
        return None
    
    face_landmarks = detection_result.face_landmarks[0]

    face_landmarks_numpy = np.zeros((478, 3))

    for i, landmark in enumerate(face_landmarks):
        face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

    return face_landmarks_numpy

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

def create_backbone(backbone_name, pretrained=True):
    backbone = timm.create_model(backbone_name, 
                        pretrained=pretrained,
                        features_only=True)
    feature_dim = backbone.feature_info[-1]['num_chs']
    return backbone, feature_dim

class ExpressionEncoder(nn.Module):
    def __init__(self, n_exp=50) -> None:
        super().__init__()

        self.encoder, feature_dim = create_backbone('tf_mobilenetv3_large_minimal_100')
        
        self.expression_layers = nn.Sequential( 
            nn.Linear(feature_dim, n_exp+2+3) # num expressions + jaw + eyelid
        )

        self.n_exp = n_exp
        self.init_weights()


    def init_weights(self):
        self.expression_layers[-1].weight.data *= 0.1
        self.expression_layers[-1].bias.data *= 0.1


    def forward(self, img):
        features = self.encoder(img)[-1]
            
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


        parameters = self.expression_layers(features).reshape(img.size(0), -1)

        outputs = {}

        outputs['expression_params'] = parameters[...,:self.n_exp]
        outputs['eyelid_params'] = torch.clamp(parameters[...,self.n_exp:self.n_exp+2], 0, 1)
        outputs['jaw_params'] = torch.cat([F.relu(parameters[...,self.n_exp+2].unsqueeze(-1)), 
                                           torch.clamp(parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)

        return outputs

def run_smirk(encoder, image, device='cuda'):

    kpt_mediapipe = run_mediapipe(image[:,:,::-1])
    if kpt_mediapipe is None:
        return None
    kpt_mediapipe = kpt_mediapipe[..., :2]
    tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=224)
    cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    cropped_image = cv2.resize(cropped_image, (224,224))
    cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
    with torch.no_grad():
        outputs = encoder(cropped_image.to(device))

    return outputs