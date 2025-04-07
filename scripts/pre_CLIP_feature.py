import torch
import clip
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# sydney
path = "./dataset/UCM_captions/imgs"
output_dir='./dataset/clip_feature/ucm_224'


for root, dirs, files in os.walk(path):
    for i in files:
        i = os.path.join(path, i)
        image_name = i.split('/')[-1]
        image = preprocess(Image.open(i)).unsqueeze(0).to(device) # 1，3，224，224
        with torch.no_grad():
            image_features = model.encode_image(image) #
            # print(image_features.shape)
            # input()
            """
            Note: Delete part of the code in CLIP VisionTransformer as follows:
                if self.proj is not None:
                x = x @ self.proj
            """

        np.save(os.path.join(output_dir, str(image_name)), image_features.data.cpu().float().numpy())


