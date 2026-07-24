import json
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
import faiss
import requests
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = f"{REPO_ROOT}/Datasets/TMDb"
metadata = json.load(open(f"{ROOT_DIR}/descriptions.json"))


paths = [row["image_path"] for row in metadata]
titles = [row["description"] for row in metadata]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to("cuda")
model = model.half()
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def embed_image(img_path):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda", non_blocking=True, dtype=torch.float16)
        image_features = model.encode_image(image).float()
        image_features = F.normalize(image_features, p=2, dim=1)

    arr = image_features.detach().cpu().numpy().astype("float32")
    return arr


def predict_title(query_image_path, k=1, threshold = None):
    q = embed_image(query_image_path)
    D, I = index.search(q, k)
    # return np.take(titles,I)[0], D[0]
    best_idx, best_sim = I[0,0], D[0,0]

    if threshold is not None and best_sim < threshold:
        return None, float(best_sim)
    return titles[best_idx], float(best_sim)

index = faiss.read_index(f"{ROOT_DIR}/embeddings/index_ViT-B-32_laion2b_s34b_b79k_UPDATED.faiss")
res = faiss.StandardGpuResources()
try:
    cloner_options = faiss.GpuClonerOptions()
    if hasattr(cloner_options, "useFloat16"):
        cloner_options.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, cloner_options)
except (AttributeError, TypeError):
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
