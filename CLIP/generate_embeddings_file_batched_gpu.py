import json
from PIL import Image
import numpy as np
import torch
import open_clip
import faiss
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import math
from pathlib import Path

class PosterSet(Dataset):
    def __init__(self, paths, transform, target_transform = None):
        self.paths = paths
        self.preprocess = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.preprocess(image), i

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = f"{REPO_ROOT}/Datasets/TMDb"
metadata = json.load(open(f"{ROOT_DIR}/descriptions.json"))

paths = [row["image_path"] for row in metadata]
N = len(paths)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to("cuda")
model = model.half()
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

BATCH_SIZE = 512    

ds = PosterSet(paths, preprocess)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers = 8, pin_memory=True, prefetch_factor=4)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    x0 = preprocess(Image.open(paths[0]).convert("RGB")).unsqueeze(0).to("cuda", non_blocking = True, dtype=torch.float16)
    d = int(model.encode_image(x0).shape[-1])

res = faiss.StandardGpuResources()
cfg = faiss.GpuIndexFlatConfig()
cfg.device = 0
gpu_index = faiss.GpuIndexFlatIP(res, d, cfg)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    for batch, i in tqdm(dl, total=math.ceil(N/BATCH_SIZE)):
        batch = batch.to("cuda", non_blocking = True, dtype=torch.float16)
        image_features = model.encode_image(batch).float()
        
        image_features_norm = F.normalize(image_features, p=2, dim=1)
        arr = image_features_norm.detach().cpu().numpy().astype("float32")
        gpu_index.add(arr)

index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(index, f"{ROOT_DIR}/embeddings/index_ViT-B-32_laion2b_s34b_b79k_UPDATED.faiss")