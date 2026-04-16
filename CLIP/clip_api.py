import json
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
import faiss
from blacksheep import post, Application, status_code
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

app = Application()

ROOT_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "TMDb"
metadata = json.load(open(ROOT_DIR / "descriptions.json"))

paths = [row["image_path"] for row in metadata]
titles = [row["description"] for row in metadata]

model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model = model.to("cuda")
model.eval()
model = torch.compile(model, mode="reduce-overhead")
tokenizer = open_clip.get_tokenizer('ViT-g-14')

index = faiss.read_index(str(ROOT_DIR / "embeddings" / "index_ViT-g-14_laion2b_s12b_b42k_UPDATED_5.faiss"))
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

def embed_image(img_path):
    with torch.no_grad(), torch.autocast("cuda"):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda", non_blocking=True)
        image_features = model.encode_image(image).float()
        image_features = F.normalize(image_features, p=2, dim=1)

    arr = image_features.detach().cpu().numpy().astype("float32")
    return arr


_io_pool = ThreadPoolExecutor(max_workers=16)

def _load_one(p):
    return preprocess(Image.open(p).convert("RGB"))

def embed_images(img_paths, batch_size=64):
    all_features = []
    for start in range(0, len(img_paths), batch_size):
        # loading_start = datetime.now()
        chunk = img_paths[start:start + batch_size]
        imgs = list(_io_pool.map(_load_one, chunk))
        batch = torch.stack(imgs, dim=0).to("cuda", non_blocking=True)
        # loading_end = datetime.now()
        # embedding_start = datetime.now()
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = model.encode_image(batch).float()
            image_features = F.normalize(image_features, p=2, dim=1)
        # torch.cuda.synchronize()
        # embedding_end = datetime.now()
        # print(f"loading images: {(loading_end-loading_start).total_seconds()*1000}\tembedding images: {(embedding_end-embedding_start).total_seconds()*1000}")
        all_features.append(image_features)

    return torch.cat(all_features, dim=0).cpu().numpy().astype("float32")

@post("/api/clip")
async def clip_analyse(request):
    payload = await request.json()
    img_paths = sorted([str(p) for p in Path(payload.get("images_path")).rglob("*.jpg")])

    if not img_paths:
        return status_code(400, {"success": False, "error": "No path specified!"})

    # embed_start = datetime.now()
    q = embed_images(img_paths)
    # embed_end = datetime.now()

    # search_start = datetime.now()
    D, I = gpu_index.search(q, 1)
    # search_end = datetime.now()

    # print(f"embedding time: {(embed_end-embed_start).total_seconds()*1000}\nsearching time: {(search_end-search_start).total_seconds()*1000}")

    out = []
    for i in range(len(img_paths)):
        best_idx = int(I[i, 0])
        best_sim = float(D[i, 0])

        pred = None if best_sim < 0.5 else titles[best_idx]
        out.append({"prediction": pred, "confidence": best_sim})

    return {"movie_ids": out}

