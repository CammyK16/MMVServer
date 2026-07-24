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
from time import perf_counter

debug = True

import logging
log = logging.getLogger("uvicorn.error")
log.setLevel(logging.DEBUG)

app = Application()

ROOT_DIR = Path(__file__).resolve().parents[1] / "Datasets" / "TMDb"

startup_total_start = perf_counter()
metadata_start = perf_counter()
metadata = json.load(open(ROOT_DIR / "descriptions.json"))
# log.debug(f"Output time | startup.metadata_load_ms={(perf_counter() - metadata_start) * 1000:.2f}")

paths = [row["image_path"] for row in metadata]
titles = [row["description"] for row in metadata]

model_load_start = perf_counter()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model = model.to("cuda")
model = model.half()
model.eval()
model = torch.compile(model, mode="reduce-overhead")
# log.debug(f"Output time | startup.model_ready_ms={(perf_counter() - model_load_start) * 1000:.2f}")
tokenizer = open_clip.get_tokenizer('ViT-g-14')
# log.debug(f"Output time | startup.tokenizer_ready_ms={(perf_counter() - model_load_start) * 1000:.2f}")

faiss_load_start = perf_counter()
index = faiss.read_index(str(ROOT_DIR / "embeddings" / "index_ViT-g-14_laion2b_s12b_b42k_UPDATED_5.faiss"))
res = faiss.StandardGpuResources()
try:
    cloner_options = faiss.GpuClonerOptions()
    if hasattr(cloner_options, "useFloat16"):
        cloner_options.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, cloner_options)
except (AttributeError, TypeError):
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

try:
    import faiss.contrib.torch_utils  # noqa: F401
    FAISS_TORCH_INTEROP = True
    log.debug(f"Output time | startup.faiss_torch_interop=True")
except Exception:
    FAISS_TORCH_INTEROP = False
    log.debug(f"Output time | startup.faiss_torch_interop=False")

# log.debug(f"Output time | startup.faiss_torch_interop={FAISS_TORCH_INTEROP}")
# log.debug(f"Output time | startup.faiss_gpu_ready_ms={(perf_counter() - faiss_load_start) * 1000:.2f}")
# log.debug(f"Output time | startup.total_ms={(perf_counter() - startup_total_start) * 1000:.2f}")

def embed_image(img_path):
    embed_single_start = perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to("cuda", non_blocking=True, dtype=torch.float16)
        image_features = model.encode_image(image).float()
        image_features = F.normalize(image_features, p=2, dim=1)

    arr = image_features.detach().cpu().numpy().astype("float32")
    # log.debug(f"Output time | embed_image.total_ms={(perf_counter() - embed_single_start) * 1000:.2f}")
    return arr


_io_pool = ThreadPoolExecutor(max_workers=16)

def _load_one(p):
    return preprocess(Image.open(p).convert("RGB"))

def embed_images(img_paths, batch_size=64, return_numpy=True):
    embed_images_start = perf_counter()
    all_features = []
    for start in range(0, len(img_paths), batch_size):
        chunk_start = perf_counter()
        chunk = img_paths[start:start + batch_size]
        loading_start = perf_counter()
        imgs = list(_io_pool.map(_load_one, chunk))
        loading_ms = (perf_counter() - loading_start) * 1000

        transfer_start = perf_counter()
        batch = torch.stack(imgs, dim=0).to("cuda", non_blocking=True, dtype=torch.float16)
        transfer_ms = (perf_counter() - transfer_start) * 1000

        embedding_start = perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            image_features = model.encode_image(batch).float()
            image_features = F.normalize(image_features, p=2, dim=1)
        embedding_ms = (perf_counter() - embedding_start) * 1000
        chunk_ms = (perf_counter() - chunk_start) * 1000
        # log.debug(
        #     f"Output time | embed_images.chunk_start={start} chunk_size={len(chunk)} "
        #     f"load_ms={loading_ms:.2f} transfer_ms={transfer_ms:.2f} "
        #     f"encode_ms={embedding_ms:.2f} chunk_total_ms={chunk_ms:.2f}"
        # )
        all_features.append(image_features)

    cat_start = perf_counter()
    output = torch.cat(all_features, dim=0)
    # log.debug(f"Output time | embed_images.cat_ms={(perf_counter() - cat_start) * 1000:.2f}")

    if return_numpy:
        d2h_start = perf_counter()
        output = output.cpu().numpy()
        # log.debug(f"Output time | embed_images.d2h_ms={(perf_counter() - d2h_start) * 1000:.2f}")

    # log.debug(f"Output time | embed_images.total_ms={(perf_counter() - embed_images_start) * 1000:.2f}")
    return output

@post("/api/clip")
async def clip_analyse(request):
    request_start = perf_counter()
    payload_parse_start = perf_counter()
    payload = await request.json()
    # log.debug(f"Output time | clip_analyse.payload_parse_ms={(perf_counter() - payload_parse_start) * 1000:.2f}")

    path_collect_start = perf_counter()
    img_paths = sorted([str(p) for p in Path(payload.get("images_path")).rglob("*.jpg")])
    # log.debug(f"Output time | clip_analyse.collect_paths_ms={(perf_counter() - path_collect_start) * 1000:.2f} count={len(img_paths)}")

    if not img_paths:
        return status_code(400, {"success": False, "error": "No path specified!"})

    embed_start = perf_counter()
    use_torch_query = FAISS_TORCH_INTEROP
    q = embed_images(img_paths, return_numpy=not use_torch_query)
    # log.debug(f"Output time | clip_analyse.embed_images_ms={(perf_counter() - embed_start) * 1000:.2f}")

    search_start = perf_counter()
    D, I = gpu_index.search(q, 1)
    # log.debug(f"Output time | clip_analyse.faiss_search_ms={(perf_counter() - search_start) * 1000:.2f}")

    if torch.is_tensor(D):
        tensor_convert_start = perf_counter()
        D = D.detach().cpu().numpy()
        I = I.detach().cpu().numpy()
        # log.debug(f"Output time | clip_analyse.search_result_to_numpy_ms={(perf_counter() - tensor_convert_start) * 1000:.2f}")

    postprocess_start = perf_counter()
    out = []
    for i in range(len(img_paths)):
        best_idx = int(I[i, 0])
        best_sim = float(D[i, 0])

        pred = None if best_sim < 0.5 else titles[best_idx]
        out.append({"prediction": pred, "confidence": best_sim})

    # log.debug(f"Output time | clip_analyse.postprocess_ms={(perf_counter() - postprocess_start) * 1000:.2f}")
    # log.debug(f"Output time | clip_analyse.total_ms={(perf_counter() - request_start) * 1000:.2f}")

    return {"movie_ids": out}

