import json
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
import faiss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math


class PosterSet(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.preprocess = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.preprocess(image), i


def main():
    parser = argparse.ArgumentParser(description="Add new embeddings to an existing FAISS index")
    parser.add_argument("--index", type=str, required=True,
                        help="Path to existing .faiss index file")
    parser.add_argument("--descriptions", type=str, required=True,
                        help="Path to JSON file with new entries (same format as descriptions.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for updated index (defaults to overwriting --index)")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    output_path = args.output or args.index

    # Load existing index
    print(f"Loading existing index from {args.index}...")
    index = faiss.read_index(args.index)
    existing_count = index.ntotal
    print(f"Existing index contains {existing_count} vectors (dim={index.d})")

    # Load new metadata
    with open(args.descriptions, "r") as f:
        new_metadata = json.load(f)

    new_paths = [row["image_path"] for row in new_metadata]
    N = len(new_paths)

    if N == 0:
        print("No new entries to add.")
        return

    print(f"Encoding {N} new images...")

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-g-14', pretrained='laion2b_s12b_b42k'
    )
    model = model.to("cuda")
    model.eval()

    # Verify dimensionality matches
    with torch.no_grad(), torch.autocast("cuda"):
        x0 = preprocess(Image.open(new_paths[0]).convert("RGB")).unsqueeze(0).to("cuda")
        d = int(model.encode_image(x0).shape[-1])

    if d != index.d:
        raise ValueError(f"Dimension mismatch: model produces {d}D vectors, "
                         f"but existing index expects {index.d}D")

    ds = PosterSet(new_paths, preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, prefetch_factor=4)

    added = 0
    with torch.no_grad(), torch.autocast("cuda"):
        for batch, _ in tqdm(dl, total=math.ceil(N / args.batch_size), desc="Embedding"):
            batch = batch.to("cuda", non_blocking=True)
            image_features = model.encode_image(batch).float()
            image_features = F.normalize(image_features, p=2, dim=1)
            arr = image_features.detach().cpu().numpy().astype("float32")
            index.add(arr)
            added += arr.shape[0]

    print(f"Added {added} vectors. Index now contains {index.ntotal} vectors.")

    faiss.write_index(index, output_path)
    print(f"Saved updated index to {output_path}")


if __name__ == "__main__":
    main()