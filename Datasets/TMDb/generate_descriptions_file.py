import json
import os
from pprint import pprint
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTERS_DIR = f"{REPO_ROOT}/Datasets/TMDb/posters"

paths = []
for file in os.listdir(POSTERS_DIR):
    filename = os.fsdecode(file)

    id = filename[:filename.find("-")]
    paths.append({"path":f"{POSTERS_DIR}/{filename}", "description": id})

with open(f"{REPO_ROOT}/Datasets/TMDb/descriptions.json", "w+", encoding="utf-8") as f:
    json.dump(paths,f)