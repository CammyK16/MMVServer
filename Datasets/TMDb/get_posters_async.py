import requests
import json
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv

TMDB_IMAGE_URL_TEMPLATE = "https://image.tmdb.org/t/p/w500"
REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(f"{REPO_ROOT}/.env")

def make_session():
    s = requests.Session()
    retries = Retry(
        total = 10,
        backoff_factor = 0.5,
        status_forcelist = (429, 500, 502, 503, 504),
        allowed_methods = ["GET"]
    )

    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    return s

def fetch_images(session, url, path):
    r = session.get(url)
    r.raise_for_status()
    path.write_bytes(r.content)

def get_posters(session, id, file):
    r = session.get(f"https://api.themoviedb.org/3/movie/{id}/images?include_image_language=en-US,en-GB,null", headers={"Authorization": os.getenv("TMDB_API_KEY"), "accept": "application/json"})

    # Limit to just the first 5 posters since its not likely a movie will have more than 5 
    try:
        posters = r.json()["posters"][0:5]
    except KeyError as e:
        file.write(f"FAILED: {id} - {r.json()['status_message']}")
        return

    tasks = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for i, poster in enumerate(posters):
            url = TMDB_IMAGE_URL_TEMPLATE+poster["file_path"]
            path = Path(f"{REPO_ROOT}/Datasets/TMDb/posters/{id.zfill(6)}-{i}.jpg")
            
            futures.append(pool.submit(fetch_images, session, url, path))

        for fut in as_completed(futures):
            fut.result()
        
        tasks = futures
    return len(tasks)

session = make_session()

with open(f"{REPO_ROOT}/Datasets/TMDb/movie_ids_filtered.json", "r", encoding="utf-8") as f:
    movie_ids = json.load(f)

with open(f"{REPO_ROOT}/Datasets/TMDb/completed.txt", "a+") as completed_list_file:
    completed_list_file.seek(0)
    completed_movies = [line.strip() for line in completed_list_file]

    for movie in tqdm(movie_ids, total=len(movie_ids)):
        id = str(movie["id"])
        if id not in completed_movies:  
            try:
                downloaded = get_posters(session, id, completed_list_file)
            except requests.HTTPError as e:
                completed_list_file.write(f"FAILED: {id} - {e}\n")
                completed_list_file.flush()
                continue

            completed_list_file.write(f"{id}\n")
            completed_list_file.flush()