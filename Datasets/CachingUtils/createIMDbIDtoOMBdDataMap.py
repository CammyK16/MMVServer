# Create a map from IMDb IDs to OMDb Data using the data in imdb_ids.txt and omdb_data.txt

import json

with open("imdb_ids.txt", "r") as f:
    imdb_ids = [line.strip() for line in f.readlines()]

with open("omdb_data.txt", "r") as f:
    omdb_data = [json.loads(line) for line in f.readlines()]

with open("imdb_to_omdb_map.json", "w") as f:
    json.dump(dict(zip(imdb_ids, omdb_data)), f, indent=2, ensure_ascii=False)


