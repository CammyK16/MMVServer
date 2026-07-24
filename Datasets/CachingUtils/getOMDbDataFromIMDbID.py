import requests
import json

OMDb_API_KEY = "c0ab2db"

with open("imdb_ids.txt", "r") as f:
    movie_ids = [line.strip() for line in f.readlines()]

with open("omdb_data.txt", "w") as f:
    for movie_id in movie_ids:
        print(f"Getting metadata for movie ID: {movie_id}")
        response = requests.get(f"https://www.omdbapi.com/?i={movie_id}&apikey={OMDb_API_KEY}", headers={"accept":"application/json"})
        print(response.json())
        f.write(json.dumps(response.json()) + "\n")