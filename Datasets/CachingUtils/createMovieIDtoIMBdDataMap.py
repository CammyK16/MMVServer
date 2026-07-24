# Create a map from Movie IDs to IMDb Data using the data in first_200_movie_ids.txt and imdb_data.txt

import json
import ast


with open("first_200_movie_ids.txt", "r") as f:
    movie_ids = [line.strip() for line in f.readlines()]

formattedMovieIDs = [f"0" * (6 - len(movie_id)) + movie_id for movie_id in movie_ids]

with open("imdb_data.txt", "r", encoding="utf-8") as f:
    imdb_data = [ast.literal_eval(line.strip()) for line in f.readlines()]

with open("movie_id_to_imdb_data_map.json", "w", encoding="utf-8") as f:
    json.dump(dict(zip(formattedMovieIDs, imdb_data)), f, indent=2, ensure_ascii=False)


