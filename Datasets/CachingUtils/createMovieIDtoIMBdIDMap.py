# Create a map from Movie IDs to IMDb IDs using the data in first_200_movie_ids.txt and imdb_ids.txt

with open("first_200_movie_ids.txt", "r") as f:
    movie_ids = [line.strip() for line in f.readlines()]

with open("imdb_ids.txt", "r") as f:
    imdb_ids = [line.strip() for line in f.readlines()]

with open("movie_to_imdb_id_map.txt", "w") as f:
    f.write(str(dict(zip(movie_ids, imdb_ids))))


