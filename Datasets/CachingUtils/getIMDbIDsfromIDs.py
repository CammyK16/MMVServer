import requests
TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ZTBmM2E0ZWIxODg0ZDc0ZmU1MWMyNGEwNTViY2Y1NyIsIm5iZiI6MTc1OTg2NjM2MC43NzEwMDAxLCJzdWIiOiI2OGU1NmRmODAxNTEzNGQ0NTY2MTg0ZDkiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.LTY89nro3Y_zytJrV2SVmxXy1DQelqOROVPDrqxKPN0"

with open("first_200_movie_ids.txt", "r") as f:
    movie_ids = [line.strip() for line in f.readlines()]

with open("imdb_ids.txt", "w") as f:
    for movie_id in movie_ids:
        print(f"Getting metadata for movie ID: {movie_id}")
        response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US", headers={"accept":"application/json", "Authorization": f"Bearer {TMDB_API_KEY}"})
        print(f"IMDb ID for {movie_id}: {response.json()['imdb_id']}")
        f.write(response.json()["imdb_id"] + "\n")

