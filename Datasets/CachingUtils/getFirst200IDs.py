import json



with open('TMDb/movie_ids_filtered.json', encoding="utf-8") as f:
    d = json.loads(f.read())

    ids = []
    for i in range(202):
        ids.append([d[i]['id'], d[i]['original_title']])

with open('first_200_movie_ids.txt', 'w', encoding="utf-8") as f:
    for i in range(2, 202):
        f.write(str(ids[i][0]) + "\n")
