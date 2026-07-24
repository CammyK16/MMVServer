# 1. Read in the data from the JSON file
import json
from pathlib import Path

with open("imdb_to_omdb_map.json", "r") as f:
    imdb_to_omdb_map = json.load(f)

newData = {}
for i, key in enumerate(imdb_to_omdb_map):
    newData[i] = imdb_to_omdb_map[key]
    newData[i]["imdbID"] = i

imdb_to_omdb_map = newData

# 2.1 Return the movies as json that have a rating higher than the number input by the user (between 0 and 10, float)
def filter_by_rating(imdb_to_omdb_map, user_rating):
    filtered_movies = {}
    for movie in imdb_to_omdb_map.values():
        if "imdbRating" in movie and movie["imdbRating"] != "N/A":
            if float(movie["imdbRating"]) > user_rating:
                filtered_movies[movie["imdbID"]] = movie
    return filtered_movies

# 2.2 Return the movies as json that match the country input by the user (case insensitive)
def filter_by_country(imdb_to_omdb_map, user_country, excludeFlag=False):
    filtered_movies = {}
    for movie in imdb_to_omdb_map.values():
        if "Country" in movie and movie["Country"] != "N/A":
            if user_country.lower() in movie["Country"].lower():
                if not excludeFlag:
                    filtered_movies[movie["imdbID"]] = movie
            else:
                if excludeFlag:
                    filtered_movies[movie["imdbID"]] = movie
    return filtered_movies

# 2.3 Return the movies as json that match the genre input by the user (case insensitive)
def filter_by_genre(imdb_to_omdb_map, user_genre, excludeFlag=False):
    filtered_movies = {}
    for movie in imdb_to_omdb_map.values():
        if "Genre" in movie and movie["Genre"] != "N/A":
            if user_genre.lower() in movie["Genre"].lower():
                if not excludeFlag:
                    filtered_movies[movie["imdbID"]] = movie
            else:
                if excludeFlag:
                    filtered_movies[movie["imdbID"]] = movie
    return filtered_movies


# 3. Return a dictionary with the keys being countries that at least 1 film in the dataset was produced in, and the values being the number of films produced in that country
# Sorted from highest to lowest number of films produced in that country
def count_movies_by_country(imdb_to_omdb_map):

    country_count = {}
    for movie in imdb_to_omdb_map.values():
        if "Country" in movie and movie["Country"] != "N/A":
            countries = [country.strip() for country in movie["Country"].split(",")]
            for country in countries:
                if country in country_count:
                    country_count[country] += 1
                else:
                    country_count[country] = 1

    # Sort the dictionary by values in descending order
    sorted_country_count = dict(sorted(country_count.items(), key=lambda x: x[1], reverse=True))
    return sorted_country_count

# 3.2 Return a dictionary with the keys being genres that at least 1 film in the dataset was produced in, and the values being the number of films produced in that genre
# Sorted from highest to lowest number of films produced in that genre
def count_movies_by_genre(imdb_to_omdb_map):

    genre_count = {}
    for movie in imdb_to_omdb_map.values():
        if "Genre" in movie and movie["Genre"] != "N/A":
            genres = [genre.strip() for genre in movie["Genre"].split(",")]
            for genre in genres:
                if genre in genre_count:
                    genre_count[genre] += 1
                else:
                    genre_count[genre] = 1

    # Sort the dictionary by values in descending order
    sorted_genre_count = dict(sorted(genre_count.items(), key=lambda x: x[1], reverse=True))
    return sorted_genre_count

def main():
    genre_options = [
        "Drama",
        "Crime",
        "Adventure",
        "Comedy",
        "Action",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "Mystery",
        "Fantasy",
        "Horror",
        "Music",
        "Biography",
        "Documentary",
        "Animation",
        "War",
        "Family",
        "Sport",
        "History",
        "Musical",
        "Western",
        "Short",
        "News",
        "Film-Noir",
    ]

    country_options = [
        "United States",
        "United Kingdom",
        "France",
        "Germany",
        "Spain",
        "Italy",
        "Japan",
        "Sweden",
        "Canada",
        "Denmark",
        "Poland",
        "Finland",
        "Switzerland",
        "New Zealand",
        "West Germany",
        "Netherlands",
        "Norway",
        "Austria",
        "Australia",
        "Iceland",
        "Israel",
        "Mexico",
        "Belgium",
        "Kazakhstan",
        "Armenia",
        "Isle of Man",
        "Occupied Palestinian Territory",
        "Hong Kong",
        "Paraguay",
        "Uruguay",
        "Jordan",
        "Taiwan",
        "Malta",
        "Morocco",
        "Luxembourg",
        "Ireland",
        "South Africa",
        "United Arab Emirates",
        "Jamaica",
        "Czech Republic",
        "Hungary",
    ]

    rating_options = [value / 10 for value in range(46, 94)]
    qualifying_combinations = []
    seen_movie_id_lists = set()

    def apply_spec(results, spec_type, value, exclude_flag=False):
        if spec_type == "genre":
            return filter_by_genre(results, value, excludeFlag=exclude_flag)
        if spec_type == "country":
            return filter_by_country(results, value, excludeFlag=exclude_flag)
        if spec_type == "rating":
            return filter_by_rating(results, value)
        raise ValueError(f"Unknown filter type: {spec_type}")

    def format_spec(spec_type, value, exclude_flag=False):
        if spec_type == "rating":
            return f"Rating above {value:.1f}"
        if spec_type == "country":
            if exclude_flag:
                return f"Not Produced in {value}"
            return f"Produced in {value}"
        if exclude_flag:
            return f"not {value}"
        return value

    def format_comment(specs):
        parts = []
        for spec_type, value, exclude_flag in specs:
            if spec_type == "rating":
                parts.append(f"Rating above {value:.1f}")
            elif spec_type == "country":
                if exclude_flag:
                    parts.append(f"Not Produced in {value}")
                else:
                    parts.append(f"Produced in {value}")
            else:
                if exclude_flag:
                    parts.append(f"Not {value}")
                else:
                    parts.append(value)

        return " and ".join(parts)

    def evaluate_combination(specs):
        filtered_movies = imdb_to_omdb_map
        for spec_type, value, exclude_flag in specs:
            filtered_movies = apply_spec(filtered_movies, spec_type, value, exclude_flag)
            if not filtered_movies:
                break

        result_count = len(filtered_movies)
        if 9 <= result_count <= 11:
            movie_ids = list(filtered_movies.keys())
            movie_ids_key = tuple(movie_ids)
            if movie_ids_key in seen_movie_id_lists:
                return

            seen_movie_id_lists.add(movie_ids_key)
            qualifying_combinations.append(
                [movie_ids, format_comment(specs)]
            )

    # Single filters.
    for genre in genre_options:
        for exclude_flag in (False, True):
            evaluate_combination([("genre", genre, exclude_flag)])

    for country in country_options:
        for exclude_flag in (False, True):
            evaluate_combination([("country", country, exclude_flag)])

    for rating in rating_options:
        evaluate_combination([("rating", rating, False)])

    # Combinations of two filters.
    for genre in genre_options:
        for genre_exclude_flag in (False, True):
            for country in country_options:
                for country_exclude_flag in (False, True):
                    evaluate_combination(
                        [
                            ("genre", genre, genre_exclude_flag),
                            ("country", country, country_exclude_flag),
                        ]
                    )

    for genre in genre_options:
        for genre_exclude_flag in (False, True):
            for rating in rating_options:
                evaluate_combination(
                    [
                        ("genre", genre, genre_exclude_flag),
                        ("rating", rating, False),
                    ]
                )

    for country in country_options:
        for country_exclude_flag in (False, True):
            for rating in rating_options:
                evaluate_combination(
                    [
                        ("country", country, country_exclude_flag),
                        ("rating", rating, False),
                    ]
                )

    # # Combinations of three filters.
    # for genre in genre_options:
    #     for genre_exclude_flag in (False, True):
    #         for country in country_options:
    #             for country_exclude_flag in (False, True):
    #                 for rating in rating_options:
    #                     evaluate_combination(
    #                         [
    #                             ("genre", genre, genre_exclude_flag),
    #                             ("country", country, country_exclude_flag),
    #                             ("rating", rating, False),
    #                         ]
    #                     )

    output_path = Path(__file__).with_name("filter_search_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qualifying_combinations, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(qualifying_combinations)} matching combinations to {output_path}")
    print("[")
    for index, combination in enumerate(qualifying_combinations):
        comma = "," if index < len(qualifying_combinations) - 1 else ""
        print(f"    [{combination[0]}, \"{combination[1]}\"]{comma}")
    print("]")
    


main()