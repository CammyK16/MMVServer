import json
import unicodedata as ud

latin_letters= {}

def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def is_roman(unistr):
    return all(is_latin(uchr) for uchr in unistr if uchr.isalpha())

movie_ids_filtered = []

with open("movie_ids_09_10_2025.json", "r", encoding="utf-8") as f:
    for row in f:
        row_json = json.loads(row)

        if is_roman(row_json["original_title"]) and float(row_json["popularity"]) > 1.4 and row_json["video"] == False:
            movie_ids_filtered.append(row_json)

print(len(movie_ids_filtered))

with open("movie_ids_filtered.json", "w", encoding="utf-8") as f2:
    json.dump(movie_ids_filtered, f2, ensure_ascii=False, indent=4)