# import csv
# import json

# csv_path = "./image_processing/dataset.csv"

# def extract_char_map(csv_path):
#     seen = set()

#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         next(reader, None)
#         for row in reader:
#             if len(row) < 4:
#                 continue
#             label = row[3].strip()
#             for char in label:
#                 seen.add(char)

#     # Sort by Unicode code point (natural Khmer alphabetical order)
#     sorted_chars = sorted(seen, key=lambda c: ord(c))
#     char_string = "".join(sorted_chars)
#     return char_string


# char_string = extract_char_map(csv_path)

# print(char_string)
# print(f"\n# Total unique characters: {len(char_string)}")


import csv
import json

csv_path = "./image_processing/dataset.csv"  # <-- change this to your file path

def extract_char_map(csv_path):
    unique_chars = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 4:
                continue
            label = row[3].strip()
            for char in label:
                if char not in seen:
                    seen.add(char)
                    unique_chars.append(char)

    unique_chars = sorted(seen, key=lambda c: ord(c))
    index_to_char = {str(idx): char for idx, char in enumerate(unique_chars)}
    return index_to_char

char_map = extract_char_map(csv_path)

print(json.dumps(char_map, ensure_ascii=False))

# Save to JSON
output_path = "char_map.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(char_map, f, ensure_ascii=False, indent=2)

print(f"\n# Total unique characters: {len(char_map)}")
print(f"# Saved to {output_path}")