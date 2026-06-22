import csv

with open("image_processing/dataset.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Columns:", reader.fieldnames)
print("First 3 rows:", rows[:3])
print("Total rows:", len(rows))

# find all bad rows
for i, r in enumerate(rows):
    if not r.get("label"):
        print(f"Bad row {i}: {r}")