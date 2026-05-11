import csv

rows = []
next_id = 1

with open("dataset_generation/translated_nouns.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        text = row["translation"].strip()
        if not text:
            continue

        rows.append({"id": str(next_id), "khmer_text": text})
        next_id += 1

with open("dataset_generation/khmer_name_corpus.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        name_text = row.get("text", row.get("khmer_name", "")).strip()
        if not name_text:
            continue

        rows.append({"id": str(next_id), "khmer_text": name_text})
        next_id += 1

with open("CambodiaDistrictList2025.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        district_text = row.get("khmer_text", row.get("district_kh", "")).strip()
        if not district_text:
            continue

        rows.append({"id": str(next_id), "khmer_text": district_text})
        next_id += 1

with open("dataset_generation/merged_corpus.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "khmer_text"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Done. {len(rows)} rows written to dataset_generation/merged_corpus.csv")