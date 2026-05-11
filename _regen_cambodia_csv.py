import ast
import csv
from pathlib import Path

base = Path('/Users/kolbotpen/Documents/1-school/Sem8/Akhara')
sql_path = base / 'addresses.sql'
out_path = base / 'CambodiaDistrictList2025.csv'

records = []
for line in sql_path.read_text(encoding='utf-8').splitlines():
    if not line.startswith('INSERT INTO `addresses` VALUES ('):
        continue
    row_text = line[len('INSERT INTO `addresses` VALUES '):-1].replace('NULL', 'None')
    row = ast.literal_eval(row_text)
    type_kh = str(row[1]).strip().replace('\u200b', '').replace('\u200c', '').replace('\xa0', '').replace('\t', '')
    records.append({
        'id': int(row[0]),
        'type_kh': type_kh,
        'code': str(row[2]).strip(),
        'khmer_text': str(row[3]).strip(),
        'english_text': 'Siem Reap' if str(row[2]).strip() == '17' and type_kh == 'ខេត្ត' else str(row[4]).strip(),
        'parent': None if row[5] is None else int(row[5]),
    })

by_id = {record['id']: record for record in records}
root_code_by_id = {}
path_by_id = {}


def build_path(record_id):
    if record_id in path_by_id:
        return path_by_id[record_id]

    record = by_id[record_id]
    if record['parent'] is None:
        path = (record['id'],)
    else:
        path = build_path(record['parent']) + (record['id'],)

    path_by_id[record_id] = path
    return path


for record in records:
    current = record
    while current['parent'] is not None:
        current = by_id[current['parent']]
    root_code_by_id[record['id']] = current['code']

allowed_types = {
    'ខេត្ត': 'province',
    'ក្រុង': 'municipality',
    'ខណ្ឌ': 'district',
    'ស្រុក': 'district',
    'ឃុំ': 'commune',
    'សង្កាត់': 'commune',
}
focus_provinces = {'2', '12', '17'}

output_records = []
for record in records:
    type_en = allowed_types.get(record['type_kh'])
    if not type_en:
        continue
    if record['parent'] is None or root_code_by_id[record['id']] in focus_provinces:
        output_records.append((int(root_code_by_id[record['id']]), build_path(record['id']), type_en, record['khmer_text'], record['english_text']))

output_records.sort(key=lambda item: (item[0], item[1]))
output_rows = [
    [str(index), type_en, khmer_text, english_text]
    for index, (_, _, type_en, khmer_text, english_text) in enumerate(output_records, start=1)
]

with out_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'type', 'khmer_text', 'english_text'])
    writer.writerows(output_rows)

print(f'wrote {len(output_rows)} rows')
