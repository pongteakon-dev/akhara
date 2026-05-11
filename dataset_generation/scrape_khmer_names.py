#!/usr/bin/env python3
"""Build a Khmer name-part corpus from the Perchance name generator source.

This script pulls the generator payload from:
https://perchance.org/api/downloadGenerator?generatorName=namegen-as-kh

It extracts masculine names (nameM), feminine names (nameF), and surnames,
keeps Khmer Unicode only, deduplicates snippets, and writes a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SOURCE_URL = "https://perchance.org/api/downloadGenerator?generatorName=namegen-as-kh"
KHMER_RE = re.compile(r"[\u1780-\u17FF]+")
ENTRY_RE = re.compile(r"\(([^)]*)\)(?:\^(\d+(?:\.\d+)?))?$")
DISALLOWED_END_CHAR = "្"


@dataclass(frozen=True)
class WeightedName:
    text: str
    weight: float


def is_valid_name_part(name_part: str) -> bool:
    return bool(name_part) and not name_part.endswith(DISALLOWED_END_CHAR)


def fetch_generator_source(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    # The generator payload is URL-encoded inside HTML and often keeps line
    # breaks as escaped text ("\\n"). Decode and normalize so list sections are
    # easy to parse line-by-line.
    decoded = urllib.parse.unquote(raw)
    return decoded.replace("\\r\\n", "\n").replace("\\n", "\n")


def extract_section(decoded_html: str, section_name: str) -> str:
    marker = f"\n{section_name}\n"
    start = decoded_html.find(marker)
    if start == -1:
        raise ValueError(f"Section '{section_name}' not found in payload")

    start += len(marker)
    # Sections in this generator are contiguous and terminated by the next
    # list label or explicit END marker.
    candidates = []
    for next_marker in ("\nnameM\n", "\nnameF\n", "\nsurname\n", "\n// END //"):
        idx = decoded_html.find(next_marker, start)
        if idx != -1:
            candidates.append(idx)

    if not candidates:
        raise ValueError(f"Could not detect end of section '{section_name}'")

    end = min(candidates)
    return decoded_html[start:end]


def parse_weighted_names(section_text: str) -> list[WeightedName]:
    names: list[WeightedName] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        match = ENTRY_RE.search(line)
        if not match:
            continue

        khmer_in_parens = match.group(1)
        weight = float(match.group(2)) if match.group(2) is not None else 1.0
        khmer_only = "".join(KHMER_RE.findall(khmer_in_parens))
        if is_valid_name_part(khmer_only):
            names.append(WeightedName(text=khmer_only, weight=weight))

    if not names:
        raise ValueError("No Khmer names were parsed from section")

    return names


def load_existing_snippets(existing_csv: Path | None) -> set[str]:
    if existing_csv is None or not existing_csv.exists():
        return set()

    known: set[str] = set()
    with existing_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        preferred_columns = (
            "name_part",
            "text",
            "khmer_name",
            "name",
            "word",
            "khmer_word",
        )

        for row in reader:
            if not row:
                continue

            values: list[str] = []
            for column in preferred_columns:
                value = row.get(column)
                if value:
                    values.append(value)

            if not values:
                values = [str(v) for v in row.values() if v]

            for value in values:
                for token in str(value).split():
                    khmer_only = "".join(KHMER_RE.findall(token))
                    if is_valid_name_part(khmer_only):
                        known.add(khmer_only)

    return known


def generate_snippet_rows(
    surnames: list[WeightedName],
    male_names: list[WeightedName],
    female_names: list[WeightedName],
    seed: int | None,
    max_count: int,
    unique: bool,
    existing_snippets: set[str],
) -> list[tuple[int, str]]:
    snippets: list[str] = []
    seen_texts: set[str] = set(existing_snippets)

    for item in surnames:
        text = " ".join(item.text.split())
        if not is_valid_name_part(text):
            continue
        if unique and text in seen_texts:
            continue
        snippets.append(text)
        seen_texts.add(text)

    for item in male_names:
        text = " ".join(item.text.split())
        if not is_valid_name_part(text):
            continue
        if unique and text in seen_texts:
            continue
        snippets.append(text)
        seen_texts.add(text)

    for item in female_names:
        text = " ".join(item.text.split())
        if not is_valid_name_part(text):
            continue
        if unique and text in seen_texts:
            continue
        snippets.append(text)
        seen_texts.add(text)

    rng = random.Random(seed)
    rng.shuffle(snippets)

    if max_count > 0:
        snippets = snippets[:max_count]

    return [(index, snippet) for index, snippet in enumerate(snippets, start=1)]


def write_csv(rows: list[tuple[int, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Khmer name snippets from the Perchance generator source, "
            "deduplicate surname/given-name items, and export CSV."
        )
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=0,
        help=(
            "Maximum number of snippets to export after deduplication "
            "(default: 0, export all)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("khmer_name_parts.csv"),
        help="Output CSV path (default: khmer_name_parts.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible output",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate snippets across surname/given-name pools",
    )
    parser.add_argument(
        "--existing-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV path of existing items; overlapping Khmer snippets "
            "will be skipped"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 0:
        print("--count must be >= 0", file=sys.stderr)
        return 2

    try:
        decoded = fetch_generator_source(SOURCE_URL)
        male_section = extract_section(decoded, "nameM")
        female_section = extract_section(decoded, "nameF")
        surname_section = extract_section(decoded, "surname")

        male_names = parse_weighted_names(male_section)
        female_names = parse_weighted_names(female_section)
        surnames = parse_weighted_names(surname_section)

        existing_snippets = load_existing_snippets(args.existing_csv)

        rows = generate_snippet_rows(
            surnames=surnames,
            male_names=male_names,
            female_names=female_names,
            seed=args.seed,
            max_count=args.count,
            unique=not args.allow_duplicates,
            existing_snippets=existing_snippets,
        )
        if args.count > 0 and len(rows) < args.count:
            print(
                (
                    f"Requested {args.count} snippets, generated {len(rows)} rows. "
                    "Use --allow-duplicates if you need more rows."
                ),
                file=sys.stderr,
            )

        write_csv(rows, args.output)
        print(f"Wrote {len(rows)} rows to {args.output}")
        return 0
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Network error while fetching generator source: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Parsing error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
