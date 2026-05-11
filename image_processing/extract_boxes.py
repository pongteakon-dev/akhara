"""
Handwriting Box Extractor
=========================
Detects handwriting boxes in scanned pages, reads the ID number printed
above-left of each box via OCR, crops the interior (no border), resizes
to exactly 960x384 px, saves as <word_id>.png, and appends a row to
dataset.csv.

dataset.csv columns:
    id          - auto-incrementing integer (continues across runs)
    word_id     - the number printed on the page next to the box
    image_path  - path to the saved crop
    writer_id   - auto-incremented per image file (1 image = 1 writer)
                  override the starting number with --writer-id-start
    label       - Khmer text from word_labels.csv (blank if not found)

Usage:
    python extract_boxes.py
    python extract_boxes.py --debug
    python extract_boxes.py --writer-id-start 3   # if adding more writers later

Requirements:
    pip install opencv-python pytesseract pandas numpy
    sudo apt-get install tesseract-ocr
    (macOS: brew install tesseract)
"""

import cv2
import numpy as np
import csv
import re
import argparse
from pathlib import Path

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠  pytesseract not installed.")
    print("   pip install pytesseract  +  apt/brew install tesseract-ocr")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_W        = 960
OUT_H        = 384
BORDER_INSET = 4   # px to inset from detected box edge to exclude border line


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(labels_csv):
    """Return {word_id (int) -> khmer_text (str)} from word_labels.csv."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    # word_id may appear multiple times (same word on multiple PDFs) — just
    # take the first occurrence; the text is the same for identical word_ids.
    return {int(row['word_id']): str(row['word_text'])
            for _, row in df.drop_duplicates('word_id').iterrows()}


# ── Dataset CSV helpers ───────────────────────────────────────────────────────

DATASET_FIELDS = ['id', 'image_path', 'writer_id', 'label']
PROCESSED_MANIFEST = 'converted_sources.txt'


def next_auto_id(dataset_csv: Path) -> int:
    """Return the next auto-increment id (max existing id + 1, or 1 if new)."""
    if not dataset_csv.exists():
        return 1
    max_id = 0
    with open(dataset_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row['id']))
            except (ValueError, KeyError):
                pass
    return max_id + 1


def append_rows(dataset_csv: Path, rows: list):
    """Append rows to dataset CSV, writing header if file is new."""
    file_exists = dataset_csv.exists()
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_processed_sources(manifest_path: Path) -> set[str]:
    """Return the set of source image filenames that were already processed."""
    if not manifest_path.exists():
        return set()
    with open(manifest_path, encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def append_processed_source(manifest_path: Path, source_name: str):
    """Record one processed source image filename."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'a', encoding='utf-8') as f:
        f.write(f'{source_name}\n')


# ── Box detection ─────────────────────────────────────────────────────────────

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix  = max(ax, bx);       iy  = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0


def _merge_rects(rects, iou_thresh=0.35):
    if not rects:
        return rects
    rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
    keep = []
    for r in rects:
        if all(_iou(r, k) < iou_thresh for k in keep):
            keep.append(r)
    return keep


def find_boxes(img_bgr):
    """Find all handwriting rectangles, sorted left-col top->bottom then right."""
    h, w  = img_bgr.shape[:2]
    total = h * w
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=10
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(adaptive, otsu)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area   = bw * bh
        aspect = bw / bh if bh else 0
        if (0.004 * total < area < 0.20 * total) and (1.4 < aspect < 7.0):
            candidates.append((x, y, bw, bh))

    candidates = _merge_rects(candidates)
    mid   = w / 2
    left  = sorted([r for r in candidates if r[0]+r[2]/2 <  mid], key=lambda r: r[1])
    right = sorted([r for r in candidates if r[0]+r[2]/2 >= mid], key=lambda r: r[1])
    return left + right


# ── OCR: read ID above the box ────────────────────────────────────────────────

def _ocr_number(gray_roi, valid_ids=None):
    """Try multiple OCR passes and return the highest-confidence numeric candidate."""
    expected_len = None
    if valid_ids:
        lengths = [len(str(v)) for v in valid_ids]
        if lengths:
            expected_len = max(set(lengths), key=lengths.count)

    candidate_scores = {}

    for scale in [2, 3, 4]:
        up = cv2.resize(gray_roi, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_inv = cv2.bitwise_not(th)

        for variant in [th, th_inv]:
            padded = cv2.copyMakeBorder(variant, 20, 20, 20, 20,
                                        cv2.BORDER_CONSTANT, value=255)

            # Light denoise to stabilize OCR segmentation.
            padded = cv2.medianBlur(padded, 3)

            for psm in [7, 6, 8, 13]:
                cfg = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789'
                try:
                    data = pytesseract.image_to_data(
                        padded,
                        config=cfg,
                        output_type=pytesseract.Output.DICT,
                    )

                    tokens = []
                    for i, txt in enumerate(data['text']):
                        digits = re.sub(r'\D', '', txt or '')
                        if not digits:
                            continue
                        try:
                            conf = float(data['conf'][i])
                        except (ValueError, TypeError):
                            conf = -1.0
                        if conf < 0:
                            continue
                        try:
                            num = int(digits)
                        except ValueError:
                            continue

                        # Longer digit groups and higher confidence are preferred.
                        score = conf + (len(digits) * 12)
                        if expected_len is not None:
                            score -= abs(len(digits) - expected_len) * 20
                        candidate_scores[num] = max(candidate_scores.get(num, -1e9), score)
                        tokens.append((int(data['left'][i]), digits, conf))

                    # Combine left-to-right digit tokens into one line candidate (e.g. 4|2|9|0).
                    if tokens:
                        tokens.sort(key=lambda t: t[0])
                        joined = ''.join(t[1] for t in tokens)
                        if joined:
                            joined_num = int(joined)
                            avg_conf = sum(t[2] for t in tokens) / len(tokens)
                            joined_score = avg_conf + (len(joined) * 14)
                            if expected_len is not None:
                                joined_score -= abs(len(joined) - expected_len) * 20
                            candidate_scores[joined_num] = max(
                                candidate_scores.get(joined_num, -1e9),
                                joined_score,
                            )

                except Exception:
                    pass

    if not candidate_scores:
        return None

    # If label IDs are known, prefer OCR outputs that exist in labels.
    if valid_ids:
        for num in list(candidate_scores.keys()):
            if num in valid_ids:
                candidate_scores[num] += 12

    # Final safety: if we know expected length, ignore much shorter results.
    if expected_len is not None:
        filtered = {
            num: score for num, score in candidate_scores.items()
            if len(str(num)) >= max(1, expected_len - 1)
        }
        if filtered:
            candidate_scores = filtered

    return max(candidate_scores.items(), key=lambda kv: kv[1])[0]


def read_id_above_box(img_bgr, box, valid_ids=None):
    """OCR the number printed above-left of the box."""
    if not OCR_AVAILABLE:
        return None
    x, y, w, h   = box
    ih, iw        = img_bgr.shape[:2]
    # Use a wider OCR window above the box to avoid clipping digits.
    x1 = max(0,  x - 10)
    x2 = min(iw, x + w)
    y1 = max(0,  y - int(h * 1.2))
    y2 = max(0,  y - BORDER_INSET)
    if y2 <= y1 or x2 <= x1:
        return None

    roi  = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Try left-third first (where the ID number sits), then full strip.
    left = gray[:, : max(1, gray.shape[1] // 3)]
    return _ocr_number(left, valid_ids=valid_ids) or _ocr_number(gray, valid_ids=valid_ids)


# ── Crop + resize ─────────────────────────────────────────────────────────────

def crop_box(img_bgr, box):
    """Crop box with border removed, then tightly crop around detected drawing."""
    x, y, w, h = box
    ih, iw = img_bgr.shape[:2]
    
    # First, remove the printed border
    x1 = max(0,  x + BORDER_INSET)
    y1 = max(0,  y + BORDER_INSET)
    x2 = min(iw, x + w - BORDER_INSET)
    y2 = min(ih, y + h - BORDER_INSET)
    crop = img_bgr[y1:y2, x1:x2]
    
    # Add additional inset to skip any remaining border pixels
    extra_inset = 20
    eh, ew = crop.shape[:2]
    ex1 = min(extra_inset, ew // 3)
    ey1 = min(extra_inset, eh // 3)
    ex2 = max(ew - extra_inset, ew * 2 // 3)
    ey2 = max(eh - extra_inset, eh * 2 // 3)
    inner_crop = crop[ey1:ey2, ex1:ex2]
    
    # Convert to grayscale and detect non-white content
    gray = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find dark pixels (handwriting)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the drawing content
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find bounding rectangle of all drawing content
        all_points = np.vstack(contours)
        content_x, content_y, content_w, content_h = cv2.boundingRect(all_points)
        
        # Add small margin (2px) to keep some breathing room
        margin = 2
        cx1 = max(0, content_x - margin)
        cy1 = max(0, content_y - margin)
        cx2 = min(inner_crop.shape[1], content_x + content_w + margin)
        cy2 = min(inner_crop.shape[0], content_y + content_h + margin)
        
        # Map back to original crop coordinates
        cx1 += ex1
        cy1 += ey1
        cx2 += ex1
        cy2 += ey1
        
        # Ensure minimum size
        if cx2 - cx1 > 10 and cy2 - cy1 > 10:
            tight_crop = crop[cy1:cy2, cx1:cx2]
        else:
            tight_crop = crop
    else:
        # No content detected, use full crop
        tight_crop = crop
    
    return cv2.resize(tight_crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)


# ── Per-image processing ──────────────────────────────────────────────────────

def process_image(img_path, output_dir, writer_id, labels,
                  dataset_csv, debug=False):
    """
    Process one scanned image. Appends to dataset_csv immediately after
    each image so partial runs are not lost.
    Returns number of crops saved.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ✗ Cannot read: {img_path.name}")
        return 0

    print(f"\n── {img_path.name}  (writer_id={writer_id})")
    boxes = find_boxes(img)
    print(f"   Found {len(boxes)} box(es)")
    if not boxes:
        print("   ⚠  No boxes detected — skipping.")
        return 0

    # Get current auto-increment base before we start writing
    auto_id   = next_auto_id(dataset_csv)
    saved     = 0
    new_rows  = []
    debug_img = img.copy() if debug else None

    valid_ids = set(labels.keys()) if labels else None

    for i, box in enumerate(boxes):
        word_id = read_id_above_box(img, box, valid_ids=valid_ids)
        if word_id is None:
            print(f"   ⚠  box {i+1}: OCR could not read ID — skipping.")
            continue

        label = labels.get(word_id, '')
        if not str(label).strip():
            print(f"   ⚠  box {i+1}: word_id={word_id} has empty label — skipping.")
            continue

        resized  = crop_box(img, box)
        out_name = f"{word_id}_{writer_id}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), resized)

        row   = {
            'id':         auto_id,
            'image_path': str(out_path),
            'writer_id':  writer_id,
            'label':      label,
        }
        new_rows.append(row)
        auto_id += 1
        saved   += 1

        print(f"   ✓  box {i+1}: word_id={word_id}  label=\"{label}\"  -> {out_name}")

        if debug:
            x, y, w, h = box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 200, 0), 3)
            cv2.putText(debug_img, str(word_id), (x+6, y+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)

    # Append this image's rows immediately
    if new_rows:
        append_rows(dataset_csv, new_rows)

    if debug and debug_img is not None:
        dbg_path = output_dir / f"DEBUG_{img_path.stem}.jpg"
        cv2.imwrite(str(dbg_path), debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"   Debug -> {dbg_path.name}")

    return saved


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f'Crop handwriting boxes ({OUT_W}x{OUT_H}px) and build dataset CSV.'
    )
    parser.add_argument('--images-dir',      default='./image_processing/images',
                        help='Folder with scanned images')
    parser.add_argument('--output-dir',      default='./image_processing/processed_images',
                        help='Where to save cropped images')
    parser.add_argument('--labels-csv',      default='./pdfs_to_print/word_labels.csv',
                        help='word_labels.csv from generate_pdf.py')
    parser.add_argument('--dataset-csv',     default='./image_processing/dataset.csv',
                        help='Dataset CSV to create or append to')
    parser.add_argument('--writer-id',       default=None, type=int,
                        help='Fixed writer_id to use for all images (overrides --writer-id-start)')
    parser.add_argument('--writer-id-start', default=1, type=int,
                        help='First writer_id to assign (increments per image file)')
    parser.add_argument('--debug',           action='store_true',
                        help='Save annotated debug image per page')
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    images_dir  = Path(args.images_dir)
    dataset_csv = Path(args.dataset_csv)
    manifest_csv = output_dir / PROCESSED_MANIFEST
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels (gracefully handle missing file)
    labels      = {}
    labels_path = Path(args.labels_csv)
    if labels_path.exists():
        labels = load_labels(labels_path)
        print(f"✓ Loaded {len(labels)} labels from {labels_path.name}")
    else:
        print(f"⚠  word_labels.csv not found at {labels_path} — label column will be blank")

    # Find images
    exts   = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    images = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in exts and not f.name.startswith('DEBUG_')
    ])
    if not images:
        print(f"✗ No images found in {images_dir}")
        return

    processed_sources = load_processed_sources(manifest_csv)
    if processed_sources:
        print(f"  Skip list  : {len(processed_sources)} already-processed image(s)")

    print(f"✓ {len(images)} image(s) found")
    print(f"  Output size : {OUT_W}x{OUT_H}px")
    print(f"  Dataset CSV : {dataset_csv}  (appending)")
    if args.writer_id is not None:
        print(f"  writer_id   : {args.writer_id} (fixed for all images)")
    else:
        print(f"  writer_id   : {args.writer_id_start} .. {args.writer_id_start + len(images) - 1}")

    total_saved = 0
    for i, img_path in enumerate(images):
        if img_path.name in processed_sources:
            print(f"\n── {img_path.name}")
            print("   ✓ already processed — skipping.")
            continue

        if args.writer_id is not None:
            writer_id = args.writer_id
        else:
            writer_id = args.writer_id_start + i
        saved = process_image(
            img_path, output_dir,
            writer_id   = writer_id,
            labels      = labels,
            dataset_csv = dataset_csv,
            debug       = args.debug,
        )
        total_saved += saved
        if saved > 0:
            append_processed_source(manifest_csv, img_path.name)
            processed_sources.add(img_path.name)

    print(f"\n{'='*52}")
    print(f"  Done. {total_saved} crops saved.")
    print(f"  Images  -> {output_dir}/")
    print(f"  Dataset -> {dataset_csv}")
    print(f"{'='*52}")
    print()
    print("To add more writers later:")
    print("  python extract_boxes.py --writer-id-start 3")
    print("  (id column auto-continues from last row in dataset.csv)")

if __name__ == '__main__':
    main()