"""
PDF to JPG Converter
====================
Converts each page of one or more PDF files to a JPG image.

Uses pymupdf (fitz) as the primary engine — no system dependencies needed.
Falls back to pdf2image (poppler) if pymupdf is not installed.

Output naming (default batch mode):
    {writer_number}_page{page_number}.jpg
    e.g. 1_page1.jpg, 1_page2.jpg, 2_page1.jpg ...

    Each PDF in the folder gets its own writer_number (auto-incrementing).
    Use --writer-start to continue numbering across runs.

Single-file explicit mode (pass a specific .pdf path):
    Naming still follows writer_number scheme; writer_number defaults to
    --writer-start value (default 1).

Usage:
    # Convert all PDFs in scanned_pdfs/ folder (recommended)
    python pdf_to_jpg.py

    # Continue numbering from writer 5
    python pdf_to_jpg.py --writer-start 5

    # Custom folders
    python pdf_to_jpg.py --input-dir ./scanned_pdfs --output-dir ./images

    # Convert specific files (writer numbers start at --writer-start)
    python pdf_to_jpg.py --files doc1.pdf doc2.pdf

    # Legacy: higher DPI or quality
    python pdf_to_jpg.py --dpi 200 --quality 85

Requirements (pick one):
    pip install pymupdf            <- recommended, no system deps
    pip install pdf2image pillow   <- also needs: apt install poppler-utils
"""

import argparse
import sys
from pathlib import Path


# ── Conversion engines ────────────────────────────────────────────────────────

def convert_with_pymupdf(pdf_path: Path, output_dir: Path,
                         dpi: int, quality: int, writer_number: int):
    import fitz  # pymupdf
    doc   = fitz.open(str(pdf_path))
    n     = len(doc)
    saved = []
    mat   = fitz.Matrix(dpi / 72, dpi / 72)   # 72 pt/inch is PDF native

    for i, page in enumerate(doc):
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        out_name = f"{writer_number}_page{i + 1}.jpg"
        out_path = output_dir / out_name
        pix.save(str(out_path))                # pymupdf writes JPEG directly
        saved.append(out_path)
        print(f"  ✓  page {i+1}/{n}  ->  {out_name}")

    doc.close()
    return saved


def convert_with_pdf2image(pdf_path: Path, output_dir: Path,
                           dpi: int, quality: int, writer_number: int):
    from pdf2image import convert_from_path
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    n     = len(pages)
    saved = []

    for i, page in enumerate(pages):
        out_name = f"{writer_number}_page{i + 1}.jpg"
        out_path = output_dir / out_name
        page.save(str(out_path), "JPEG", quality=quality)
        saved.append(out_path)
        print(f"  ✓  page {i+1}/{n}  ->  {out_name}")

    return saved


def convert_pdf(pdf_path: Path, output_dir: Path,
                dpi: int, quality: int, writer_number: int):
    print(f"\n── {pdf_path.name}  (writer={writer_number}, {dpi} DPI)")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try pymupdf first (no system deps), then pdf2image
    try:
        import fitz
        return convert_with_pymupdf(pdf_path, output_dir, dpi, quality, writer_number)
    except ImportError:
        pass
    except Exception as e:
        print(f"  ✗ pymupdf error: {e}")
        sys.exit(1)

    try:
        from pdf2image import convert_from_path
        return convert_with_pdf2image(pdf_path, output_dir, dpi, quality, writer_number)
    except ImportError:
        print("  ✗ No PDF engine found. Install one:")
        print("      pip install pymupdf")
        print("      -- or --")
        print("      pip install pdf2image pillow && apt install poppler-utils")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ pdf2image error: {e}")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to JPG images named {writer_number}_page{n}.jpg"
    )

    # Source selection — mutually exclusive
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--input-dir", "-i",
        default="./image_processing/scanned_pdfs",
        help="Folder of PDFs to batch-convert (default: ./image_processing/scanned_pdfs)",
    )
    src.add_argument(
        "--files", "-f", nargs="+",
        help="Explicit PDF file(s) to convert instead of scanning a folder",
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="./image_processing/images",
        help="Output folder for JPGs (default: ./image_processing/images)",
    )
    parser.add_argument(
        "--writer-start", type=int, default=1,
        help="Writer number for the first PDF; increments per PDF (default: 1)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Render resolution (default: 300)",
    )
    parser.add_argument(
        "--quality", type=int, default=90,
        help="JPEG quality 1-95 (default: 90; pymupdf ignores this)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ── Collect PDFs ──────────────────────────────────────────────────────────
    if args.files:
        pdf_list = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                print(f"⚠  Not found: {p}")
            elif p.suffix.lower() != ".pdf":
                print(f"⚠  Skipping non-PDF: {p.name}")
            else:
                pdf_list.append(p)
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"✗ Input folder not found: {input_dir}")
            sys.exit(1)
        pdf_list = sorted(input_dir.glob("*.pdf"))
        if not pdf_list:
            print(f"✗ No PDF files found in {input_dir}")
            sys.exit(1)

    print(f"✓ Found {len(pdf_list)} PDF(s)")
    print(f"  Output dir  : {output_dir}")
    print(f"  Writer range: {args.writer_start} .. {args.writer_start + len(pdf_list) - 1}")
    print(f"  Resolution  : {args.dpi} DPI")

    # ── Convert ───────────────────────────────────────────────────────────────
    all_outputs = []
    for offset, pdf_path in enumerate(pdf_list):
        writer_number = args.writer_start + offset
        all_outputs += convert_pdf(
            pdf_path, output_dir,
            dpi=args.dpi, quality=args.quality,
            writer_number=writer_number,
        )

    print(f"\n{'='*50}")
    print(f"  Done. {len(all_outputs)} page(s) saved to {output_dir}/")
    print(f"{'='*50}")
    print()
    print("To add more writers later:")
    print(f"  python pdf_to_jpg.py --writer-start {args.writer_start + len(pdf_list)}")


if __name__ == "__main__":
    main()