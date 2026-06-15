import argparse
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
import torchvision.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated images against real images.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images.")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images.")
    parser.add_argument("--resize", type=str, default=None, help="Resize to HxW (e.g. 384x960).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mps.")
    parser.add_argument("--text_pairs", type=str, default=None, help="CSV with ref and hyp text (optional).")
    return parser.parse_args()


def parse_resize(value: str | None) -> Tuple[int, int] | None:
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--resize must be like HxW, e.g. 384x960")
    return int(parts[0]), int(parts[1])


def list_images(root: Path) -> Dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            files[p.stem] = p
    return files


def load_image(path: Path, resize_hw: Tuple[int, int] | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]))
    return T.ToTensor()(img)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def edit_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    # Levenshtein distance with counts for substitutions, insertions, deletions.
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = len(ref), len(hyp)
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            if i > 0 and j > 0 and ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1

    return subs, ins, dels


def compute_wer_cer(pairs_path: Path) -> Tuple[float, float]:
    with pairs_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "ref" not in reader.fieldnames or "hyp" not in reader.fieldnames:
            raise SystemExit("text_pairs CSV must have ref and hyp columns.")

        total_word = total_word_err = 0
        total_char = total_char_err = 0

        for row in reader:
            ref_text = row["ref"].strip()
            hyp_text = row["hyp"].strip()

            ref_words = ref_text.split()
            hyp_words = hyp_text.split()
            s, i, d = edit_distance(ref_words, hyp_words)
            total_word_err += (s + i + d)
            total_word += max(1, len(ref_words))

            ref_chars = list(ref_text)
            hyp_chars = list(hyp_text)
            s, i, d = edit_distance(ref_chars, hyp_chars)
            total_char_err += (s + i + d)
            total_char += max(1, len(ref_chars))

    wer = 100.0 * total_word_err / total_word
    cer = 100.0 * total_char_err / total_char
    return wer, cer


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    resize_hw = parse_resize(args.resize)

    real_dir = Path(args.real_dir)
    gen_dir = Path(args.gen_dir)
    real_files = list_images(real_dir)
    gen_files = list_images(gen_dir)

    common = sorted(set(real_files.keys()) & set(gen_files.keys()))
    if not common:
        raise SystemExit("No matching filenames between real_dir and gen_dir.")

    # Optional metrics
    fid = None
    ssim = None
    lpips_fn = None

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048).to(device)
    except Exception:
        print("FID unavailable (install torchmetrics).")

    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        print("SSIM unavailable (install torchmetrics).")

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
    except Exception:
        print("LPIPS unavailable (install lpips).")

    rmse_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []

    for stem in common:
        real = load_image(real_files[stem], resize_hw).to(device)
        fake = load_image(gen_files[stem], resize_hw).to(device)

        if real.shape != fake.shape:
            raise SystemExit(f"Shape mismatch for {stem}: {real.shape} vs {fake.shape}")

        rmse_vals.append(rmse(real, fake))

        if ssim is not None:
            ssim_vals.append(ssim(fake.unsqueeze(0), real.unsqueeze(0)).item())

        if lpips_fn is not None:
            # LPIPS expects [-1, 1]
            lpips_vals.append(lpips_fn(fake.unsqueeze(0) * 2 - 1, real.unsqueeze(0) * 2 - 1).item())

        if fid is not None:
            # FID expects uint8 0-255
            fid.update((real * 255).byte().unsqueeze(0), real=True)
            fid.update((fake * 255).byte().unsqueeze(0), real=False)

    print(f"RMSE: {sum(rmse_vals) / len(rmse_vals):.6f}")
    if ssim_vals:
        print(f"SSIM: {sum(ssim_vals) / len(ssim_vals):.6f}")
    if lpips_vals:
        print(f"LPIPS: {sum(lpips_vals) / len(lpips_vals):.6f}")
    if fid is not None:
        print(f"FID: {fid.compute().item():.6f}")

    if args.text_pairs:
        wer, cer = compute_wer_cer(Path(args.text_pairs))
        print(f"WER: {wer:.2f}%")
        print(f"CER: {cer:.2f}%")


if __name__ == "__main__":
    main()
