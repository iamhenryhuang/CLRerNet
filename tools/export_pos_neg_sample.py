import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_lines_txt(lines_path: Path, w: int, h: int):
    lanes = []
    if not lines_path.exists():
        return lanes
    with lines_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            pts = []
            for i in range(0, len(vals), 2):
                if i + 1 >= len(vals):
                    break
                x = int(np.clip(vals[i], 0, w - 1))
                y = int(np.clip(vals[i + 1], 0, h - 1))
                pts.append([x, y])
            if len(pts) >= 2:
                lanes.append(np.array(pts, dtype=np.int32))
    return lanes


def make_lane_mask(h: int, w: int, lanes):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in lanes:
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=8)
    return mask


def apply_lane_enhance_positive(img_bgr_u8: np.ndarray, mask_hw: np.ndarray):
    """Mirror CLRerNet._apply_lane_enhance_positive for a single image.

    - x_unit in [0,1]
    - lane_alpha = max_pool(mask,15) then avg_pool(11)
    - enhanced = unsharp + 0.10
    - positive = x*(1-alpha) + enhanced*alpha
    """
    x = img_bgr_u8.astype(np.float32) / 255.0
    m = mask_hw.astype(np.float32)

    # max_pool2d(mask, 15, stride=1, padding=7)
    lane_alpha = cv2.dilate(m, np.ones((15, 15), dtype=np.uint8), iterations=1)

    # avg_pool2d(mask, 11, stride=1, padding=5) with zero padding
    k11 = np.ones((11, 11), dtype=np.float32) / 121.0
    lane_alpha = cv2.filter2D(lane_alpha, ddepth=-1, kernel=k11, borderType=cv2.BORDER_CONSTANT)
    lane_alpha = np.clip(lane_alpha, 0.0, 1.0)

    if float(lane_alpha.sum()) <= 0:
        positive = x
    else:
        # avg_pool2d(x, 5, stride=1, padding=2) with zero padding
        k5 = np.ones((5, 5), dtype=np.float32) / 25.0
        blur = np.stack(
            [
                cv2.filter2D(x[:, :, c], ddepth=-1, kernel=k5, borderType=cv2.BORDER_CONSTANT)
                for c in range(3)
            ],
            axis=2,
        )
        sharp = x + 0.8 * (x - blur)
        enhanced = np.clip(sharp + 0.10, 0.0, 1.0)
        alpha3 = lane_alpha[:, :, None]
        positive = x * (1.0 - alpha3) + enhanced * alpha3

    pos_u8 = np.clip(np.round(positive * 255.0), 0, 255).astype(np.uint8)
    lane_alpha_u8 = np.clip(np.round(lane_alpha * 255.0), 0, 255).astype(np.uint8)
    return pos_u8, lane_alpha_u8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="dataset/culane/driver_100_30frame/05251330_0404.MP4/01380.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--lines",
        default="",
        help="Path to .lines.txt (default: same basename as image)",
    )
    parser.add_argument(
        "--out-dir",
        default="debug_views/pos_neg_samples",
        help="Output directory",
    )
    args = parser.parse_args()

    root = Path.cwd()
    img_path = (root / args.image).resolve() if not os.path.isabs(args.image) else Path(args.image)
    if args.lines:
        lines_path = (root / args.lines).resolve() if not os.path.isabs(args.lines) else Path(args.lines)
    else:
        lines_path = img_path.with_suffix(".lines.txt")

    out_dir = (root / args.out_dir).resolve() if not os.path.isabs(args.out_dir) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    lanes = parse_lines_txt(lines_path, w, h)
    mask = make_lane_mask(h, w, lanes)

    # Positive sample (matches CLRerNet._apply_lane_enhance_positive)
    pos, lane_alpha = apply_lane_enhance_positive(img, mask)

    # Negative sample (matches detector: dilate + Telea inpaint)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dil = cv2.dilate(mask, kernel, iterations=1)
    neg = cv2.inpaint(img, mask_dil, 7, cv2.INPAINT_TELEA)

    cv2.imwrite(str(out_dir / "anchor.png"), img)
    cv2.imwrite(str(out_dir / "positive_lane_enhance.png"), pos)
    cv2.imwrite(str(out_dir / "negative_inpaint.png"), neg)
    cv2.imwrite(str(out_dir / "lane_mask.png"), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "lane_mask_dilated.png"), (mask_dil * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "lane_alpha_soft.png"), lane_alpha)

    print(f"Saved to: {out_dir}")
    print(f"Image: {img_path}")
    print(f"Lines: {lines_path}")
    print(f"Lanes found: {len(lanes)}")


if __name__ == "__main__":
    main()
