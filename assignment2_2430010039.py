"""
Assignment 2 - Linear Smoothing vs Non-Linear Rank Filters on Noisy Images
=========================================================================

Objective
---------
Compare a LINEAR smoothing filter (3x3 Box Average, cv2.blur) against a
NON-LINEAR rank/order-statistic filter (3x3 Median, cv2.medianBlur) on a
grayscale image corrupted with salt-and-pepper (impulse) noise.

Pipeline
--------
    1. Read a grayscale image.
    2. Add salt-and-pepper noise (random pixels forced to 0 or 255).
    3. Filter the noisy image with:
           - cv2.blur        -> 3x3 Box Average (linear, mean of the window)
           - cv2.medianBlur  -> 3x3 Median      (non-linear, middle rank)
    4. Show original / noisy / box / median side by side and save the figure.
    5. Report MSE and PSNR for both filters against the clean original.

Run
---
    python assignment2_RollNumber.py                       # uses built-in test image
    python assignment2_RollNumber.py path/to/image.jpg     # uses your own image
    python assignment2_RollNumber.py image.jpg 0.05        # custom noise density (5%)

Author : Pratyush
Roll No: <ROLL_NUMBER>
"""

import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless backend: render to file, no GUI needed
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# 0. Utility functions
# ----------------------------------------------------------------------------
def make_test_image(size=512):
    """
    Build a synthetic 512x512 grayscale test image.

    It deliberately contains flat regions, sharp straight edges, fine
    checkerboard detail and smooth gradients. That mix matters: the whole
    point of this experiment is to see how each filter treats an EDGE versus
    how it treats ISOLATED NOISE PIXELS, and you can only judge that if the
    image actually has crisp edges to preserve.
    """
    img = np.full((size, size), 90, dtype=np.uint8)     # mid-dark background

    # Bright rectangle -> two long, perfectly straight edges
    cv2.rectangle(img, (60, 60), (250, 250), 200, thickness=-1)

    # Filled circle -> a curved (diagonal-ish) edge
    cv2.circle(img, (370, 160), 90, 40, thickness=-1)

    # Checkerboard -> high-frequency texture; a blur will smear this badly
    tile = 16
    for y in range(280, 400, tile):
        for x in range(60, 220, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y:y + tile, x:x + tile] = 240

    # Smooth horizontal gradient -> low-frequency region, both filters behave
    ramp = np.linspace(60, 220, size, dtype=np.uint8)
    img[420:480, :] = np.tile(ramp, (60, 1))

    # Thin dark line -> a one-pixel-wide feature; median preserves it, box erodes it
    cv2.line(img, (300, 280), (300, 470), 0, thickness=2)
    return img


def add_salt_and_pepper(image, density=0.05):
    """
    Add salt-and-pepper (impulse) noise to a grayscale image.

    `density` is the FRACTION OF ALL PIXELS that get corrupted, split evenly
    between the two impulses:
        * 'pepper' -> pixel forced to 0   (black)
        * 'salt'   -> pixel forced to 255 (white)

    This is the key property of impulse noise: a corrupted pixel is pushed to
    the EXTREME of the dynamic range, far away from its true value and from
    its neighbours. Keep that in mind when reading the results below.

    Returns
    -------
    noisy : np.ndarray
        The corrupted image (uint8).
    mask  : np.ndarray
        Boolean map of corrupted pixel locations, useful for analysis/debug.
    """
    noisy = image.copy()
    h, w = image.shape
    total = h * w

    n_corrupt = int(density * total)
    n_pepper = n_corrupt // 2
    n_salt = n_corrupt - n_pepper

    # np.random.choice on a flat index avoids duplicate (y, x) draws
    coords = np.random.choice(total, size=n_corrupt, replace=False)
    ys, xs = np.unravel_index(coords, (h, w))

    mask = np.zeros((h, w), dtype=bool)
    mask[ys, xs] = True

    noisy[ys[:n_pepper], xs[:n_pepper]] = 0      # pepper (black)
    noisy[ys[n_pepper:], xs[n_pepper:]] = 255    # salt   (white)

    return noisy, mask


def mse(a, b):
    """Mean Squared Error between two images (lower is better)."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    """Peak Signal-to-Noise Ratio in dB (higher is better). 255 = 8-bit peak."""
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / m))


# ----------------------------------------------------------------------------
# 1. Load the input image
# ----------------------------------------------------------------------------
def load_gray(path=None):
    """Read an image as single-channel grayscale, or synthesise a test image."""
    if path and os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Could not decode image: {path}")
        return img, os.path.basename(path)
    print("[info] No valid input image supplied -> using built-in synthetic test image.")
    return make_test_image(), "synthetic_test_image"


def main():
    # Command-line: [image_path] [noise_density]
    path = sys.argv[1] if len(sys.argv) > 1 else None
    density = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05   # 5% impulse noise

    original, source_name = load_gray(path)
    print(f"[info] Image source      : {source_name}")
    print(f"[info] Image shape       : {original.shape} (grayscale, uint8)")
    print(f"[info] Noise density     : {density:.1%} of pixels")

    # ------------------------------------------------------------------
    # 2. Corrupt the image with salt-and-pepper noise
    # ------------------------------------------------------------------
    noisy, noise_mask = add_salt_and_pepper(original, density=density)
    print(f"[info] Corrupted pixels  : {noise_mask.sum()} "
          f"(~{noise_mask.mean():.1%})")

    # ------------------------------------------------------------------
    # 3. Apply the two 3x3 filters
    # ------------------------------------------------------------------
    # (a) LINEAR: 3x3 Box Average. Every pixel becomes the MEAN of the 3x3
    #     window. It is linear because the output is a weighted sum of inputs.
    #     A single salt or pepper pixel contributes 255/9 or 0/9 to the mean,
    #     so it drags the result of the whole 3x3 neighbourhood with it -- the
    #     impulse is not removed, it is SMEARED into a grey blotch.
    box_filtered = cv2.blur(noisy, (3, 3))

    # (b) NON-LINEAR: 3x3 Median. Every pixel becomes the MIDDLE value (rank 5
    #     of 9) of the sorted 3x3 window. It is a rank/order-statistic filter,
    #     so it is not a weighted sum and cannot be described by a convolution
    #     kernel. Because an impulse is an extreme value, it sits at one END of
    #     the sorted list; the middle element is unaffected as long as fewer
    #     than half the window is corrupted, so the impulse is DISCARDED and
    #     the true underlying value is restored.
    median_filtered = cv2.medianBlur(noisy, 3)   # ksize=3 -> 3x3 median

    # ------------------------------------------------------------------
    # 4. Quantitative comparison against the clean original
    # ------------------------------------------------------------------
    results = {
        "Noisy (unfiltered)": noisy,
        "3x3 Box Average (linear)": box_filtered,
        "3x3 Median (non-linear)": median_filtered,
    }
    print("\n[metrics] vs. clean original  (PSNR higher = better)")
    print(f"{'stage':<28}{'MSE':>12}{'PSNR (dB)':>12}")
    for name, img in results.items():
        print(f"{name:<28}{mse(original, img):>12.2f}{psnr(original, img):>12.2f}")

    # Edge preservation check: how much did each filter soften the image?
    # A high Laplacian variance means crisp edges survived; a low value means
    # the filter blurred them away. Computed on the filtered outputs only.
    lap_box = cv2.Laplacian(box_filtered, cv2.CV_64F).var()
    lap_med = cv2.Laplacian(median_filtered, cv2.CV_64F).var()
    lap_clean = cv2.Laplacian(original, cv2.CV_64F).var()
    print(f"\n[edges]   Laplacian variance (clean original = {lap_clean:.0f})")
    print(f"{'3x3 Box Average':<28}{lap_box:>12.0f}")
    print(f"{'3x3 Median':<28}{lap_med:>12.0f}")

    # ------------------------------------------------------------------
    # 5. Side-by-side display
    # ------------------------------------------------------------------
    panels = [
        ("Original (clean)", original),
        (f"Noisy\nsalt & pepper ({density:.0%})", noisy),
        ("3x3 Box Average\n(linear)", box_filtered),
        ("3x3 Median\n(non-linear rank)", median_filtered),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(
        "Salt-and-pepper noise removal: 3x3 Box Average vs 3x3 Median",
        fontsize=15, y=0.98,
    )
    fig.text(
        0.5, 0.015,
        "Median (rank) filter removes impulses while keeping edges sharp; "
        "Box Average smears impulses into blotches and blurs edges.",
        ha="center", fontsize=11, style="italic",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out_png = "comparison_plot.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[done] Saved comparison figure -> {out_png}")

    # ------------------------------------------------------------------
    # 6. Which filter performed better, and why (summary)
    # ------------------------------------------------------------------
    # VERDICT: the 3x3 MEDIAN filter wins decisively on salt-and-pepper noise.
    #
    # WHY:
    #   * Box Average is a LINEAR filter: it outputs a fixed weighted SUM of
    #     the window. An impulse at 0 or 255 is an outlier, and averaging is
    #     not robust to outliers -- the corrupted value is folded into the
    #     neighbourhood mean and spread over a 3x3 area as a grey blob. The
    #     noise energy is redistributed, not deleted, and simultaneously every
    #     genuine edge is averaged across, so edges go soft.
    #
    #   * Median is a NON-LINEAR RANK filter: it outputs the middle element of
    #     the sorted window. It makes no use of the magnitude of an outlier --
    #     only its position in the ordering. As long as fewer than half of the
    #     9 window pixels are impulses, the median equals one of the clean
    #     pixels, so the noise pixel is simply discarded. This is exactly why
    #     it is called an order-statistic / robust estimator.
    #
    #   * Edge preservation: on an edge the 3x3 window holds pixels from both
    #     sides, and the median falls on whichever side holds the majority --
    #     so the edge is reproduced almost intact, unlike the mean which
    #     interpolates a ramp across it.
    #
    # CAVEAT: the median's advantage holds while noise density is moderate
    # (roughly below ~40-50% corruption, where impulses stop being a minority
    # of any window). Above that, the median too begins to break down. The
    # printed PSNR/Laplacian numbers above quantify both claims for this run.
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    np.random.seed(42)   # reproducible noise pattern for the submitted figure
    main()
