import os
import sys
import numpy as np
from os import urandom
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264
import simon3264 as simon3264

cipher_dict = {
    "speck3264":speck3264,
    "simon3264":simon3264
}

word_size_to_data_type = {
    16: np.uint16,
    32: np.uint32,
    64: np.uint64
}


def diff_distribution_generate(cipher, diff, nr, n, label=1, single_key=0):
    """Generate the ciphertext differential distribution."""
    if single_key:
        keys = np.frombuffer(
            urandom(4 * (cipher.WORD_SIZE() // 8)),
            dtype=word_size_to_data_type[cipher.WORD_SIZE()]
        ).reshape(4, -1)
    else:
        keys = np.frombuffer(
            urandom(4 * (cipher.WORD_SIZE() // 8) * n),
            dtype=word_size_to_data_type[cipher.WORD_SIZE()]
        ).reshape(4, -1)

    plain0l = np.frombuffer(
        urandom((cipher.WORD_SIZE() // 8) * n),
        dtype=word_size_to_data_type[cipher.WORD_SIZE()]
    )
    plain0r = np.frombuffer(
        urandom((cipher.WORD_SIZE() // 8) * n),
        dtype=word_size_to_data_type[cipher.WORD_SIZE()]
    )

    if label == 1:
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
    else:
        plain1l = np.frombuffer(
            urandom((cipher.WORD_SIZE() // 8) * n),
            dtype=word_size_to_data_type[cipher.WORD_SIZE()]
        )
        plain1r = np.frombuffer(
            urandom((cipher.WORD_SIZE() // 8) * n),
            dtype=word_size_to_data_type[cipher.WORD_SIZE()]
        )

    ks = cipher.expand_key(keys, nr)
    ctdata0l, ctdata0r = cipher.encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = cipher.encrypt((plain1l, plain1r), ks)

    # "ctfiff_l" kept for downstream compatibility
    return pd.DataFrame({
        'ctfiff_l': ctdata0l ^ ctdata1l,
        'ctdiff_r': ctdata0r ^ ctdata1r
    })


def format_diff(diff):
    """Return concatenated 4‑hex‑digit strings of the differential words."""
    return f"{diff[0]:04x}{diff[1]:04x}"


# ========================= HEATMAP (FIXED SCALE) ========================= #

def plot_frequency_heatmap(filename: str, ctdiff: pd.DataFrame, bins: int = 128):
    """Plot a 2‑D heat‑map of **relative frequencies** with a **fixed colour scale**.

    The colour scale is logarithmic, spanning from 10^-7 (vmin) to 1 (vmax)
    so that results from different experiments remain directly comparable.
    """
    # Extract differential words
    x_vals = ctdiff['ctfiff_l'].values
    y_vals = ctdiff['ctdiff_r'].values

    # Count occurrences of each differential pair
    counts = Counter(zip(x_vals, y_vals))
    pairs, pair_counts = zip(*counts.items())

    total_pairs = sum(pair_counts)
    rel_freqs = [c / total_pairs for c in pair_counts]

    x_coords = [p[0] for p in pairs]
    y_coords = [p[1] for p in pairs]

    # 2‑D histogram weighted by relative frequency (probability)
    heatmap, xedges, yedges = np.histogram2d(
        x_coords, y_coords,
        bins=bins,
        range=[[0, 65535], [0, 65535]],
        weights=rel_freqs
    )

    # Replace zeros with NaN to avoid log10(0) issues
    heatmap = np.where(heatmap == 0, np.nan, heatmap)

    # Fixed logarithmic colour scale: 1e‑7 → 1
    vmin, vmax = 1e-7, 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        heatmap.T,
        origin='lower',
        cmap='YlGnBu',
        extent=[0, 65535, 0, 65535],
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )

    # Add colour‑bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Relative Frequency", fontsize=14)
    cb.set_ticks([1e-7, 1e-5, 1e-3, 1])
    cb.set_ticklabels([r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$", r"$1$"])

    # Axis formatting
    ax.set_xlabel("Left word of ciphertext difference", fontsize=14)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_ylabel("Right word of ciphertext difference", fontsize=14)
    ax.invert_yaxis()

    ticks = np.linspace(0, 65535, 16)
    hex_ticks = [f"{int(t):x}" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(hex_ticks, rotation=45, fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(hex_ticks, fontsize=12)

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"The plot has been saved at {filename}")


# ========================= MAIN DRIVER ========================= #

def cluster_analysis(cipher_name, diff, nr, n, label=1, single_key=0):
    diff_str = format_diff(diff)
    filename = f"./visual_result_relative_frequency/{cipher_name}_{diff_str}_nr{nr}_label{label}_single_key{single_key}.pdf"
    ctdiff = diff_distribution_generate(
        cipher_dict[cipher_name], diff, nr, n, label, single_key
    )
    plot_frequency_heatmap(filename, ctdiff)


if __name__ == '__main__':
    # Example usage
    cluster_analysis('speck3264', (0x0040, 0x0000), 3, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0040, 0x0000), 4, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0040, 0x0000), 5, 10 ** 7, label=1, single_key=0)

    cluster_analysis('speck3264', (0x0010, 0x2000), 3, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0010, 0x2000), 4, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0010, 0x2000), 5, 10 ** 7, label=1, single_key=0)

    cluster_analysis('speck3264', (0x0010, 0x0000), 3, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0010, 0x0000), 4, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0010, 0x0000), 5, 10 ** 7, label=1, single_key=0)

    cluster_analysis('speck3264', (0x0002, 0x0400), 3, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0002, 0x0400), 4, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0002, 0x0400), 5, 10 ** 7, label=1, single_key=0)

    cluster_analysis('speck3264', (0x0000, 0x0080), 3, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0000, 0x0080), 4, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0000, 0x0080), 5, 10 ** 7, label=1, single_key=0)

    cluster_analysis('speck3264', (0x0001, 0x8000), 3, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0001, 0x8000), 4, 10 ** 7, label=1, single_key=0)
    cluster_analysis('speck3264', (0x0001, 0x8000), 5, 10 ** 7, label=1, single_key=0)
