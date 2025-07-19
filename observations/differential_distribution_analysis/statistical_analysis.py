import os
import sys
import numpy as np
from os import urandom
import pandas as pd
from collections import Counter

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264
import simon3264 as simon3264

cipher_dict = {
    "speck3264":speck3264,
    "simon3264":simon3264
}


def cluster_analysis(n, nr, diff, random=False):
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2*n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    if random:
        plain1l = np.frombuffer(urandom(2*n), dtype=np.uint16)
        plain1r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    else:
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
    ks = cipher_dict['speck3264'].expand_key(keys, nr)
    ctdata0l, ctdata0r = cipher_dict['speck3264'].encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = cipher_dict['speck3264'].encrypt((plain1l, plain1r), ks)
    ctdiff = pd.DataFrame({'ctfiff_l': ctdata0l^ctdata1l, 'ctdiff_r' : ctdata0r^ctdata1r})
    combinations = list(zip(ctdiff['ctfiff_l'], ctdiff['ctdiff_r']))
    comb_counter = Counter(combinations)
    filtered_combinations_1 = [count for comb, count in comb_counter.items() if count > 1]
    total_count_1 = sum(filtered_combinations_1)
    print(f"Total count of differences {len(filtered_combinations_1)} with count > 1: {total_count_1}")
    print(f"Proportion of these differences: {(total_count_1/n):.2%}")
    filtered_combinations_2 = [count for comb, count in comb_counter.items() if count > 1]
    total_count_2 = sum(filtered_combinations_2)
    print(f"Total count of differences {len(filtered_combinations_2)} with count > 2: {total_count_2}")
    print(f"Proportion of these differences: {(total_count_2/n):.2%}")
    top_25_combinations = comb_counter.most_common(25)
    print("Top 25 most frequent combinations:")
    top_25_counts = 0
    for comb, count in top_25_combinations:
        print(f"Difference: ({comb[0]:#06x}, {comb[1]:#06x}), Count: {count}")
        top_25_counts += count
    print(f"Proportion of top 25 differences: {(top_25_counts/n):.2%}")


if __name__ == '__main__':
    print('(0x0040, 0x0000) 3r cluster analysis:')
    cluster_analysis(10 ** 7, 3, (0x0040, 0x0000))
    print('\n(0x0010, 0x2000) 3r cluster analysis:')
    cluster_analysis(10 ** 7, 3, (0x0010, 0x2000))
    print('\n(0x0010, 0x0000) 3r cluster analysis:')
    cluster_analysis(10 ** 7, 3, (0x0010, 0x0000))
    print('\n(0x0002, 0x0400) 3r cluster analysis:')
    cluster_analysis(10 ** 7, 3, (0x0002, 0x0400))
    print('\n(0x0000, 0x0080) 3r cluster analysis:')
    cluster_analysis(10 ** 7, 3, (0x0000, 0x0080))
    print('\n(0x0001, 0x8000) 3r cluster analysis:')
    cluster_analysis(10 ** 7, 3, (0x0001, 0x8000))
    print()
    print('\n(0x0040, 0x0000) 4r cluster analysis:')
    cluster_analysis(10 ** 7, 4, (0x0040, 0x0000))
    print('\n(0x0010, 0x2000) 4r cluster analysis:')
    cluster_analysis(10 ** 7, 4, (0x0010, 0x2000))
    print('\n(0x0010, 0x0000) 4r cluster analysis:')
    cluster_analysis(10 ** 7, 4, (0x0010, 0x0000))
    print('\n(0x0002, 0x0400) 4r cluster analysis:')
    cluster_analysis(10 ** 7, 4, (0x0002, 0x0400))
    print('\n(0x0000, 0x0080) 4r cluster analysis:')
    cluster_analysis(10 ** 7, 4, (0x0000, 0x0080))
    print('\n(0x0001, 0x8000) 4r cluster analysis:')
    cluster_analysis(10 ** 7, 4, (0x0001, 0x8000))
    print()
    print('\n(0x0040, 0x0000) 5r cluster analysis:')
    cluster_analysis(10 ** 7, 5, (0x0040, 0x0000))
    print('\n(0x0010, 0x2000) 5r cluster analysis:')
    cluster_analysis(10 ** 7, 5, (0x0010, 0x2000))
    print('\n(0x0010, 0x0000) 5r cluster analysis:')
    cluster_analysis(10 ** 7, 5, (0x0010, 0x0000))
    print('\n(0x0002, 0x0400) 5r cluster analysis:')
    cluster_analysis(10 ** 7, 5, (0x0002, 0x0400))
    print('\n(0x0000, 0x0080) 5r cluster analysis:')
    cluster_analysis(10 ** 7, 5, (0x0000, 0x0080))
    print('\n(0x0001, 0x8000) 5r cluster analysis:')
    cluster_analysis(10 ** 7, 5, (0x0001, 0x8000))

    # def generate_hamming_weight_one_tuples():
    #     tuples = []
    #     for i in range(16):
    #         left = 1 << i
    #         tuples.append((left, 0x0000))
    #
    #     for i in range(16):
    #         right = 1 << i
    #         tuples.append((0x0000, right))
    #
    #     return tuples
    #
    # for r in range(3, 7):
    #     for diff in generate_hamming_weight_one_tuples():
    #         print(f'(0x{hex(diff[0])[2:].zfill(4)}, 0x{hex(diff[1])[2:].zfill(4)}) {r}r cluster analysis:')
    #         cluster_analysis(10 ** 7, r, diff)
    #     print()
