import os
import sys
import itertools
from tqdm import tqdm
import pickle
import concurrent.futures

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264
import simon3264 as simon3264

cipher_dict = {
    "speck3264":speck3264,
    "simon3264":simon3264,
}


def cacl_bias_score(cipher_name, n, nr, index, weights=[1, 2, 3], max_workers=16):
    cipher = cipher_dict[cipher_name]
    scores = {}
    block_size = cipher.WORD_SIZE() * 2
    total_combinations = sum(len(list(itertools.combinations(range(block_size), weight))) for weight in weights)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        futures = []
        with tqdm(total=total_combinations,
                  desc="Calculating {} Bias Scores for {}".format(cipher_name, str(index))) as pbar:
            for weight in weights:
                for bits in itertools.combinations(range(block_size), weight):
                    diff = (sum(1 << bit for bit in bits if bit < cipher.WORD_SIZE()),
                            sum(1 << (bit - cipher.WORD_SIZE()) for bit in bits if bit >= cipher.WORD_SIZE()))
                    future = executor.submit(cipher.cacl_bais_score, n, nr, diff)
                    futures.append((future, diff))

            for future, diff in futures:
                try:
                    score = future.result()
                    scores[diff] = score
                except Exception as e:
                    print(f"An error occurred for diff {diff}: {e}")
                pbar.update(1)

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    with open('{}_search_results/search_results/bias_{}r_hamWeightlower_{}.pkl'.format(cipher_name, nr, str(len(weights) + 1)), 'wb') as f:
        pickle.dump(sorted_scores, f)
    top_100_scores = list(sorted_scores.items())[:100]
    with open('{}_search_results/search_results/bias_{}r_hamWeightlower_{}_{}.txt'.format(cipher_name, nr, str(len(weights) + 1),
                                                                       str(index)), 'w') as txtfile:
        for diff, score in top_100_scores:
            diff_hex = (f'({diff[0]:#0{cipher.WORD_SIZE() // 4 + 2}x},{diff[1]:#0{cipher.WORD_SIZE() // 4 + 2}x})')
            txtfile.write(f'{diff_hex}  bias = {score}\n')


def cacl_bias_score_best(cipher_name, n, nr, index, weights=[1, 2, 3], max_workers=16):
    cipher = cipher_dict[cipher_name]
    scores = {}
    block_size = cipher.WORD_SIZE() * 2
    total_combinations = sum(len(list(itertools.combinations(range(block_size), weight))) for weight in weights)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        futures = []
        with tqdm(total=total_combinations,
                  desc="Calculating {} Bias Scores for {}".format(cipher_name, str(index))) as pbar:
            for weight in weights:
                for bits in itertools.combinations(range(block_size), weight):
                    diff = (sum(1 << bit for bit in bits if bit < cipher.WORD_SIZE()),
                            sum(1 << (bit - cipher.WORD_SIZE()) for bit in bits if bit >= cipher.WORD_SIZE()))
                    future = executor.submit(cipher.cacl_bais_score, n, nr, diff)
                    futures.append((future, diff))

            for future, diff in futures:
                try:
                    score = future.result()
                    scores[diff] = score
                except Exception as e:
                    print(f"An error occurred for diff {diff}: {e}")
                pbar.update(1)

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    with open('{}_search_results/refined_search_results/bias_{}r_hamWeightlower_{}_{}.pkl'.format(cipher_name, nr, str(len(weights) + 1),
                                                                       str(index)), 'wb') as f:
        pickle.dump(sorted_scores, f)
    top_100_scores = list(sorted_scores.items())[:100]
    with open('{}_search_results/refined_search_results/bias_{}r_hamWeightlower_{}_{}.txt'.format(cipher_name, nr, str(len(weights) + 1),
                                                                       str(index)), 'w') as txtfile:
        for diff, score in top_100_scores:
            diff_hex = (f'({diff[0]:#0{cipher.WORD_SIZE() // 4 + 2}x},{diff[1]:#0{cipher.WORD_SIZE() // 4 + 2}x})')
            txtfile.write(f'{diff_hex}  bias = {score}\n')

    best_scores = {}
    top_500_diffs = list(sorted_scores.keys())[:500]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        futures = []
        with tqdm(total=len(top_500_diffs),
                  desc="Calculating {} Best Bias Scores for {}".format(cipher_name, str(index))) as pbar:
            for diff in top_500_diffs:
                future = executor.submit(cipher.cacl_bais_score, 100 * n, nr, diff)
                futures.append((future, diff))

            for future, diff in futures:
                try:
                    score = future.result()
                    best_scores[diff] = score
                except Exception as e:
                    print(f"An error occurred for diff {diff}: {e}")
                pbar.update(1)

    best_sorted_scores = dict(sorted(best_scores.items(), key=lambda item: item[1], reverse=True))
    with open('{}_search_results/refined_search_results/best_bias_{}r_hamWeightlower_{}_{}.pkl'.format(cipher_name, nr, str(len(weights) + 1),
                                                                            str(index)), 'wb') as f:
        pickle.dump(best_sorted_scores, f)
    best_top_100_scores = list(best_sorted_scores.items())[:100]
    with open('{}_search_results/refined_search_results/best_bias_{}r_hamWeightlower_{}_{}.txt'.format(cipher_name, nr, str(len(weights) + 1),
                                                                            str(index)), 'w') as txtfile:
        for diff, score in best_top_100_scores:
            diff_hex = (f'({diff[0]:#0{cipher.WORD_SIZE() // 4 + 2}x},{diff[1]:#0{cipher.WORD_SIZE() // 4 + 2}x})')
            txtfile.write(f'{diff_hex}  bias = {score}\n')


def cacl_ddc_score(cipher_name, n, nr, index, weights=[1, 2, 3], max_workers=16):
    cipher = cipher_dict[cipher_name]
    scores = {}
    block_size = cipher.WORD_SIZE() * 2
    total_combinations = sum(len(list(itertools.combinations(range(block_size), weight))) for weight in weights)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        futures = []
        with tqdm(total=total_combinations,
                  desc="Calculating {} DDC Value for {}".format(cipher_name, str(index))) as pbar:
            for weight in weights:
                for bits in itertools.combinations(range(block_size), weight):
                    diff = (sum(1 << bit for bit in bits if bit < cipher.WORD_SIZE()),
                            sum(1 << (bit - cipher.WORD_SIZE()) for bit in bits if bit >= cipher.WORD_SIZE()))
                    future = executor.submit(cipher.cacl_ddc, n, nr, diff)
                    futures.append((future, diff))

            for future, diff in futures:
                try:
                    score = future.result()
                    scores[diff] = score
                except Exception as e:
                    print(f"An error occurred for diff {diff}: {e}")
                pbar.update(1)

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    with open('{}_search_results/search_results/ddc_{}r_hamWeightlower_{}.pkl'.format(cipher_name, nr, str(len(weights) + 1)),
              'wb') as f:
        pickle.dump(sorted_scores, f)
    top_100_scores = list(sorted_scores.items())[:100]
    with open('{}_search_results/search_results/ddc_{}r_hamWeightlower_{}_{}.txt'.format(cipher_name, nr, str(len(weights) + 1),
                                                                          str(index)), 'w') as txtfile:
        for diff, score in top_100_scores:
            diff_hex = (f'({diff[0]:#0{cipher.WORD_SIZE() // 4 + 2}x},{diff[1]:#0{cipher.WORD_SIZE() // 4 + 2}x})')
            txtfile.write(f'{diff_hex}  ddc = {score}\n')


def cacl_ddc_score_best(cipher_name, n, nr, index, weights=[1, 2, 3], max_workers=16):
    cipher = cipher_dict[cipher_name]
    scores = {}
    block_size = cipher.WORD_SIZE() * 2
    total_combinations = sum(len(list(itertools.combinations(range(block_size), weight))) for weight in weights)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        futures = []
        with tqdm(total=total_combinations,
                  desc="Calculating {} DDC Value for {}".format(cipher_name, str(index))) as pbar:
            for weight in weights:
                for bits in itertools.combinations(range(block_size), weight):
                    diff = (sum(1 << bit for bit in bits if bit < cipher.WORD_SIZE()),
                            sum(1 << (bit - cipher.WORD_SIZE()) for bit in bits if bit >= cipher.WORD_SIZE()))
                    future = executor.submit(cipher.cacl_ddc, n, nr, diff)
                    futures.append((future, diff))

            for future, diff in futures:
                try:
                    score = future.result()
                    scores[diff] = score
                except Exception as e:
                    print(f"An error occurred for diff {diff}: {e}")
                pbar.update(1)

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    with open('{}_search_results/refined_search_results/ddc_{}r_hamWeightlower_{}_{}.pkl'.format(cipher_name, nr, str(len(weights) + 1),
                                                                          str(index)), 'wb') as f:
        pickle.dump(sorted_scores, f)
    top_100_scores = list(sorted_scores.items())[:100]
    with open('{}_search_results/refined_search_results/ddc_{}r_hamWeightlower_{}_{}.txt'.format(cipher_name, nr, str(len(weights) + 1),
                                                                          str(index)), 'w') as txtfile:
        for diff, score in top_100_scores:
            diff_hex = (f'({diff[0]:#0{cipher.WORD_SIZE() // 4 + 2}x},{diff[1]:#0{cipher.WORD_SIZE() // 4 + 2}x})')
            txtfile.write(f'{diff_hex}  ddc = {score}\n')

    best_scores = {}
    top_500_diffs = list(sorted_scores.keys())[:500]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        from concurrent.futures import as_completed
        futures = []
        with tqdm(total=len(top_500_diffs),
                  desc="Calculating {} Best ddc Scores for {}".format(cipher_name, str(index))) as pbar:
            for diff in top_500_diffs:
                future = executor.submit(cipher.cacl_ddc, 100 * n, nr, diff)
                futures.append((future, diff))

            for future, diff in futures:
                try:
                    score = future.result()
                    best_scores[diff] = score
                except Exception as e:
                    print(f"An error occurred for diff {diff}: {e}")
                pbar.update(1)

    best_sorted_scores = dict(sorted(best_scores.items(), key=lambda item: item[1], reverse=False))
    with open('{}_search_results/refined_search_results/best_ddc_{}r_hamWeightlower_{}_{}.pkl'.format(cipher_name, nr, str(len(weights) + 1),
                                                                               str(index)), 'wb') as f:
        pickle.dump(best_sorted_scores, f)
    best_top_100_scores = list(best_sorted_scores.items())[:100]
    with open('{}_search_results/refined_search_results/best_ddc_{}r_hamWeightlower_{}_{}.txt'.format(cipher_name, nr, str(len(weights) + 1),
                                                                               str(index)), 'w') as txtfile:
        for diff, score in best_top_100_scores:
            diff_hex = (f'({diff[0]:#0{cipher.WORD_SIZE() // 4 + 2}x},{diff[1]:#0{cipher.WORD_SIZE() // 4 + 2}x})')
            txtfile.write(f'{diff_hex}  ddc = {score}\n')


if __name__ == '__main__':
    for index in range(10):
        cacl_bias_score('speck3264', 10 ** 6, 3, index, [1, 2, 3, 4, 5])