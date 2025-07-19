import ctypes
import itertools
import json
import numpy as np
import pandas as pd
from scipy.special import entr
import multiprocessing as mp
from os import urandom


class SearchBestDiff:
    def __init__(self, config_path, coarse_search_mode, coarse_search_num, fine_search_mode, fine_search_num, nr, weights=None, max_workers=16):
        self.config = self._load_config(config_path)
        self.coarse_search_mode = coarse_search_mode
        self.coarse_search_num = coarse_search_num
        self.fine_search_mode = fine_search_mode
        self.fine_search_num = fine_search_num
        self.nr = nr
        if self.nr > self.config['encrypt_rounds']:
            raise ValueError('Number of rounds to search cannot exceed the total rounds of the encryption algorithm')
        self.weights = weights or [1, 2, 3]
        self.max_workers = max_workers
        self.block_size = self.config['ciphertext_size']
        self.data_type = {
            'uint8': np.uint8,
            'uint16': np.uint16,
            'uint32': np.uint32,
            'uint64': np.uint64
        }[self.config['data_type']]
        self.ctypes_type = {
            'uint8': ctypes.c_uint8,
            'uint16': ctypes.c_uint16,
            'uint32': ctypes.c_uint32,
            'uint64': ctypes.c_uint64
        }[self.config['data_type']]
        self.word_size = {
            'uint8': 8,
            'uint16': 16,
            'uint32': 32,
            'uint64': 64
        }[self.config['data_type']]
        self.cipher_path = self.config['saved_path']

    @staticmethod
    def _load_config(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_encrypt_function(self):
        """Load DLL in each process to avoid passing ctypes pointers across processes."""
        cipher_dll = ctypes.CDLL(self.cipher_path)
        encrypt_batch = getattr(cipher_dll, 'encrypt_batch')

        encrypt_batch.argtypes = (
            ctypes.POINTER(self.ctypes_type),  # plaintext
            ctypes.POINTER(self.ctypes_type),  # master_key
            ctypes.c_int,           # rounds
            ctypes.POINTER(self.ctypes_type),  # ciphertext
            ctypes.c_int            # batch_size
        )
        encrypt_batch.restype = None
        return cipher_dll, encrypt_batch

    def _worker_bias(self, diff, sample_size):
        """Load DLL in subprocess and calculate bias score."""
        try:
            cipher_dll, encrypt_batch = self._load_encrypt_function()
            score = self._calculate_bias_score(diff, sample_size, encrypt_batch)
            return diff, score
        finally:
            del cipher_dll

    def _worker_entropy(self, diff, sample_size):
        """Load DLL in subprocess and calculate entropy score."""
        try:
            cipher_dll, encrypt_batch = self._load_encrypt_function()
            score = self._calculate_entropy_score(diff, sample_size, encrypt_batch)
            return diff, score
        finally:
            del cipher_dll

    def _generate_data(self, diff, sample_size, encrypt_batch):
        """Generate data with sample size of sample_size."""
        master_keys = np.frombuffer(
            urandom(sample_size * (self.config['master_key_size'] // 8)), dtype=self.data_type
        ).reshape(sample_size, self.config['master_key_size'] // self.word_size)
        master_keys = np.require(master_keys, dtype=self.data_type, requirements=['C', 'A'])

        plaintext0 = np.frombuffer(
            urandom(sample_size * (self.config['plaintext_size'] // 8)), dtype=self.data_type
        ).reshape(sample_size, self.config['plaintext_size'] // self.word_size)
        plaintext0 = np.require(plaintext0, dtype=self.data_type, requirements=['C', 'A'])

        plaintext1 = np.bitwise_xor(plaintext0, np.array(diff, dtype=self.data_type))
        plaintext1 = np.require(plaintext1, dtype=self.data_type, requirements=['C', 'A'])

        ciphertext0 = np.zeros_like(plaintext0)
        ciphertext1 = np.zeros_like(plaintext1)

        self._encrypt(plaintext0, master_keys, ciphertext0, sample_size, encrypt_batch)
        self._encrypt(plaintext1, master_keys, ciphertext1, sample_size, encrypt_batch)

        return ciphertext0, ciphertext1

    def _encrypt(self, plaintext, keys, ciphertext, sample_size, encrypt_batch):
        encrypt_batch(
            plaintext.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
            keys.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
            self.nr,
            ciphertext.ctypes.data_as(ctypes.POINTER(self.ctypes_type)),
            sample_size
        )

    def _calculate_bias_score(self, diff, sample_size, encrypt_batch):
        ciphertext0, ciphertext1 = self._generate_data(diff, sample_size, encrypt_batch)
        diff_ct = [ciphertext0 ^ ciphertext1]

        d = self._convert_to_binary(np.concatenate(diff_ct, axis=1).transpose(), self.word_size).transpose()
        score = np.average(np.abs(0.5 - np.average(d, axis=0)), axis=0)
        return score

    def _calculate_entropy_score(self, diff, sample_size, encrypt_batch):
        ciphertext0, ciphertext1 = self._generate_data(diff, sample_size, encrypt_batch)

        d_combined = pd.DataFrame((ciphertext0 ^ ciphertext1).tolist())
        counts = d_combined.value_counts()
        probabilities = counts / counts.sum()
        entropy = np.sum(entr(probabilities))

        return entropy

    @staticmethod
    def _convert_to_binary(arr, word_size):
        x = np.zeros((len(arr) * word_size, len(arr[0])), dtype=np.uint8)
        for i in range(len(arr) * word_size):
            index = i // word_size
            offset = word_size - (i % word_size) - 1
            x[i] = (arr[index] >> offset) & 1
        x = x.transpose()
        return x

    def _generate_differentials(self):
        for weight in self.weights:
            for bits in itertools.combinations(range(self.block_size), weight):
                diff = tuple(
                    sum(1 << (bit - self.word_size * i)
                        for bit in bits if self.word_size * i <= bit < self.word_size * (i + 1))
                    for i in range(self.block_size // self.word_size)
                )
                yield diff

    def _process_in_parallel(self, worker_func, diffs, sample_size, task_name,):
        """Process tasks in parallel and return raw results without sorting."""
        with mp.Pool(processes=self.max_workers) as pool:
            futures = [
                pool.apply_async(worker_func, (diff, sample_size))
                for diff in diffs
            ]
            scores = {}
            total_tasks = len(futures)

            for i, future in enumerate(futures, start=1):
                diff, score = future.get()
                if score is not None:
                    scores[diff] = score
                if i % 1000 == 0 and task_name == 'coarse_search':
                    print('coarse_search progress: ' + str(i) + '/' + str(total_tasks))
                if i % 100 == 0 and task_name == 'fine_search':
                    print('fine_search progress: ' + str(i) + '/' + str(total_tasks))

        return scores

    def search_bias_scores(self):
        """Calculate bias scores, sort in descending order, and return top 1000 differentials."""
        raw_scores = self._process_in_parallel(self._worker_bias, self._generate_differentials(),
                                               self.coarse_search_num, 'coarse_search')
        sorted_scores = dict(sorted(raw_scores.items(), key=lambda item: item[1], reverse=True))

        formatted_scores = [(self.format_diff(diff), score) for diff, score in sorted_scores.items()]
        for diff, score in formatted_scores[:100]:
            print(diff, ' bias = ', '{:.5f}'.format(score))

        return list(sorted_scores.items())[:500]

    def search_best_bias_scores(self, diffs):
        """Finely calculate bias scores and sort in descending order."""
        raw_scores = self._process_in_parallel(self._worker_bias, diffs, self.fine_search_num, 'fine_search')
        sorted_scores = dict(sorted(raw_scores.items(), key=lambda item: item[1], reverse=True))
        formatted_scores = [(self.format_diff(diff), score) for diff, score in sorted_scores.items()]
        return formatted_scores[:100]

    def search_ddc_scores(self):
        """Calculate DDC scores, sort in descending order, and return top 1000 differentials."""
        raw_scores = self._process_in_parallel(self._worker_entropy, self._generate_differentials(),
                                               self.coarse_search_num, 'coarse_search')
        score_factor = np.log2(self.coarse_search_num)
        adjusted_scores = {diff: score_factor - score for diff, score in raw_scores.items()}
        sorted_scores = dict(sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True))

        formatted_scores = [(self.format_diff(diff), score) for diff, score in sorted_scores.items()]
        for diff, score in formatted_scores[:100]:
            print(diff, ' ddc = ', '{:.5f}'.format(score))

        return list(sorted_scores.items())[:500]

    def search_best_ddc_scores(self, diffs):
        """Finely calculate DDC scores and sort in descending order."""
        raw_scores = self._process_in_parallel(self._worker_entropy, diffs, self.fine_search_num, 'fine_search')
        score_factor = np.log2(self.fine_search_num)
        adjusted_scores = {diff: score_factor - score for diff, score in raw_scores.items()}
        sorted_scores = dict(sorted(adjusted_scores.items(), key=lambda item: item[1], reverse=True))
        formatted_scores = [(self.format_diff(diff), score) for diff, score in sorted_scores.items()]
        return formatted_scores[:100]

    def coarse_search(self):
        if self.coarse_search_mode == 'bias':
            return self.search_bias_scores()
        elif self.coarse_search_mode == 'DDC':
            return self.search_ddc_scores()

    def fine_search(self, diffs):
        if self.fine_search_mode == 'bias':
            return self.search_best_bias_scores(diffs)
        elif self.fine_search_mode == 'DDC':
            return self.search_best_ddc_scores(diffs)

    def search_best_diff(self):
        return self.fine_search([diff for diff, _ in self.coarse_search()])

    def format_diff(self, diff):
        """Format diff based on the current word_size, ensuring leading zeros are not removed."""
        if self.word_size == 8:
            hex_string = ''.join([f"{byte:02x}" for byte in diff])
        elif self.word_size == 16:
            hex_string = ''.join([f"{byte:04x}" for byte in diff])
        elif self.word_size == 32:
            hex_string = ''.join([f"{byte:08x}" for byte in diff])
        elif self.word_size == 64:
            hex_string = ''.join([f"{byte:016x}" for byte in diff])
        else:
            raise ValueError(f"Unsupported word size: {self.word_size}")
        return f'0x{hex_string}'


if __name__ == '__main__':
    searcher = SearchBestDiff(
        config_path='../ciphers_info/des_info.json',
        coarse_search_mode='DDC',
        coarse_search_num=10000,
        fine_search_mode='DDC',
        fine_search_num=10000000,
        nr=4,
        weights=[1, 2, 3, 4],
        max_workers=32
    )

    _top_ddc_scores = searcher.search_best_diff()
    for diff, score in _top_ddc_scores:
        print(diff, ' ddc = ', '{:.5f}'.format(score))
