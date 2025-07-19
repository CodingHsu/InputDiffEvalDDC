import ctypes
from os import urandom
import numpy as np


class DatasetGenerate:
    def __init__(self):
        pass

    @staticmethod
    def _data_type_to_size(data_type):
        if data_type == 'uint8':
            return 8
        elif data_type == 'uint16':
            return 16
        elif data_type == 'uint32':
            return 32
        elif data_type == 'uint64':
            return 64
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    @staticmethod
    def _data_type_to_np_type(data_type):
        if data_type == 'uint8':
            return np.uint8
        elif data_type == 'uint16':
            return np.uint16
        elif data_type == 'uint32':
            return np.uint32
        elif data_type == 'uint64':
            return np.uint64
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    @staticmethod
    def _data_type_to_ctype(data_type):
        if data_type == 'uint8':
            return ctypes.c_uint8
        elif data_type == 'uint16':
            return ctypes.c_uint16
        elif data_type == 'uint32':
            return ctypes.c_uint32
        elif data_type == 'uint64':
            return ctypes.c_uint64
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    @staticmethod
    def _parse_diff(feature, plaintext_size, data_type):
        if 4 * len(feature) != plaintext_size:
            raise ValueError(f"Invalid feature string length, expected input length is {plaintext_size // 4}")
        step = DatasetGenerate._data_type_to_size(data_type) // 4
        return [int(feature[i:i + step], 16) for i in range(0, len(feature), step)]

    def generate_differential_dataset(self, cipher_saved_path, data_type, plaintext_size, ciphertext_size,
                                      master_key_size, test_feature, test_rounds, dataset_size, batch_size):
        try:
            encrypt_batch = DLLLoader.load(cipher_saved_path, data_type)

            label = self._generate_labels(dataset_size)
            master_keys = self._generate_random_data(data_type, master_key_size, dataset_size)
            plaintext0 = self._generate_random_data(data_type, plaintext_size, dataset_size)
            plaintext1 = plaintext0 ^ np.array(self._parse_diff(test_feature, ciphertext_size, data_type),
                                               dtype=self._data_type_to_np_type(data_type))
            num_rand_samples = np.sum(label == 0)
            plaintext1[label == 0, :] = np.frombuffer(urandom(num_rand_samples * (plaintext_size // 8)),
                                                      dtype=self._data_type_to_np_type(data_type))\
                                          .reshape(num_rand_samples, plaintext_size // self._data_type_to_size(data_type))
            plaintext0 = self._transfer_to_c_style(data_type, plaintext0)
            plaintext1 = self._transfer_to_c_style(data_type, plaintext1)
            master_keys = self._transfer_to_c_style(data_type, master_keys)

            ciphertext0 = np.zeros((dataset_size, ciphertext_size // self._data_type_to_size(data_type)),
                                   dtype=self._data_type_to_np_type(data_type))
            ciphertext1 = np.zeros((dataset_size, ciphertext_size // self._data_type_to_size(data_type)),
                                   dtype=self._data_type_to_np_type(data_type))

            for i in range(0, dataset_size, batch_size):
                end_index = min(i + batch_size, dataset_size)

                batch_pt0_ctypes = self._transfer_to_ctypes(data_type, plaintext0[i:end_index])
                batch_pt1_ctypes = self._transfer_to_ctypes(data_type, plaintext1[i:end_index])
                batch_mks_ctypes = self._transfer_to_ctypes(data_type, master_keys[i:end_index])
                batch_ct0_ctypes = self._transfer_to_ctypes(data_type, ciphertext0[i:end_index])
                batch_ct1_ctypes = self._transfer_to_ctypes(data_type, ciphertext1[i:end_index])

                encrypt_batch(batch_pt0_ctypes, batch_mks_ctypes, test_rounds, batch_ct0_ctypes, end_index - i)
                encrypt_batch(batch_pt1_ctypes, batch_mks_ctypes, test_rounds, batch_ct1_ctypes, end_index - i)

            return [ciphertext0, ciphertext1], label
        finally:
            DLLLoader.release()

    @staticmethod
    def _generate_labels(dataset_size):
        return np.frombuffer(urandom(dataset_size), dtype=np.uint8) & 1

    def _generate_random_data(self, data_type, input_size, dataset_size):
        return np.frombuffer(urandom(dataset_size * (input_size // 8)), dtype=self._data_type_to_np_type(data_type)) \
                 .reshape(dataset_size, input_size // self._data_type_to_size(data_type))

    def _transfer_to_c_style(self, data_type, data):
        # Ensure data alignment to avoid data corruption or undefined behavior due to memory layout issues,
        # which is especially important when interacting with C functions
        # that have strict requirements on input data memory layout
        return np.require(data, dtype=self._data_type_to_np_type(data_type), requirements=['C', 'A'])

    def _transfer_to_ctypes(self, data_type, data):
        return data.ctypes.data_as(ctypes.POINTER(self._data_type_to_ctype(data_type)))


class DLLLoader:
    _loaded_dll = None

    @staticmethod
    def load(saved_path, data_type):
        """Load DLL and bind necessary functions"""
        try:
            DLLLoader._loaded_dll = ctypes.CDLL(saved_path)

            # Load primary function and round key generation function based on data type
            ctype = DLLLoader._get_ctype(data_type)
            primary_function = DLLLoader._load_primary_function(DLLLoader._loaded_dll, ctype)

            if primary_function is None:
                raise RuntimeError("Neither 'encrypt_batch' nor 'generate_key_stream_byte_batch' found in the DLL.")

            return primary_function

        except OSError as e:
            raise RuntimeError(f"Failed to load DLL from {saved_path}: {str(e)}")

    @staticmethod
    def _get_ctype(data_type):
        """Convert NumPy data type to ctypes type"""
        if data_type == 'uint8':
            return ctypes.c_uint8
        elif data_type == 'uint16':
            return ctypes.c_uint16
        elif data_type == 'uint32':
            return ctypes.c_uint32
        elif data_type == 'uint64':
            return ctypes.c_uint64
        else:
            raise ValueError(f"Unsupported DLL loading data type: {data_type}")

    @staticmethod
    def _load_primary_function(dll, ctype):
        """Bind the primary batch processing function encrypt_batch based on data type"""
        if hasattr(dll, 'encrypt_batch'):
            return DLLLoader._bind_encrypt_batch(dll, ctype)
        return None

    @staticmethod
    def _bind_encrypt_batch(dll, ctype):
        """Bind encrypt_batch function"""
        func = getattr(dll, 'encrypt_batch')
        func.argtypes = (
            ctypes.POINTER(ctype),  # plaintext
            ctypes.POINTER(ctype),  # master_key
            ctypes.c_int,           # rounds
            ctypes.POINTER(ctype),  # ciphertext
            ctypes.c_int            # batch_size
        )
        func.restype = None
        return func

    @staticmethod
    def release():
        if DLLLoader._loaded_dll is not None:
            handle = DLLLoader._loaded_dll._handle
            ctypes.windll.kernel32.FreeLibrary(handle)
            DLLLoader._loaded_dll = None
