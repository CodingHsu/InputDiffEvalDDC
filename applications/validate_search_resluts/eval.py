import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DataGenerator import DataGeneratorManager
from tensorflow.keras.models import load_model

path = '../saved_trained_net/speck64/speck64128_single-key_0x9010000000_dbitnet_round8.h5'
dm  = DataGeneratorManager(
                        cipher_info_path=str(cipher_info_path),
                        test_feature=test_feature,
                        test_rounds=test_rounds,
                        num_samples=NUM_SAMPLES,
                        batch_size=BATCH_SIZE,
                        train_ratio=TRAIN_RATIO,
                        val_ratio=VAL_RATIO,
                    )

net = load_model(path)
net.evaluate(dm.test_generator, verbose=1)