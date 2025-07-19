import os
import sys

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264

cipher_dict = {
    "speck3264":speck3264
}

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model


def make_fig(nr, z0, z1, z2):
    plt.figure(figsize=(14, 7), dpi=300)

    sns.kdeplot(z0, fill=True, color="gold", label="Random", alpha=0.8)
    sns.kdeplot(z1, fill=True, color="#32CD32", label="Real", alpha=0.8)
    sns.kdeplot(z2, fill=True, color="#4682B4", label="Real'", alpha=0.8)

    plt.xlabel('Scores of samples', fontsize=14)
    plt.ylabel('Densities of samples', fontsize=14)

    plt.legend(loc='upper left', fontsize=12)

    # plt.savefig(str(nr) + 'r_00400000_backward2.svg', dpi=300)
    # plt.close()
    plt.savefig(f"{nr}r_00400000_backward2.pdf", dpi=300)
    plt.close()


if __name__ == '__main__':
    cipher = cipher_dict['speck3264']
    net = load_model('./best7depth10.h5')

    x0, y0 = cipher.make_random_data(10**7, 7)
    x1, y1 = cipher.make_diff_data(10**7, 7, (0x0040, 0))
    x2, y2 = cipher.make_backward_real_diff_data(10**7, 7, 2, (0x0040, 0))
    z0 = net.predict(x0, batch_size=10000).flatten()
    z1 = net.predict(x1, batch_size=10000).flatten()
    z2 = net.predict(x2, batch_size=10000).flatten()

    offset = 0.005
    z2_offset = z2 + offset
    make_fig(7, z0, z1, z2_offset)
    # print(f"The plot has been saved at '5r_00400000_backward2.svg'.")
    print("The plot has been saved at '7r_00400000_backward2.pdf'.")
