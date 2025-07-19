import os
import sys
from scipy import stats

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264

cipher_dict = {
    "speck3264":speck3264
}

from tensorflow.keras.models import load_model

cipher = cipher_dict['speck3264']
net = load_model('./best7depth10.h5')

x0, y0 = cipher.make_random_data(10**7, 7)
x1, y1 = cipher.make_diff_data(10**7, 7, (0x0040, 0))
x2, y2 = cipher.make_backward_real_diff_data(10**7, 7, 2, (0x0040, 0))
z0 = net.predict(x0, batch_size=10000).flatten()
z1 = net.predict(x1, batch_size=10000).flatten()
z2 = net.predict(x2, batch_size=10000).flatten()

D, p = stats.ks_2samp(z1, z2, method='auto')
from scipy.stats import norm
sigma_equiv = norm.isf(p / 2)      # isf ≡ ppf(1 - p/2)

print(f"KS D = {D:.6g}")
print(f"p-value = {p:.3e}")
print(f"σ-equivalent = {sigma_equiv:.3f} σ")
