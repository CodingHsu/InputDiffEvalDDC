import os
import sys
import time
from secrets import randbits
import numpy as np

sys.path.append(os.path.abspath('../ciphers'))

import speck3264 as speck3264

cipher_dict = {
    "speck3264":speck3264
}


N_SAMPLES = 1000000       # n
N_ROUNDS  = 10            # nr
LOOPS     = 10

bias_times     = []
ddc_times  = []

for i in range(LOOPS):
    # (ΔL, ΔR)
    diff = (randbits(16), randbits(16))

    t0 = time.perf_counter()
    _  = cipher_dict['speck3264'].cacl_bias_score(N_SAMPLES, N_ROUNDS, diff)
    bias_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    _  = cipher_dict['speck3264'].cacl_ddc(N_SAMPLES, N_ROUNDS, diff)
    ddc_times.append(time.perf_counter() - t0)

    print(f"[{i:02d}] diff={tuple(hex(x) for x in diff)} | "
          f"bias {bias_times[-1]:.3f}s | entropy {ddc_times[-1]:.3f}s")

print("\n=== Summary ===")
print(f"bias_score  : avg {np.mean(bias_times):.3f}s  ± {np.std(bias_times):.3f}")
print(f"entropy     : avg {np.mean(ddc_times):.3f}s  ± {np.std(ddc_times):.3f}")
