import numpy as np
from os import urandom
from scipy.special import entr
import pandas as pd

def WORD_SIZE():
    return 16
    
def ALPHA():
    return 7
    
def BETA():
    return 2
    
MASK_VAL = 2 ** WORD_SIZE() - 1

def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def enc_one_round(p, k):
    c0, c1 = p[0], p[1]
    c0 = ror(c0, ALPHA())
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rol(c1, BETA())
    c1 = c1 ^ c0
    return c0, c1

def dec_one_round(c, k):
    c0, c1 = c[0], c[1]
    c1 = c1 ^ c0
    c1 = ror(c1, BETA())
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rol(c0, ALPHA())
    return c0, c1

def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0] = k[len(k)-1]
    l = list(reversed(k[:len(k)-1]))
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i)
    return ks

def encrypt(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return x, y

def decrypt(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k)
    return x, y

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return True
  else:
    print("Testvector not verified.")
    return False

def convert_to_binary(arr):
  X = np.zeros((WORD_SIZE() * len(arr), len(arr[0])),dtype=np.uint8)
  for i in range(WORD_SIZE() * len(arr)):
    index = i // WORD_SIZE()
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose()
  return X

def make_train_data(n, nr, diff=(0,0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y==0)
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X, Y

def make_train_data_for_predict(n, nr, backwards_num=2, diff=(0,0)):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    decdata0l, decdata0r = decrypt((ctdata0l, ctdata0r), ks[-backwards_num:])
    decdata1l, decdata1r = decrypt((ctdata1l, ctdata1r), ks[-backwards_num:])
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    Y = convert_to_binary([decdata0l ^ decdata1l, decdata0r ^ decdata1r])
    return X, Y

def make_train_data_for_truncated_diff_predict(n, nr, backwards_num=2, diff=(0, 0)):
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    decdata0l, decdata0r = decrypt((ctdata0l, ctdata0r), ks[-backwards_num:])
    decdata1l, decdata1r = decrypt((ctdata1l, ctdata1r), ks[-backwards_num:])
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    # Create Y with shape (n, 12) and dtype np.uint8
    Y = np.zeros((n, 12), dtype=np.uint8)
    
    # Populate Y with bit values from decdata0l ^ decdata1l and decdata0r ^ decdata1r
    diff_l = decdata0l ^ decdata1l
    diff_r = decdata0r ^ decdata1r

    Y[:, 0] = (diff_l >> 15) & 1  # Bit 0
    Y[:, 1] = (diff_l >> 14) & 1  # Bit 1
    Y[:, 2] = (diff_l >> 8) & 1   # Bit 7
    Y[:, 3] = (diff_l >> 7) & 1   # Bit 8
    Y[:, 4] = (diff_l >> 1) & 1   # Bit 14
    Y[:, 5] = diff_l & 1          # Bit 15

    Y[:, 6] = (diff_r >> 15) & 1  # Bit 0
    Y[:, 7] = (diff_r >> 14) & 1  # Bit 1
    Y[:, 8] = (diff_r >> 8) & 1   # Bit 7
    Y[:, 9] = (diff_r >> 7) & 1   # Bit 8
    Y[:, 10] = (diff_r >> 1) & 1  # Bit 14
    Y[:, 11] = diff_r & 1         # Bit 15

    return X, Y

def make_train_data_for_truncated_diff_analysis(n):
    td3 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    td3_index = [0, 1, 7, 8, 14, 15, 16, 17, 23, 24, 30, 31]
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    decdata0l, decdata0r = decrypt((ctdata0l, ctdata0r), ks[-backwards_num:])
    decdata1l, decdata1r = decrypt((ctdata1l, ctdata1r), ks[-backwards_num:])
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    # Create Y with shape (n, 12) and dtype np.uint8
    Y = np.zeros((n, 12), dtype=np.uint8)
    
    # Populate Y with bit values from decdata0l ^ decdata1l and decdata0r ^ decdata1r
    diff_l = decdata0l ^ decdata1l
    diff_r = decdata0r ^ decdata1r

    Y[:, 0] = (diff_l >> 15) & 1  # Bit 0
    Y[:, 1] = (diff_l >> 14) & 1  # Bit 1
    Y[:, 2] = (diff_l >> 8) & 1   # Bit 7
    Y[:, 3] = (diff_l >> 7) & 1   # Bit 8
    Y[:, 4] = (diff_l >> 1) & 1   # Bit 14
    Y[:, 5] = diff_l & 1          # Bit 15

    Y[:, 6] = (diff_r >> 15) & 1  # Bit 0
    Y[:, 7] = (diff_r >> 14) & 1  # Bit 1
    Y[:, 8] = (diff_r >> 8) & 1   # Bit 7
    Y[:, 9] = (diff_r >> 7) & 1   # Bit 8
    Y[:, 10] = (diff_r >> 1) & 1  # Bit 14
    Y[:, 11] = diff_r & 1         # Bit 15

    return X, Y

def make_train_data_for_truncated_diff_predict_random(n, nr1, nr2, diff=(0, 0)):
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ks = expand_key(keys, nr1)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    decdata0l, decdata0r = decrypt((ctdata0l, ctdata0r), ks[-nr2:])
    decdata1l, decdata1r = decrypt((ctdata1l, ctdata1r), ks[-nr2:])
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    # Create Y with shape (n, 12) and dtype np.uint8
    Y = np.zeros((n, 12), dtype=np.uint8)
    
    # Populate Y with bit values from decdata0l ^ decdata1l and decdata0r ^ decdata1r
    diff_l = decdata0l ^ decdata1l
    diff_r = decdata0r ^ decdata1r

    Y[:, 0] = (diff_l >> 14) & 1  # Bit 0
    Y[:, 1] = (diff_l >> 11) & 1  # Bit 1
    Y[:, 2] = (diff_l >> 10) & 1   # Bit 7
    Y[:, 3] = (diff_l >> 8) & 1   # Bit 8
    Y[:, 4] = (diff_l >> 4) & 1   # Bit 14
    Y[:, 5] = diff_l & 1          # Bit 15

    Y[:, 6] = (diff_r >> 15) & 1  # Bit 0
    Y[:, 7] = (diff_r >> 10) & 1  # Bit 1
    Y[:, 8] = (diff_r >> 5) & 1   # Bit 7
    Y[:, 9] = (diff_r >> 4) & 1   # Bit 8
    Y[:, 10] = (diff_r >> 2) & 1  # Bit 14
    Y[:, 11] = diff_r & 1         # Bit 15

    return X, Y

def real_differences_data(n, nr, diff=(0,0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    num_rand_samples = np.sum(Y==0)
    ks = expand_key(keys, nr)
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks)
    k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0
    ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1
    ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0
    ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X, Y

def backward_real_differences_data(n, nr, backwards_num, diff=(0,0)):
    #generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    #generate keys
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    #generate plaintexts
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #apply input difference
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y==0);
    #expand keys and encrypt
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks[:-backwards_num]);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks[:-backwards_num]);
    #generate blinding values
    k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    #apply blinding to the samples labelled as random
    ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
    ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
    
    ctdata0l, ctdata0r = encrypt((ctdata0l, ctdata0r), ks[-backwards_num:]);
    ctdata1l, ctdata1r = encrypt((ctdata1l, ctdata1r), ks[-backwards_num:]);
    #convert to input data for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def make_random_data(n, nr):
    #generate labels
    Y = np.zeros(n,dtype=np.uint8);
    #generate keys
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    #generate plaintexts
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #expand keys and encrypt
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    #convert to input data for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def make_diff_data(n, nr, diff=(0,0)):
    #generate labels
    Y = np.ones(n,dtype=np.uint8);
    #generate keys
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    #generate plaintexts
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #apply input difference
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    #expand keys and encrypt
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    #convert to input data for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def make_backward_real_diff_data(n, nr, backwards_num, diff=(0,0)):
    #generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    #generate keys
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    #generate plaintexts
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #apply input difference
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    #expand keys and encrypt
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks[:-backwards_num]);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks[:-backwards_num]);
    #generate blinding values
    k0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    k1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #apply blinding to the samples labelled as random
    ctdata0l = ctdata0l ^ k0; ctdata0r = ctdata0r ^ k1;
    ctdata1l = ctdata1l ^ k0; ctdata1r = ctdata1r ^ k1;
    
    ctdata0l, ctdata0r = encrypt((ctdata0l, ctdata0r), ks[-backwards_num:]);
    ctdata1l, ctdata1r = encrypt((ctdata1l, ctdata1r), ks[-backwards_num:]);
    #convert to input data for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def make_real_diff_data(n, nr, diff=(0,0)):
    #generate labels
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    #generate keys
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    #generate plaintexts
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #apply input difference
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    #expand keys and encrypt
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    #generate blinding values
    k0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    k1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    #apply blinding to the samples labelled as random
    ctdata0l = ctdata0l ^ k0; ctdata0r = ctdata0r ^ k1;
    ctdata1l = ctdata1l ^ k0; ctdata1r = ctdata1r ^ k1;
    #convert to input data for neural networks
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def cacl_bias_score(n, nr, diff=(0,0)):
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4,-1)
    p0_l = np.frombuffer(urandom(2*n), dtype=np.uint16)
    p0_r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    p1_l = p0_l ^ diff[0]
    p1_r = p0_r ^ diff[1]
    ks = expand_key(keys, nr)
    c0_l, c0_r = encrypt((p0_l, p0_r), ks)
    c1_l, c1_r = encrypt((p1_l, p1_r), ks)
    d = convert_to_binary([c0_l ^ c1_l, c0_r ^ c1_r]).transpose()
    score = np.average(np.abs(0.5-np.average(d, axis=0)), axis=0)
    return score

def cacl_ddc(n, nr, diff=(0,0)):
    keys = np.frombuffer(urandom(8*n), dtype=np.uint16).reshape(4,-1)
    p0_l = np.frombuffer(urandom(2*n), dtype=np.uint16)
    p0_r = np.frombuffer(urandom(2*n), dtype=np.uint16)
    p1_l = p0_l ^ diff[0]
    p1_r = p0_r ^ diff[1]
    ks = expand_key(keys, nr)
    c0_l, c0_r = encrypt((p0_l, p0_r), ks)
    c1_l, c1_r = encrypt((p1_l, p1_r), ks)
    d_l, d_r = c0_l ^ c1_l, c0_r ^ c1_r
    
    # Combine d_l and d_r into a single DataFrame
    d_combined = pd.DataFrame({'d_l': d_l, 'd_r': d_r})
    
    # Calculate the frequency of each unique tuple in d_combined
    counts = d_combined.value_counts()
    
    # Calculate the probability distribution
    probabilities = counts / counts.sum()
    
    # Calculate the entropy using scipy.special.entr
    entropy_value = np.sum(entr(probabilities))
    
    return np.log2(n) - entropy_value
