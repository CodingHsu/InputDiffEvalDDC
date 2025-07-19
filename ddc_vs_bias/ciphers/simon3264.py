import numpy as np
from os import urandom
from scipy.special import entr
import pandas as pd

def WORD_SIZE():
    return 16
    
def ALPHA():
    return 1
    
def BETA():
    return 8
    
def GAMMA():
    return 2

MASK_VAL = 2**WORD_SIZE() - 1

def rol(x, k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))
    
def ror(x, k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def enc_one_round(p, k):
    tmp, c1 = p[0], p[1]
    tmp = rol(tmp, ALPHA()) & rol(tmp, BETA())
    tmp = tmp ^ rol(p[0], GAMMA())
    c1 = c1 ^ tmp
    c1 = c1 ^ k
    return c1, p[0]

def dec_one_round(c, k):
    p0, p1 = c[0], c[1]
    tmp = tmp ^ rol(p1, GAMMA())
    p1 = tmp ^ c[0] ^ k
    p0 = c1
    return p0, p1

def encrypt(p, k):
    x, y = p[0], p[1]
    for rk in k:
        x,y = enc_one_round((x,y), rk)
    return x, y

def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0:4] = reversed(k[0:4])
    m = 4
    round_constant = MASK_VAL ^ 3
    z = (0b01100111000011010100100010111110110011100001101010010001011111)
    for i in range(m, t):
        c_z = ((z >> ((i-m) % 62)) & 1) ^ round_constant
        tmp = ror(ks[i-1], 3)
        tmp = tmp ^ ks[i-3]
        tmp = tmp ^ ror(tmp, 1)
        ks[i] = ks[i-m] ^ tmp ^ c_z
    return ks

def check_testvector():
    p = (0x6565, 0x6877)
    k = (0x1918, 0x1110, 0x0908, 0x0100)
    ks = expand_key(k, 32)
    c = encrypt(p, ks)
    if (c == (0xc69b, 0xe9bb)):
        print("Testvector verified.")
        return True
    else:
        print("Testvector not verified.")
        return False

def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8)
    for i in range(len(arr) * WORD_SIZE()):
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

def cacl_bais_score(n, nr, diff=(0,0)):
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
