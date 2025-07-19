#include <windows.h>
#include <stdint.h>

#define ROTATE_BITS(x, n) (((x) << (n)) | ((x) >> (8 - (n))) & 0xFF)

uint8_t delta[128] = {
    0x5A,0x6D,0x36,0x1B,0x0D,0x06,0x03,0x41,
    0x60,0x30,0x18,0x4C,0x66,0x33,0x59,0x2C,
    0x56,0x2B,0x15,0x4A,0x65,0x72,0x39,0x1C,
    0x4E,0x67,0x73,0x79,0x3C,0x5E,0x6F,0x37,
    0x5B,0x2D,0x16,0x0B,0x05,0x42,0x21,0x50,
    0x28,0x54,0x2A,0x55,0x6A,0x75,0x7A,0x7D,
    0x3E,0x5F,0x2F,0x17,0x4B,0x25,0x52,0x29,
    0x14,0x0A,0x45,0x62,0x31,0x58,0x6C,0x76,
    0x3B,0x1D,0x0E,0x47,0x63,0x71,0x78,0x7C,
    0x7E,0x7F,0x3F,0x1F,0x0F,0x07,0x43,0x61,
    0x70,0x38,0x5C,0x6E,0x77,0x7B,0x3D,0x1E,
    0x4F,0x27,0x53,0x69,0x34,0x1A,0x4D,0x26,
    0x13,0x49,0x24,0x12,0x09,0x04,0x02,0x01,
    0x40,0x20,0x10,0x08,0x44,0x22,0x11,0x48,
    0x64,0x32,0x19,0x0C,0x46,0x23,0x51,0x68,
    0x74,0x3A,0x5D,0x2E,0x57,0x6B,0x35,0x5A
};

void whitening_key_generation(uint8_t MK[16], uint8_t WK[8]) {
    for (int i = 0; i < 4; i++) {
        WK[i] = MK[i + 12];
        WK[i + 4] = MK[i];
    }
}

void subkey_generation(uint8_t delta[128], uint8_t MK[16], uint8_t SK[128]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            SK[16 * i + j] = (MK[(j - i + 8) % 8] + delta[16 * i + j]) % 256;
        }
        for (int j = 0; j < 8; j++) {
        	SK[16 * i + j + 8] = (MK[(j - i + 8) % 8 + 8] + delta[16 * i + j + 8]) % 256;
		}
    }
}

void encryption_key_schedule(uint8_t MK[16], uint8_t WK[8], uint8_t SK[128]) {
    whitening_key_generation(MK, WK);
    subkey_generation(delta, MK, SK);
}

void encryption_initial_transformation(uint8_t P[8], uint8_t WK[8], uint8_t X[8]) {
    X[0] = (P[0] + WK[0]) % 256;
    X[1] = P[1];
    X[2] = P[2] ^ WK[1];
    X[3] = P[3];
    X[4] = (P[4] + WK[2]) % 256;
    X[5] = P[5];
    X[6] = P[6] ^ WK[3];
    X[7] = P[7];
}

void encryption_final_transformation(uint8_t X[8], uint8_t WK[8], uint8_t C[8]) {
    C[0] = (X[1] + WK[4]) % 256;
    C[1] = X[2];
    C[2] = X[3] ^ WK[5];
    C[3] = X[4];
    C[4] = (X[5] + WK[6]) % 256;
    C[5] = X[6];
    C[6] = X[7] ^ WK[7];
    C[7] = X[0];
}

uint8_t f_0(uint8_t x) {
    return ROTATE_BITS(x, 1) ^ ROTATE_BITS(x, 2) ^ ROTATE_BITS(x, 7);
}

uint8_t f_1(uint8_t x) {
    return ROTATE_BITS(x, 3) ^ ROTATE_BITS(x, 4) ^ ROTATE_BITS(x, 6);
}

void encryption_round_function(int i, uint8_t X[8], uint8_t SK[128], uint8_t X_out[8]) {
    X_out[0] = X[7] ^ ((f_0(X[6]) + SK[4 * i + 3]) % 256);
    X_out[1] = X[0];
    X_out[2] = (X[1] + (f_1(X[0]) ^ SK[4 * i])) % 256;
    X_out[3] = X[2];
    X_out[4] = X[3] ^ ((f_0(X[2]) + SK[4 * i + 1]) % 256);
    X_out[5] = X[4];
    X_out[6] = (X[5] + (f_1(X[4]) ^ SK[4 * i + 2])) % 256;
    X_out[7] = X[6];
}

void encryption_transformation(uint8_t P[8], uint8_t WK[8], uint8_t SK[128], int round, uint8_t C[8]) {
    uint8_t X[8];
    encryption_initial_transformation(P, WK, X);
    
    for (int i = 0; i < round-1; i++) {
        uint8_t X_temp[8];
        encryption_round_function(i, X, SK, X_temp);
        for (int j = 0; j < 8; j++) {
            X[j] = X_temp[j];
        }
    }
    
    uint8_t X_temp[8];
    encryption_round_function(round-1, X, SK, X_temp);
    for (int j = 0; j < 8; j++) {
        X[j] = X_temp[j];
    }

    encryption_final_transformation(X, WK, C);
}

extern "C" __declspec(dllexport) void encrypt(uint8_t plaintext[8], uint8_t key[16], int round, uint8_t ciphertext[8]) {
    uint8_t WK[8], SK[128];
    encryption_key_schedule(key, WK, SK);
    encryption_transformation(plaintext, WK, SK, round, ciphertext);
}

extern "C" __declspec(dllexport) void encrypt_batch(uint8_t plaintext[][8], uint8_t key[][16], const int rounds, uint8_t ciphertext[][8], const int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        encrypt(plaintext[i], key[i], rounds, ciphertext[i]);
    }
}

