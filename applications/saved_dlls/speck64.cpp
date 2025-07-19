#include <windows.h>
#include <stdint.h>
#include <cstdlib>

static inline void speck_round(uint32_t& x, uint32_t& y, uint32_t k) {
	x = (x >> 8) | (x << (8 * sizeof(x) - 8)); // x = ROTR(x, 8)
	x += y;
	x ^= k;
	y = (y << 3) | (y >> (8 * sizeof(y) - 3)); // y = ROTL(y, 3)
	y ^= x;
}

extern "C" __declspec(dllexport) void generate_round_key(const uint32_t key[4], uint32_t round_key[27][1]) {
    uint32_t b = key[0];
    uint32_t a0 = key[1];
    uint32_t a1 = key[2];
    uint32_t a2 = key[3];
    
    for (int i = 0; i < 27; i++) {
        round_key[i][0] = b;
        uint32_t a = a0;
        speck_round(a, b, i);
        a0 = a1;
        a1 = a2;
        a2 = a;
    }
}

extern "C" __declspec(dllexport) void encrypt(const uint32_t plaintext[2], const uint32_t key[4], const int rounds, uint32_t ciphertext[2]) {
    ciphertext[0] = plaintext[0];
    ciphertext[1] = plaintext[1];
    uint32_t b  = key[0];
    uint32_t a0 = key[1];
    uint32_t a1 = key[2];
    uint32_t a2 = key[3];
    for (unsigned i = 0; i < rounds; i++) {
        speck_round(ciphertext[1], ciphertext[0], b);
        uint32_t a = a0;
        speck_round(a, b, i);
        a0 = a1;
        a1 = a2;
        a2 = a;
    }
}

extern "C" __declspec(dllexport) void encrypt_batch(const uint32_t plaintext[][2], const uint32_t key[][4], const int rounds, uint32_t ciphertext[][2], const int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        encrypt(plaintext[i], key[i], rounds, ciphertext[i]);
    }
}
