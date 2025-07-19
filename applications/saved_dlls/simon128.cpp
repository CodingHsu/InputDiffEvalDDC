#include <windows.h>
#include <stdint.h>

#define ROTATE_R(v, d) ((v >> d) | (v << (8 * sizeof(v) - d)))
#define ROTATE_L(v, d) ((v << d) | (v >> (8 * sizeof(v) - d)))

#define FIESTEL_ROTATE(v) ((ROTATE_L(v, 1) & ROTATE_L(v, 8)) ^ ROTATE_L(v, 2))

#define CONSTANT_C 0xfffffffffffffffc


// TODO: accept other sizes
#define KEY_WORDS  4
#define Z_VECTOR_J 4


const uint64_t Z_VECTOR[5][62] = {
    {1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0},
    {1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0},
    {1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,1},
    {1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1},
    {1,1,0,1,0,0,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,1,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,1,1}
};

void key_schedule(uint64_t key[], int round, uint64_t dest[])
{
    for (int i = 0; i < KEY_WORDS; i++)
        dest[i] = key[i];
    
    for (int i = KEY_WORDS; i < round; i++){
        uint64_t y = ROTATE_R(dest[i - 1], 3);
        if (KEY_WORDS == 4)
            y ^= dest[i - 3];
        dest[i] = dest[i - KEY_WORDS] ^ y ^ ROTATE_R(y, 1) ^ CONSTANT_C ^ Z_VECTOR[Z_VECTOR_J][(i - KEY_WORDS) % 62];
    }
}

static inline void simon_round(uint64_t* pt1, uint64_t* pt2, uint64_t key)
{
    uint64_t _pt1 = *pt1;
    *pt1 = *pt2 ^ FIESTEL_ROTATE(*pt1) ^ key;
    *pt2 = _pt1;
}

extern "C" __declspec(dllexport) void encrypt(const uint64_t pt[2], uint64_t key[4], int round, uint64_t ct[2])
{
    ct[0] = pt[0];
    ct[1] = pt[1];
    uint64_t sk[round];
    key_schedule(key, round, sk);
    
    for (int i = 0; i < round; i++)
        simon_round(&ct[0], &ct[1], sk[i]);
}

extern "C" __declspec(dllexport) void encrypt_batch(const uint64_t pt[][2], uint64_t key[][4], int round, uint64_t ct[][2], const int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        encrypt(pt[i], key[i], round, ct[i]);
    }
}
