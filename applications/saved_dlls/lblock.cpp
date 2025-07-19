#include <windows.h>
#include <stdint.h>

uint16_t S[10][16]={{14,9,15,0,13,4,10,11,1,2,8,3,7,6,12,5},{4,11,14,9,15,13,0,10,7,12,5,6,2,8,1,3},{1,14,7,12,15,13,0,6,11,5,9,3,2,4,8,10},
                    {7,6,8,11,0,15,3,14,9,10,12,13,5,2,4,1},{14,5,15,0,7,2,12,13,1,8,4,9,11,10,6,3},{2,13,11,12,15,14,0,9,7,10,6,3,1,8,4,5},
                    {11,9,4,14,0,15,10,13,6,12,5,7,3,8,1,2},{13,10,15,0,14,4,9,11,2,1,8,3,7,5,12,6},{8,7,14,5,15,13,0,6,11,12,9,10,2,4,1,3},
                    {11,5,15,0,7,2,9,13,4,8,1,12,14,10,3,6}};

uint16_t P[8]={1,3,0,2,5,7,4,6};

static inline uint32_t load_uint32_be(const uint16_t *b, uint32_t n) {
	return ( ((uint32_t)b[n*2] << 16) | ((uint32_t)b[n*2+1]) );
}

static inline void store_uint32_be(uint32_t n, uint16_t * const b) {
	b[0] = (uint16_t)(n>>16);
	b[1] = (uint16_t)(n);
}

void roundkey(uint32_t k[5], int rounds, uint32_t *rk){
    uint32_t i,t1,t2,t3,t4,t0,c1,c2;
    rk[0]=(k[0]<<16)^ k[1];
    for(i=1;i<rounds;i++){
        // 32 left shift (then 3 right shift)
        t1=k[0]; t2=k[1];
        k[0]=k[2];  k[1]=k[3]; k[2]=k[4]; k[3]=t1; k[4]=t2;
        //3 right shift
        t0=(k[0]&0x7);              t1=(k[1]&0x7);              t2=(k[2]&0x7);
        t3=(k[3]&0x7);              t4=(k[4]&0x7);
        k[4]=(k[4]>>3)^(t3<<13);
        k[3]=(k[3]>>3)^(t2<<13);
        k[2]=(k[2]>>3)^(t1<<13);
        k[1]=(k[1]>>3)^(t0<<13);
        k[0]=(k[0]>>3)^(t4<<13);
        //s-box
        t1=(k[0]>>12)&0xF;
        t2=(k[0]>>8)&0xF;
        t1=S[9][t1];
        t2=S[8][t2];
        k[0]=(t1<<12)^(t2<<8)^(k[0]&0x00FF);
        //counter
        c1=i&0x3; c2=i>>2;
        k[2]^=c1<<14; k[1]^=c2;
        //get roundkey
        rk[i]=(k[0]<<16)^ k[1];
    }
}

uint32_t S_Layer(uint32_t x){
    uint32_t temp=0x0;
    int i;
    for(i=0;i<7;i++){
        temp^=S[7-i][(x>>(28-4*i))&0xF];
        temp<<=4;
    }                               
    temp^=S[7-i][x&0xF];
    return temp;
}

uint32_t P_Layer(uint32_t x){
    uint16_t temp[8],i;
    uint32_t t=0x0;
   
    for(i=0;i<8;i++)
        temp[i]=(x>>(28-(4*i)))&0xF;
   
    for(i=0;i<7;i++){
        t^=temp[P[i]];
        t<<=4;
    }
    t^=temp[P[i]];
   
    return t;
}

uint32_t F(uint32_t x, uint32_t k){
    x^=k;
    x=S_Layer(x);
    x=P_Layer(x);
    return x;
}

void swap(uint32_t *left, uint32_t *right){
    uint32_t temp;
    temp=(*left);
    (*left)=(*right);
    (*right)=temp;
}

extern "C" __declspec(dllexport) void encrypt(const uint16_t plaintext[4], const uint16_t key[5], const int rounds, uint16_t ciphertext[4]) {
	uint32_t rk[rounds];
	uint32_t k1[5] = {(uint32_t)key[0], (uint32_t)key[1], (uint32_t)key[2], (uint32_t)key[3], (uint32_t)key[4]};
	roundkey(k1, rounds, rk);
	uint32_t left, right;
	left = load_uint32_be(plaintext, 0); right = load_uint32_be(plaintext, 1);
	           
	for(int i=0; i<rounds; i++){
	    right=(right<<8)^(right>>24);       
	    right^=F(left, rk[i]);
	    swap(&left,&right);
	}
	
	store_uint32_be(right, ciphertext);
	store_uint32_be(left, ciphertext+2);
}

extern "C" __declspec(dllexport) void encrypt_batch(const uint16_t plaintext[][4], const uint16_t key[][5], const int rounds, uint16_t ciphertext[][4], const int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        encrypt(plaintext[i], key[i], rounds, ciphertext[i]);
    }
}

