#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <string.h>

#define N_THREADS 8

#define MAX_KEY_LENGTH 2000
#define MAX_QUERY_LENGTH 300

int distArray[MAX_QUERY_LENGTH + 1][MAX_KEY_LENGTH + 1];

int findShift(float* &audioPCH, float* symbPCH) {
  float bc;
  float max = 0;
  int shift = -1;
  for (int s=0; s<120; s++) {
    bc = 0;
    for (int i=0; i<120; i++)
      bc += sqrt(symbPCH[ (s+i) % 120 ] * audioPCH[i]);
    if (bc > max) {
      max = bc;
      shift = s;
    }
  }
  return int(rint(shift / 10.)) % 12;
}

int* spellAsPtr(std::vector<int> notes, int shift) {
  int* shifted = (int*)malloc((notes.size() + 1) * sizeof(int));
  for (int i=0; i<notes.size(); i++)
    shifted[i] = (notes.at(i) == -1 ? 12 : (notes.at(i) + shift) % 12);
  shifted[notes.size()] = -99;
  return shifted;
}

__uint128_t B_cache[12][13];
int k = 64;
__uint128_t R_base[65];
__uint128_t R[N_THREADS][65];

void init_bitmasks() {
  for (int i=0; i<k+1; i++)
    R_base[i] = ((__uint128_t)1 << i) - 1; // pow(2, i) - 1;

  printf("Bitmasks initialized\n");
}

void make_B_shift(int* &pattern, int pLen, int shift) {
  for (int i=0; i<13; i++)
    B_cache[shift][i] = 0;

  for (int i=0; i<pLen; i++) {
    B_cache[shift][pattern[i]] |= (__uint128_t)1 << i;

    // 'Z' ~ 12 in pattern is a wildcard
    if (pattern[i] == 12)
      for (int j=0; j<13; j++)
	B_cache[shift][j] |= (__uint128_t)1 << i;
  }

  // 'Z' ~ 12 in text is a wildcard
  // B_cache[shift][12] = ((__uint128_t)1 << pLen) - 1;
}

int editDistance2(int t_id, int pLen, int shift, int* text, int tLen) {

  __uint128_t* B = B_cache[shift];

  /* memcpy(R, R_base, (k+1) * sizeof(__uint128_t)); */
  for (int i=0; i<k+1; i++)
    R[t_id][i] = R_base[i];

  __uint128_t oldR, newR;
  int lowestMask = k+1;
  __uint128_t mask = ((__uint128_t)1 << (pLen-1)); // int(pow(2, pLen-1));

  for (int p=0; p<tLen; p++) {
    oldR = R[t_id][0];
    newR = ((oldR << 1) | 1) & B[text[p]];
    R[t_id][0] = newR;
    if ((newR & mask) != 0)
      return 0; // found a perfect match
    for (int i=1; i<k+1; i++) {
      newR = ((R[t_id][i] << 1) & B[text[p]]) | oldR | ((oldR | newR) << 1);
      oldR = R[t_id][i];
      R[t_id][i] = newR;
      if ((newR & mask) != 0 && i<lowestMask)
	lowestMask = i;
    }
  }

  return lowestMask;
}

#endif
