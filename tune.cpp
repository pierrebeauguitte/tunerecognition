#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <string>
#include <string.h>
#include "key.h"
#include "tune.h"

float* Tune::convolve(float* hist, int n, float sigma) {

  // prepare kernel
  int lw = int(4.0 * sigma + 0.5);
  float *weights = (float*)malloc((2*lw+1) * sizeof(float));
  weights[lw] = 1.0;
  float sum = 1.0;
  float sd = sigma * sigma;
  for (int i=1; i<=lw; i++) {
    float tmp = exp(-0.5 * float(i * i) / sd);
    weights[lw + i] = tmp;
    weights[lw - i] = tmp;
    sum += 2.0 * tmp;
  }
  for (int i=0; i<2*lw+1; i++)
    weights[i] /= sum;

  // convolve
  float *res = (float*)malloc(n * sizeof(float));
  for (int i=0; i<n; i++) {
    float tmp = 0;
    int start = i-lw;
    if (start < 0)
      start += n;
    for (int j=0; j<2*lw+1; j++)
      tmp += hist[ (start+j) % n ] * weights[j];
    res[i] = tmp;
  }

  free(weights);
  return res;
}

#define NOMOD -10

std::vector<int> Tune::ABCtoPitchClass(char* abc, Key* k) {
  std::vector<int> notes;
  int mod = NOMOD;
  char *p = abc;
  while (*p != '\0') {
    if (*p == 'Z')
      notes.push_back(12);
    else if (*p == '_')
      mod = -1;
    else if (*p == '=')
      mod = 0;
    else if (*p == '^')
      mod = 1;
    else if (*p >= 'A' && *p <= 'G') {
      int pitch = k->pitchClass(*p);
      if (mod != NOMOD) {
	if (mod == 0)
	  pitch = k->naturalPitchClass(*p);
	else
	  pitch += mod;
	mod = NOMOD;
      }
      notes.push_back(pitch);
    } else
      printf("Unrecognized character [%c]\n", *p);
    p++;
  }
  return notes;
}

float* Tune::makePCH(char* arr, bool fifth) {
  float* hist = (float*)malloc(120 * sizeof(float));
  char* p = arr;
  p++; // skip '['
  for (int i=0; i<12; i++) {
    int v = strtof(p, &p);
    p++; //skip ','
    hist[i*10] = v;
    // Option: add fifth, to be more audio-like...
    if (fifth)
      hist[((i+7) % 12) * 10] += v/2;
    for (int j=1; j<10; j++)
      hist[i*10 + j] = 0;
  }

  float* res = convolve(hist, 120, 1.5);
  free(hist);

  float s = 0;
  for (int i=0; i<120; i++)
    s += res[i];

  for (int i=0; i<120; i++)
    res[i] /= s;

  return res;
}

int* Tune::vec2ptr(std::vector<int> k) {
  int* p = (int*)malloc((k.size() + 1) * sizeof(int));
  for (int i=0; i<k.size(); i++)
    p[i] = k.at(i);
  p[k.size()] = -99;
  return p;
}

int* Tune::vec2ptrInt(std::vector<int> k) {
  int* p = (int*)malloc((k.size()+1) * sizeof(int));
  int pos = 0;
  int lastPitch = -1;
  for (int i=0; i<k.size(); i++) {
    if (lastPitch < 0) {
      if (k.at(i) != 12)
	lastPitch = k.at(i);
      continue;
    } // loops until 1st non 12 pitch found
    if (k.at(i) == 12)
      p[pos++] = 0;
    else {
      p[pos++] = (k.at(i) - lastPitch) % 12;
      lastPitch = k.at(i); // lastPitch only takes non12 values
    }
  }
  this->intKeySize = pos-1;
  p[pos] = -99;
  return p;
}

void Tune::initTypes(char* type) {

  if (strcmp(type, "jig") == 0) {
    this->beattype = COMPOUND;
    this->tunetype = JIG;
  } else if (strcmp(type, "slip jig") == 0) {
    this->beattype = COMPOUND;
    this->tunetype = SLIPJIG;
  } else if (strcmp(type, "slide") == 0) {
    this->beattype = COMPOUND;
    this->tunetype = SLIDE;
  } else if (strcmp(type, "reel") == 0) {
    this->beattype = SIMPLE;
    this->tunetype = REEL;
  } else if (strcmp(type, "hornpipe") == 0) {
    this->beattype = SIMPLE;
    this->tunetype = HORNPIPE;
  } else if (strcmp(type, "polka") == 0) {
    this->beattype = SIMPLE;
    this->tunetype = POLKA;
  } else if (strcmp(type, "waltz") == 0 ||
	     strcmp(type, "mazurka") == 0 ||
	     strcmp(type, "three-two") == 0) {
    this->beattype = SIMPLE;
    this->tunetype = WALTZ;
  } else if (strcmp(type, "strathspey") == 0 ||
	     strcmp(type, "barndance") == 0) {
    this->beattype = SIMPLE;
    this->tunetype = OTHER44;
  }
}

Tune::Tune(sqlite3_stmt *st, bool fifth) {
  this->id          = sqlite3_column_int(st, 0);
  this->setting     = sqlite3_column_int(st, 1);
  this->name        = strdup((char*)sqlite3_column_text(st, 2));
  this->initTypes((char*)sqlite3_column_text(st, 3));
  this->key         = new Key((char*)sqlite3_column_text(st, 4));
  this->meter       = strdup((char*)sqlite3_column_text(st, 5));
  this->abc         = strdup((char*)sqlite3_column_text(st, 6));
  this->searchKey   = ABCtoPitchClass((char*)sqlite3_column_text(st, 7), this->key);
  this->ptrKey      = vec2ptr(this->searchKey);
  this->ptrIntKey   = vec2ptrInt(this->searchKey);
  this->mattWorked  = (sqlite3_column_int(st, 8) == 1 ? true : false);
  this->abc4jWorked = (sqlite3_column_int(st, 10) == 1 ? true : false);
  this->pch         = makePCH((char*)sqlite3_column_text(st, 9), fifth);
}

Tune::~Tune() {
  free(this->name);
  free(this->meter);
  free(this->abc);
  free(this->ptrKey);
  free(this->ptrIntKey);
  free(this->pch);
  delete(this->key);
}

int Tune::getId() { return this->id; }
int Tune::getSetting() { return this->setting; }
int Tune::getSearchKeySize() { return (int)(this->searchKey.size()); }
int Tune::getIntKeySize() { return this->intKeySize; }
int* Tune::getPtrKey() { return this->ptrKey; }
int* Tune::getPtrIntKey() { return this->ptrIntKey; }
float* Tune::getPCH() { return this->pch; }
std::vector<int> Tune::getSearchKey() { return this->searchKey; }
int Tune::getType(int beatOrType) {
  if (beatOrType == 1)
    return this->beattype;
  else
    return this->tunetype;
}


void Tune::print(int s) {
  printf("%d_%d\t%s [%d] (%d)\n", this->id, this->setting, this->name, s, this->tunetype);
}

char* Tune::toString() {
  char* s = (char*)malloc(150);
  snprintf(s, 150, "%s (%d)", this->name, this->tunetype);
  return s;
}
