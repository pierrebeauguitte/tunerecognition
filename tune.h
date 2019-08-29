#ifndef TUNE_H
#define TUNE_H

#include <sqlite3.h>
#include <vector>
#include "key.h"

#define COMPOUND 0
#define SIMPLE 1
#define HORNPIPE 0
#define JIG 1
#define OTHER44 2
#define POLKA 3
#define REEL 4
#define SLIDE 5
#define SLIPJIG 6
#define WALTZ 7

class Tune {
  int id;
  int setting;
  char* name;
  int tunetype;
  int beattype;
  Key* key;
  char* meter;
  char* abc;
  bool mattWorked;
  bool abc4jWorked;
  std::vector<int> searchKey;
  int* ptrKey;
  int* ptrIntKey;
  int intKeySize;
  float* pch;

  void initTypes(char* type);
  std::vector<int> ABCtoPitchClass(char* abc, Key* k);
  float* makePCH(char* arr, bool fifth);
  float* convolve(float* hist, int n, float sigma);
  int* vec2ptr(std::vector<int> k);
  int* vec2ptrInt(std::vector<int> k);

public:

  Tune(sqlite3_stmt *st, bool fifth);
  ~Tune();

  int getId();
  int getSetting();
  int getSearchKeySize();
  int getIntKeySize();
  int* getPtrKey();
  int* getPtrIntKey();
  float* getPCH();
  std::vector<int> getSearchKey();
  int getType(int beatOrType);

  void print(int s);
  char* toString();
};

#endif
