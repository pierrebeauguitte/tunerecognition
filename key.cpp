#include "key.h"
#include <stdio.h>
#include <string.h>

char* Key::noteRange = (char*)"CDEFGAB";
int Key::naturals[7] = {0, 2, 4, 5, 7, 9, 11};
int Key::modeMajor[7] = {2, 2, 1, 2, 2, 2, 1};
int Key::modeMinor[7] = {2, 1, 2, 2, 1, 2, 2};
int Key::modeDorian[7] = {2, 1, 2, 2, 2, 1, 2};
int Key::modeMixolydian[7] = {2, 2, 1, 2, 2, 1, 2};

int indexOf(char* str, char el) {
  char *ptr = strchr(str, el);
  if (!ptr)
    return -1;
  return ptr - str;
}

Key::Key(char* rep) {
  this->strFund = rep[0];
  int index = indexOf(noteRange, rep[0]);
  this->fund = naturals[index];
  this->mode = (++rep);

  int *scale;
  if (strcmp(this->mode, "major") == 0)
    scale = modeMajor;
  else if (strcmp(this->mode, "minor") == 0)
    scale = modeMinor;
  else if (strcmp(this->mode, "dorian") == 0)
    scale = modeDorian;
  else if (strcmp(this->mode, "mixolydian") == 0)
    scale = modeMixolydian;

  int acc = 0;
  for (int d=0; d<7; d++) {
    this->toClass[(index + d) % 7] = (this->fund + acc) % 12;
    acc += scale[d];
  }
}

void Key::print() {
  printf("%c [%d], %s\n[", this->strFund, this->fund, this->mode);
  for (int d=0; d<7; d++)
    printf("%d.", this->toClass[d]);
  printf("]\n");
}

int Key::pitchClass(char c) {
  int index = indexOf(noteRange, c);
  if (index < 0)
    return -1;
  return this->toClass[index];
}

int Key::naturalPitchClass(char c) {
  int index = indexOf(noteRange, c);
  if (index < 0)
    return -1;
  return naturals[index];
}
