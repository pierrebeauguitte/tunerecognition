#ifndef KEY_H
#define KEY_H

class Key {

  // static fields initialized in .cpp
  static char* noteRange;
  static int naturals[7];
  static int modeMajor[7];
  static int modeMinor[7];
  static int modeDorian[7];
  static int modeMixolydian[7];

  char strFund;
  int fund;
  char* mode;
  int toClass[7];

 public:
  Key(char* rep);
  void print();
  int pitchClass(char c);
  int naturalPitchClass(char c);
};

#endif
