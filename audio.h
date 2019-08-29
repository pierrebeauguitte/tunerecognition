#ifndef AUDIO_H
#define AUDIO_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

struct note {
  float onset;
  float duration;
  float midiNote;
};

struct bin {
  float center;
  int count;
};

std::vector<note> readNotes(char* path) {

  std::vector<note> notes;

  FILE *fp;
  float onset, duration, midiNote;

  fp = fopen(path, "r");
  if (fp == NULL)
    return notes;

  // disable next line for MATT
  fscanf (fp, "%f\n", &onset);
  while (fscanf (fp, "%f, %f, %f\n", &onset, &duration, &midiNote) > 0)
    notes.push_back({onset, duration, midiNote});

  fclose(fp);

  return notes;
}

float getQuaverDuration(std::vector<note> notes) {
  std::vector<bin> bins;
  bool found;
  for (std::vector<note>::iterator note = notes.begin() ; note != notes.end(); ++note) {
    found = false;
    for (std::vector<bin>::iterator bin = bins.begin() ; bin != bins.end(); ++bin) {
      if (note->duration >= (bin->center * 2./3.) &&
	  note->duration <= (bin->center * 4./3.)) {
	found = true;
	bin->center = (bin->center * bin->count + note->duration) / float(bin->count + 1);
	bin->count += 1;
	break;
      }	
    }
    if (!found)
      bins.push_back({note->duration, 1});
  }
  int maxCount = 0;
  float quaverDuration;
  for (std::vector<bin>::iterator bin = bins.begin() ; bin != bins.end(); ++bin) {
    // if (bin->count > maxCount && bin->center >= 0.1) { // for MATT
    if (bin->count > maxCount) {
      maxCount = bin->count;
      quaverDuration = bin->center;
    }
  }

  bins.clear();
  bins.shrink_to_fit();

  // for MATT
  /* if (maxCount == 0) */
  /*   quaverDuration = 0.1; */
  /* printf("quaver: %.3f\n", quaverDuration); */
  return quaverDuration;
}

std::vector<int> getQuavers(char* path) {
  std::vector<note> notes = readNotes(path);
  float quaver = getQuaverDuration(notes);
  std::vector<int> qNotes;

  std::vector<note>::iterator previous = notes.end();

  for (std::vector<note>::iterator note = notes.begin() ; note != notes.end(); ++note) {
    if (previous != notes.end()) {
      float gap = note->onset - (previous->onset + previous->duration);
      int nRests = rint(gap / quaver);
      for (int i=0; i<nRests; i++)
	qNotes.push_back(-1);
    }
    int nQuavers = rint(note->duration / quaver);
    for (int i=0; i<nQuavers; i++)
      qNotes.push_back((int)(note->midiNote + 0.5)); // round MIDI pitch
    previous = note;
  }

  notes.clear();
  notes.shrink_to_fit();
  
  return qNotes;
}

float* getPCH(char* path) {

  if (path == NULL)
    return NULL;

  FILE *fp;
  fp = fopen(path, "r");
  if (fp == NULL) {
    printf("\033[1;31mNo file (%s)!\033[0m\n", path);
    return NULL;
  }

  float *pch = (float*)malloc(120 * sizeof(float));
  char *line = (char*)malloc(3000);
  char *p;
  size_t len = 0;

  fgets(line, 3000, fp);
  p = line;

  for (int i=0; i<120; i++) {
    p++;
    pch[i] = strtof(p, &p);
  }

  free(line);
  fclose(fp);

  return pch;
}

#endif
