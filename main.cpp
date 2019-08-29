#include <stdio.h>
#include <sqlite3.h>
#include <vector>
#include <list>
#include <thread>
#include <mutex>
#include <queue>
#include <tuple>

#include "util.h"
#include "audio.h"
#include "tune.h"
#include "csv.h"

#define MAX_IDS 5

struct RhythmPrediction {
  int id;
  int offset;
  int predicted;
  float probs[8];
};

std::vector<Tune*> loadDB(bool fifth) {

  std::vector<Tune*> tunes;
  sqlite3 *db;
  if(sqlite3_open("corpus.db", &db)) {
    fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
    return tunes;
  }

  const char *sql = "SELECT * FROM Tunes";
  sqlite3_stmt *stmt;
  sqlite3_prepare(db, sql, -1, &stmt, NULL);
  sqlite3_step(stmt);
  while(sqlite3_column_int(stmt, 0)) {
    tunes.push_back(new Tune(stmt, fifth));
    sqlite3_step(stmt);
  }

  sqlite3_finalize(stmt);
  sqlite3_close(db);

  return tunes;
}

std::mutex mtx;

void searchThread(int t_id, std::queue<Tune*> &q,
		  std::list<std::tuple<float, Tune*, int>> &results,
		  float* &pch, int s,
		  int method, int deviation) {
  Tune* t;
  while (true) {
    mtx.lock();
    if (q.empty()) {
      mtx.unlock();
      return;
    }
    t = q.front();
    q.pop();
    mtx.unlock();
    switch (method) {
    case 0: { // original
      int dist = editDistance2(t_id, s, deviation,
			       t->getPtrKey(), t->getSearchKeySize());
      mtx.lock();
      results.push_back(std::tuple<float, Tune*, int> (dist, t, deviation));
      mtx.unlock();
      break;
    }
    case 1: { // all
      for (int shift=0; shift<12; shift++) {
	int dist = editDistance2(t_id, s, shift,
				 t->getPtrKey(), t->getSearchKeySize());
	mtx.lock();
	results.push_back(std::tuple<float, Tune*, int> (dist, t, shift));
	mtx.unlock();
      }
      break;
    }
    case 2: { // shift
      int shift = findShift(pch, t->getPCH());
      int dist = editDistance2(t_id, s, shift,
			       t->getPtrKey(), t->getSearchKeySize());
      mtx.lock();
      results.push_back(std::tuple<float, Tune*, int> (dist, t, shift));
      mtx.unlock();
      break;
    }
    case 3: {
      int dist = editDistance2(t_id, 0, deviation,
			       t->getPtrIntKey(), t->getIntKeySize());
      mtx.lock();
      results.push_back(std::tuple<float, Tune*, int> (dist, t, deviation));
      mtx.unlock();
      break;
    }
    default: // unreachable
      break;
    }
  }
}

bool isInPtr(int* refs, int id) {
  for (int* p=refs; *p != 0; p++)
    if (*p == id)
      return true;
  return false;
}

float search(char* transcriptFile,
	     char* pchFile,
	     int* real_ids,
	     std::vector<Tune*> &tunes,
	     int method,
	     FILE* f,
	     RhythmPrediction rp,
	     int rhythmPredictionType,
	     int weight_curve,
	     float threshold,
	     int thr_logic,
	     int deviation) {

  std::vector<int> notes = getQuavers(transcriptFile);
  std::vector<int> intervals;
  int origNoteSize = notes.size();
  float* pch = getPCH(pchFile);
  printf("\tTranscription read (%d notes)\n", (int)notes.size());
  if (notes.size() > 128) {
    printf("Resizing to 128");
    notes.resize(128);
  }

  
  
  if (method == 3) {
    int lastPitch = -1;
    for (int i=0; i<notes.size(); i++) {
      if (lastPitch < 0) {
	if (notes.at(i) > 0)
	  lastPitch = notes.at(i);
	continue;
      } // loops until 1st non 12 pitch found
      if (notes.at(i) < 0)
	intervals.push_back(0);
      else {
	intervals.push_back((notes.at(i) - lastPitch) % 12);
	lastPitch = notes.at(i); // lastPitch only takes non12 values
      }
    }
  }
  
  std::list<std::tuple<float, Tune*, int>> results;

  for (int i=0; i<12; i++) {
    int* ptrSpelling = spellAsPtr((method == 3 ? intervals : notes), i);
    make_B_shift(ptrSpelling, (int)notes.size(), i);
    free(ptrSpelling);
  }

  Tune *t;

  std::queue<Tune*> queue;
  bool isInSearchSpace = false;

  float threshold_c = 0.85;
  float threshold_s = 0.65;

  if (thr_logic == 0) {
    // Standard filtering, only if p_. > t
    // if (threshold == 0 || rp.probs[rp.predicted] <= threshold) {
    if ((rp.predicted == 0 && rp.probs[rp.predicted] <= threshold_c) ||
    	(rp.predicted == 1 && rp.probs[rp.predicted] <= threshold_s)) {
      for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++)
	queue.push(*tune);
      isInSearchSpace = true;
      printf("\tFull searchspace...");
    } else {
      printf("\tUsing thresholds...\n");
      for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++) {
	if ((*tune)->getType(rhythmPredictionType) == rp.predicted) {
	  queue.push(*tune);
	  if (isInPtr(real_ids, (*tune)->getId()))
	    isInSearchSpace = true;
	}
      }
      printf("\tLooking only within %d tunes of type %d\n", (int)(queue.size()), rp.predicted);
    }
  } else {
    // ArgMax(F1) way
    float p_c = rp.probs[0];
    int predType = (p_c > threshold) ? COMPOUND : SIMPLE;
    for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++) {
      if ((*tune)->getType(rhythmPredictionType) == predType) {
        queue.push(*tune);
        if (isInPtr(real_ids, (*tune)->getId()))
	  isInSearchSpace = true;
      }
    }
    printf("\tLooking only within %d tunes of type %d\n", (int)(queue.size()), rp.predicted);
  }

  // we don't care about timings in int.grid search -> skip here !
  // don't forget to comment out when timing...
  // if (!isInSearchSpace) {
  //   printf("\tNo hope of finding it...\n");
  //   if (f != NULL)
  //     fprintf(f, "-1\n");
  //   return -1;
  // }

  std::thread threads[N_THREADS];
  for (int t=0; t<N_THREADS; t++) {
    threads[t] = std::thread(searchThread, t,
			     std::ref(queue), std::ref(results),
			     std::ref(pch), (int)notes.size(),
			     method, deviation);
  }

  for (int t=0; t<N_THREADS; t++)
    threads[t].join();

  free(pch);

  if (rhythmPredictionType > 0 && threshold == 0) {
    printf("Weighing EDs with rhythm probs...\n");
    int actualType;
    for (std::list<std::tuple<float, Tune*, int>>::iterator it=results.begin();
	 it != results.end(); ++it) {
      actualType = std::get<1>(*it)->getType(rhythmPredictionType);
      switch (weight_curve) {
      case 1:
	std::get<0>(*it) *= 1-rp.probs[actualType]; break;
      case 2:
	std::get<0>(*it) *= sqrt(1-rp.probs[actualType]); break;
      case 3:
	std::get<0>(*it) *= 1 - (rp.probs[actualType] * rp.probs[actualType]); break;
      case 4:
	std::get<0>(*it) *= sqrt(1 - rp.probs[actualType] * rp.probs[actualType]); break;
      case 5:
	std::get<0>(*it) *= -log(rp.probs[actualType]); break;
      default:
	break;
      }
    }
  }

  results.sort();

  if (!isInSearchSpace) {
    // we keep this after the search to get realistic timings:
    printf("\tNo hope of finding it...\n");
    if (f != NULL)
      fprintf(f, "-1\n");
    return -1;
  }

  float s_t = 128;
  float s_f = 128;
  float lastKeptED = 128;

  bool trueMatch;
  bool atLeastOneTrue = false;
  int r = 0;
  int* printedIds = (int*)malloc(11*sizeof(int));
  for (int i=0; i<11; i++)
    printedIds[i] = 0;

  for (std::list<std::tuple<float, Tune*, int>>::iterator it=results.begin();
       it != results.end(); ++it) {
    trueMatch = isInPtr(real_ids, std::get<1>(*it)->getId());

    // float score = (std::get<0>(*it) < 0 ? 0 : 1 - std::get<0>(*it) / (int)notes.size());
    float ed = std::get<0>(*it);
    if (ed < 0) {
      printf("Negative ED, unexpected...\n");
      exit(1);
    }

    if (ed > lastKeptED) { // implies r == 10
      if (atLeastOneTrue)  // a true result in within the 10 -> job done
	break;
      // if (!trueMatch)      // we have to go on, but we can skip this (false) res.
      // 	continue;
    }

    if (trueMatch) {
      atLeastOneTrue = true;
      if (ed < s_t)
	s_t = ed;
    } else if (ed < s_f)
      s_f = ed;

    printf("\t%.5f %s ", ed, (trueMatch ? ">" : "-"));
    std::get<1>(*it)->print(std::get<2>(*it));
    char* tuneRep = std::get<1>(*it)->toString();
    if (f != NULL && !isInPtr(printedIds, std::get<1>(*it)->getId()))
      fprintf(f, "%.5f;%d;%d;%d;%d;\"%s\"\n",
	      ed,
	      std::get<1>(*it)->getId(),
	      std::get<1>(*it)->getSetting(),
	      std::get<2>(*it),
	      (trueMatch ? 1 : 0),
	      tuneRep);
    free(tuneRep);

    if (r<10 && !isInPtr(printedIds, std::get<1>(*it)->getId())) {
      printedIds[r] = std::get<1>(*it)->getId();
      printf("Added %d (%d)\n", std::get<1>(*it)->getId(), r);
      r++;
      // if (r == 10)
      // 	lastKeptED = ed;
    }
    if (r>=10 && atLeastOneTrue)
      lastKeptED = ed;
  }

  free(printedIds);

  if (f != NULL)
    fprintf(f, "%.5f;%.5f;%d\n", s_t, s_f, origNoteSize);
  return s_f - s_t;
}


int parseArguments(int argc, char* argv[],
		   int &method, int &audio_pch, int &symb_pch,
		   char* transcription, char* rhythmFile,
		   int &predType, int &weight_curve,
		   float &threshold, int &thr_logic) {

  int methodIndex = -1;

  for (int pos=1; pos<argc-1; pos++) {

    // -m <method>
    if (strcmp(argv[pos], "-m") == 0) {
      pos++;
      methodIndex = pos;
      if (strcmp(argv[pos], "original") == 0)
	method = 0;
      else if (strcmp(argv[pos], "all") == 0)
	method = 1;
      else if (strcmp(argv[pos], "interval") == 0)
	method = 3;
      else {
	method = 2;
	// shift method - scan pch options
	if (strncmp(argv[pos], "local", 5) == 0)
	  audio_pch = 1;
	else if (strncmp(argv[pos], "global", 6) == 0)
	  audio_pch = 2;
	else if (strncmp(argv[pos], "deep", 2) == 0)
	  audio_pch = 3;
	if (strstr(argv[pos], "simple") != NULL)
	  symb_pch = 1;
	else if (strstr(argv[pos], "fifth") != NULL)
	  symb_pch = 2;
      }
    }

    // -t <transcription algorithm>
    if (strcmp(argv[pos], "-t") == 0) {
      pos++;
      strcpy(transcription, argv[pos]);
    }

    // -r <rhythm prediction>
    if (strcmp(argv[pos], "-r") == 0) {
      pos++;
      strcpy(rhythmFile, argv[pos]);
      if (strstr(argv[pos], "binom") != NULL)
	predType = 1;
      else if (strstr(argv[pos], "multi") != NULL)
	predType = 2;
    }

    // -thr <threshold>
    if (strcmp(argv[pos], "-thr") == 0) {
      pos++;
      if (strcmp(argv[pos], "std") == 0)
	threshold = 1. / (predType == 1 ? 2 : 8);
      else {
	threshold = strtof(argv[pos], NULL);
	thr_logic = 1;
      }
    }

    // -c <curve=1..5>
    if (strcmp(argv[pos], "-c") == 0) {
      pos++;
      weight_curve = strtol(argv[pos], NULL, 10);
    }

  }

  return methodIndex;

  // Broken for now...
  // // adding option to search one single file
  // if (strcmp(argv[2], "file") == 0) {
  //   int* p_id = (int*)malloc(2*sizeof(int));
  //   p_id[0] = strtol(argv[4], NULL, 10);
  //   p_id[1] = 0;
  //   float a = search(argv[3], NULL, p_id, tunes, 0, NULL, "compound", 0);

  //   free(p_id);
  //   for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++)
  //     delete (*tune);

  //   return 0;
  // }

}

void readRhythmPrediction(char* rhythmFile, int predType,
			  RhythmPrediction (&predictions)[2000],
			  int (&predictionIds)[500][4]) {
  int predCount[500];
  for (int i=0; i<500; i++) predCount[i] = 0;

  int chunkId;
  int chunkOffset;
  int predicted;
  float probs[8];

  int pcount = 0;

  FILE *fp;

  fp = fopen(rhythmFile, "r");

  int nPrediction = (predType == 1) ? 2 : 8;

  int nLines = 0;
  while (true) {
    if (fscanf(fp, "%d.%d ", &chunkId, &chunkOffset) < 0)
      break;
    for (int p=0; p<nPrediction; p++)
      fscanf(fp, "%f ", &(probs[p]));
    fscanf(fp, "%d\n", &predicted);
    nLines++;
    struct RhythmPrediction rp;
    rp.id = chunkId;
    rp.offset = chunkOffset;
    rp.predicted = predicted;
    memcpy(rp.probs, probs, sizeof(probs));
    predictions[pcount] = rp;
    predictionIds[chunkId-1][predCount[chunkId-1]] = pcount;
    predCount[chunkId-1] += 1;
    pcount++;
  }
  printf("Read %d lines\n", nLines);

  fclose(fp);
}

int main(int argc, char* argv[]) {

  int method;
  int audio_pch = 0;
  int symb_pch = 0;
  char amt[50];
  char rhythmFile[100];
  int predType = 0;
  int weight_curve = 0;
  float threshold = 0;
  int thr_logic = 0;

  int methodIndex = parseArguments(argc, argv, method, audio_pch, symb_pch, amt,
				   rhythmFile, predType, weight_curve, threshold, thr_logic);
  printf("search init with %d/%d/%d and %s\n",
	 method, audio_pch, symb_pch, amt);
  if (predType > 0)
    printf("Using rhythm predictions %s [%d], threshold %.3f, curve %d\n",
	   rhythmFile, predType, threshold, weight_curve);

  RhythmPrediction prediction[2000];
  int predictionIds[500][4] = {};
  for (int i=0; i<500; i++)
    predictionIds[i][0] = -1;

  if (predType > 0)
    readRhythmPrediction(rhythmFile, predType, prediction, predictionIds);

  // loadDB
  std::vector<Tune*> tunes = loadDB((symb_pch == 2));
  printf("%d tunes loaded.\n", (int)tunes.size());

  // // code to output symbPCH...
  // for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++) {
  //   if ((*tune)->getSetting() == 2244) {
  //     float* pch = (*tune)->getPCH();
  //     for (int i=0; i<120; i++)
  // 	printf("%f,", pch[i]);
  //   }
  // }
  // return 0;

  init_bitmasks();

  int nSearches = 0;
  int nSuccesses = 0;

  io::CSVReader<7, io::trim_chars<' '>,
		io::double_quote_escape<',','\"'>> in("dataset.csv");
  in.read_header(io::ignore_extra_column, "index", "id", "t1", "t2", "t3", "t4", "key deviation");

  int index;
  char* ids;
  int t1, t2, t3, t4;
  int deviation;
  char* p;
  RhythmPrediction rp;

  char notes[1000];
  char pch[1000];
  char outfile[1000];

  while(in.read_row(index, ids, t1, t2, t3, t4, deviation)){

    p = ids;
    int* p_ids = (int*)malloc((MAX_IDS + 1) * sizeof(int));
    int c = 0;
    while (true) {
      int i = strtol(p, &p, 10);
      p_ids[c++] = i;
      if (*p == '\0') {
	p_ids[c] = 0;
	break;
      }
      p++;
    }

    if (predType > 0 && threshold > 0) {
      if (predictionIds[index-1][0] < 0) {
	printf("Skipping index %d\n", index);
	free(p_ids);
	continue;
      }
    }

    int offsets[4] = {t1, t2, t3, t4};

    for (int t=0; t<4; t++) {

      if (predType > 0) {
	for (int i=0; i<4; i++) {
	  if (prediction[predictionIds[index-1][i]].offset == offsets[t]) {
	    rp = prediction[predictionIds[index-1][i]];
	    break;
	  }
	}

	sprintf(outfile, "results/%s.%s.0.85.0.65/%03d.%d.%s.csv",
		amt, ((predType == 1) ? "binomial" : "multinomial"),
		index, offsets[t],
		argv[methodIndex]);

	// if (threshold > 0)
	//   sprintf(outfile, "results/%s.%s.%.3f/%03d.%d.%s.csv",
	// 	  amt, ((predType == 1) ? "binomial" : "multinomial"),
	// 	  threshold, index, offsets[t],
	// 	  argv[methodIndex]);
	// else
	//   sprintf(outfile, "results/%s.%s.%d/%03d.%d.%s.csv",
	// 	  amt, ((predType == 1) ? "binomial" : "multinomial"),
	// 	  weight_curve, index, offsets[t],
	// 	  argv[methodIndex]);
      } else {
	sprintf(outfile, "results/%s/%03d.%d.%s.csv",
		amt, index, offsets[t], argv[methodIndex]);
      }

      FILE *f;
      if (f = fopen(outfile, "r")) {
	printf("%s already exists, skipping...\n", outfile);
	fclose(f);
	continue;
      }
      nSearches++;
      sprintf(notes, "transcriptions/%s/%03d.%d.notes.csv", amt, index, offsets[t]);
      sprintf(pch, "transcriptions/pch/%03d.%d.%s.json", index, offsets[t],
	      (audio_pch == 1 ? "local" :
	       (audio_pch == 2 ? "global" : "deepChroma")));
      printf("Searching %d.%d [%s]...\n", index, offsets[t], ids);
      f = fopen(outfile, "w");
      printf("Write to %s\n", outfile);
      float a = search(notes, pch, p_ids, tunes, method,
		       f, rp, predType, weight_curve, threshold, thr_logic, deviation);
      printf("Search finished with margin %.3f\n--------------------\n", a);
      fclose(f);
      if (a>0)
	nSuccesses++;
    }
    free(p_ids);
  }

  printf("\n---\n%d/%d successful searches\n---\n", nSuccesses, nSearches);

  for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++)
    delete (*tune);
}
