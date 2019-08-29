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
      // int dist = editDistance2(t_id, s, shift,
      // 			       t->getPtrKey(), t->getSearchKeySize());
      int dist = 0;
      mtx.lock();
      results.push_back(std::tuple<float, Tune*, int> (dist, t, shift));
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
	     int deviation) {

  float* pch = getPCH(pchFile);

  std::list<std::tuple<float, Tune*, int>> results;

  Tune *t;

  std::queue<Tune*> queue;

  // Only add the right tune here...
  for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++)
    if (isInPtr(real_ids, (*tune)->getId()))
      queue.push(*tune);

  std::thread threads[N_THREADS];
  for (int t=0; t<N_THREADS; t++) {
    threads[t] = std::thread(searchThread, t,
			     std::ref(queue), std::ref(results),
			     std::ref(pch), 42,
			     method, deviation);
  }

  for (int t=0; t<N_THREADS; t++)
    threads[t].join();

  free(pch);

  for (std::list<std::tuple<float, Tune*, int>>::iterator it=results.begin();
       it != results.end(); ++it) {

    if (std::get<2>(*it) == deviation)
      return 1;
  }

  return 0;
}


int parseArguments(int argc, char* argv[], int &method,
		   int &audio_pch, int &symb_pch, char* transcription) {

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
  }

  return methodIndex;

}

int main(int argc, char* argv[]) {

  int method;
  int audio_pch = 0;
  int symb_pch = 0;
  char amt[50];

  int methodIndex = parseArguments(argc, argv, method, audio_pch, symb_pch, amt);
  printf("search init with %d/%d/%d and %s\n",
	 method, audio_pch, symb_pch, amt);

  // loadDB
  std::vector<Tune*> tunes = loadDB((symb_pch == 2));
  printf("%d tunes loaded.\n", (int)tunes.size());

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

  char notes[1000];
  char pch[1000];

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

    int offsets[4] = {t1, t2, t3, t4};

    for (int t=0; t<4; t++) {
      nSearches++;
      sprintf(notes, "transcriptions/%s/%03d.%d.notes.csv", amt, index, offsets[t]);
      sprintf(pch, "transcriptions/pch/%03d.%d.%s.json", index, offsets[t],
	      (audio_pch == 1 ? "local" :
	       (audio_pch == 2 ? "global" : "HPCP")));
      float a = search(notes, pch, p_ids, tunes, method,
		       NULL, deviation);
      if (a>0)
	nSuccesses++;
    }

    free(p_ids);
  }

  printf("\n---\n%d/%d successful searches\n---\n", nSuccesses, nSearches);

  for (std::vector<Tune*>::iterator tune = tunes.begin() ; tune != tunes.end(); tune++)
    delete (*tune);
}
