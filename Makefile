CC = g++

searchTune: main.cpp tune.o key.o audio.h util.h
	$(CC) -pthread main.cpp tune.o key.o -o $@ -l sqlite3

checkShift: checkShift.cpp tune.o key.o audio.h util.h
	$(CC) -pthread checkShift.cpp tune.o key.o -o $@ -l sqlite3

tune.o: tune.cpp tune.h util.h key.o
	$(CC) -c tune.cpp

key.o: key.cpp key.h
	$(CC) -c key.cpp

clean:
	rm searchTune *.o
