CXX=icpx
CXXFLAGS=-Wall -Wextra -O2 -g -xhost -fopenmp

LDFLAGS=-lm

all: quicksort

quicksort: quicksort.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

run:
	./quicksort
	
runTest:
	./quicksort 200000

clean:
	rm -rf quicksort *.o *opt *optrpt *yaml heat
