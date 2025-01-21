CC=icc
CFLAGS=-Wall -Wextra -std=c99 -O3 -xhost -qopt-report -fopenmp -g

LDFLAGS=-lm
NAME=heat

all: heat

heat: heat.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: 
	./$(NAME)

.PHONY: clean

clean:
	rm -rf *.o *opt* *yaml *png heat
