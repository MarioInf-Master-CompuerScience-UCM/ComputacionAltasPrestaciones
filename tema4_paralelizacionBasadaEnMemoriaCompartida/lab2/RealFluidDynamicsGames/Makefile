CC=icc
CFLAGS=-O3 -qopt-report=5 -std=c99 -Wall -Wextra -Wno-unused-parameter -qopenmp -xhost -g
LDFLAGS=

TARGETS=demo headless
SOURCES=$(shell echo *.c)
COMMON_OBJECTS=timing.o solver.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

demo_run:
	./demo

headless: headless.o $(COMMON_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

headless_run:
	./headless

clean:
	rm -f $(TARGETS) *.o *.opt.yaml *optrpt


.PHONY: clean all
