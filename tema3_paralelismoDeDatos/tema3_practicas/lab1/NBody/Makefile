NAME=nbody

CC=icc

CFLAGS=-O3 -g -qopt-report=3 -xCORE_AVX2 -I.

DEPS = gtime.h memory.h nbody.h nbody_routines.h

OBJ = gtime.o memory.o nbody.o nbody_routines.o

LIB = -lm

EXEC=nbody

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(NAME): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIB)

run: 
	$(info )
	$(info Run Nbody: )
	./$(EXEC)

clean: 
	rm -f *.o *.optrpt $(NAME)

source:
	$(shell source /opt/intel/oneapi/setvars.sh)

sourceForce:
	source /opt/intel/oneapi/setvars.sh --force