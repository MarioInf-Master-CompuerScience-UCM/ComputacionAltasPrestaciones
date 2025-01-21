
CC=icc
ICX=icx

CFLAGS=-O2 -g -qopt-report=2 -xhost

LIB=-lm -qopenmp

NAME=trapezoidal

NUM=10000000

VERSION=4



default: trapezoidal

trapezoidal:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).exe $(LIB)

run:
	./$(NAME).exe $(NUM) $(VERSION)

clean:
	rm -rf *exe *.o *.out *.optrpt *.yaml
