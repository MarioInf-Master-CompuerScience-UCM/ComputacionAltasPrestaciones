
CC=icc
ICX=icx

CFLAGS=-O2 -g -qopt-report=2 -xhost

LIB=-lm -qopenmp

NAME=fibo_task

THREADS=4

NUM=50


default: tareas

tareas:
	$(CC) $(CFLAGS) $(NAME).c -o $(NAME).exe $(LIB)

run:
	./$(NAME).exe $(THREADS) $(NUM)

clean:
	rm -rf *exe *.o *.out *.optrpt *.yaml
