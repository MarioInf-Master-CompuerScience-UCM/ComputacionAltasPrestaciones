CC=icx
CFLAGS=-O2 -std=c99 -fiopenmp -fopenmp-targets=spir64
LDFLAGS=-lpng -lm

IMAGEN=imgs/lenna.png
MENSAJE=imgs/logo_topsecret.png
RESULTADO=img_result.png

all: steano


steano: main.o steano_routines.o io_routines.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)


run:
	LIBOMPTARGET_DEBUG=2 ./steano -g $(IMAGEN) $(MENSAJE) $(RESULTADO)


clean:
	rm -f steano *.o *.opt.yaml *.png


.PHONY: clean all
