CC=icpx
CFLAGS= -fiopenmp -fopenmp-targets=spir64 -qopt-report=1
LDFLAGS=

export  OMP_TARGET_OFFLOAD=MANDATORY 
export 	LIBOMPTARGET_DEVICETYPE=gpu
export	LIBOMPTARGET_PLUGIN=OPENCL

all: simple

simple: simple.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

run: 

	./simple

runMon:
	LIBOMPTARGET_PLUGIN_PROFILE=T ./simple

clean:
	rm -f simple *.o *.opt.yaml



