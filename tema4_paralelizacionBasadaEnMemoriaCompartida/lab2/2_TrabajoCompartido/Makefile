
var=1000


default:
	icc -o prime.icc prime.c -qopenmp -qopt-report

run:
	./prime.icc $(var)

clean:
	rm -rf *.o *opt *yaml heat
