#include <stdio.h>
#include <math.h>
#include <omp.h>

double f(double x);    /* Function we're integrating */
double Trap_secuentual(double a, double b, int n, double h);
double Trap_atomic(double a, double b, int n, double h);
double Trap_critical(double a, double b, int n, double h);
double Trap_reduction(double a, double b, int n, double h);
void printResults(double a, double b, double n, double start, double stop, double integralSecuential, double integral);

/**
 * Second argument indicate the version of TrapFunction that we are goint to use. Possible values:
 * 	0: Default, Trap_secuentual 
 * 	1: Trap_atomic
 * 	2: Trap_critical 
 * 	3: Trap_reduction
 * 	4: All executions
 */
int main(int argc, char **argv) {
	unsigned short versionTrap=0;
	double  integral=0;	/* Store result in integral   */
	double  integralSecuential=-1;	/* Store result in secuential integral   */
	int     n=10000000;	/* Number of trapezoids       */
	const double  a=0.0, b=1.0;	/* Left and right endpoints   */
	const double  h= (b-a)/n;	/* Height of trapezoids       */

	if (argc==2){
		n = atoi(argv[1]);
	}else if(argc==3){
		n = atoi(argv[1]);
		versionTrap = atoi(argv[2]);
		if(versionTrap<0 || versionTrap>4){
			versionTrap=0;
		}
	} 
	else if(argc>2){
		printf("./exec num_traps\n");
		exit(-1);
	}
	
	if(versionTrap==0 || versionTrap==4){
		printf("\nRESULT OPTION 0 - SECUENTIAL\n");
		double start = omp_get_wtime();
		integralSecuential = Trap_secuentual(a, b, n, h);
		double stop = omp_get_wtime();
		printResults(a, b, n, start, stop, integralSecuential, integralSecuential);
	}


	if(versionTrap==1 || versionTrap==4){
		printf("\nRESULT OPTION 1 - ATOMIC\n");
		double start = omp_get_wtime();
		integral = Trap_atomic(a, b, n, h);
		double stop = omp_get_wtime();
		printResults(a, b, n, start, stop, integralSecuential, integral);
	}
		


	if(versionTrap==2 || versionTrap==4){
		printf("\nRESULT OPTION 2 - CRITICAL\n");
		double start = omp_get_wtime();
		integral = Trap_critical(a, b, n, h);
		double stop = omp_get_wtime();
		printResults(a, b, n, start, stop, integralSecuential, integral);
	}


	if(versionTrap==3 || versionTrap==4){
		printf("\nRESULT OPTION 3 - REDUCTION\n");
		double start = omp_get_wtime();
		integral = Trap_reduction(a, b, n, h);
		double stop = omp_get_wtime();
		printResults(a, b, n, start, stop, integralSecuential, integral);
	}


	return 0;
}



/*------------------------------------------------------------------
 * Function:    Trap
 * Purpose:     Estimate integral from a to b of f using trap rule anqd
 *              n trapezoids
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double Trap_secuentual(double a, double b, int n, double h) {
	double integral=0.0, area=0.0;
	int k;

	printf("Parameters: %.15f  %.15f  %.15f  %.15f\n", a, b, n, h);
	for (k = 1; k <= n; k++) {
		area = h*(f(a+k*h)+f(a+(k-1)*h))/2.0;
		integral+=area;
	}
	return integral;
}


double Trap_atomic(double a, double b, int n, double h) {
	double integral=0.0, integralThread=0.0, area=0.0;
	int idThread=0, numThreads=0, iterThread=0, K_Init=0, K_End=0, k=0;
	long countThreads=0;

	#pragma omp parallel \
		private(idThread, numThreads, iterThread, K_Init, K_End, area, integralThread, k) \
		firstprivate(a, b, n, h) \
		shared(integral) \
		reduction(+ : countThreads)
	{
		countThreads=1;
		idThread=omp_get_thread_num();
		numThreads=omp_get_num_threads();
		iterThread= n/numThreads;
		K_Init = idThread*iterThread;
		K_End = K_Init+iterThread-1;

		area=0.0;
		integralThread=0.0;
		for (k = K_Init; k <= K_End; k++) {
			area = h*(f(a+k*h)+f(a+(k-1)*h))/2.0;
			integralThread+=area;
		}
		
		#pragma omp atomic
			integral+=integralThread;
		
	}
	printf("Num of threads: %d \n", countThreads);
	return integral;
}


double Trap_critical(double a, double b, int n, double h) {
	double integral=0.0, integralThread=0.0, area=0.0;
	int idThread=0, numThreads=0, iterThread=0, K_Init=0, K_End=0, k=0;
	unsigned short countThreads=0;
	
	#pragma omp parallel \
		private(idThread, numThreads, iterThread, K_Init, K_End, area, integralThread, k) \
		firstprivate(a, b, n, h) \
		shared(integral) \
		reduction(+ : countThreads)
	{
		countThreads=1;
		idThread=omp_get_thread_num();
		numThreads=omp_get_num_threads();
		iterThread= n/numThreads;
		K_Init = idThread*iterThread;
		K_End = K_Init+iterThread-1;

		area=0.0;
		integralThread=0.0;
		for (k = K_Init; k <= K_End; k++) {
			area = h*(f(a+k*h)+f(a+(k-1)*h))/2.0;
			integralThread+=area;
		}
		
		#pragma omp critical (criticalSection)
		{
			integral+=integralThread;
		}
	}
	printf("Num of threads: %d \n", countThreads);
	return integral;
}


double Trap_reduction(double a, double b, int n, double h) {
	double integral=0.0, area=0.0;
	int k;
	unsigned short countThreads=0;
	
	#pragma omp parallel for num_threads(omp_get_max_threads()) firstprivate(area) reduction(+ : integral) reduction(+ : countThreads)
	for (k = 1; k <= n; k++) {
		countThreads=1;
		area = h*(f(a+k*h)+f(a+(k-1)*h))/2.0;
		integral+=area;
	}
	printf("Num of threads: %d \n", countThreads);
	return integral;
}



/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x) {
	double return_val;

	return_val = 1.0/(1.0+exp(x*x-4*x-10.0)+sin(x/3.14));
	return return_val;
}  /* f */


void printResults(double a, double b, double n, double start, double stop, double integralSecuential, double integral){
	printf("With n = %d trapezoids\n", n);
	printf("Computed in %f s.\n", stop-start);
	printf("Integral from %f to %f = %.15f",a, b, integral);
	if(integralSecuential!=-1 && integral==integralSecuential){
		printf("-> CORRECT!!\n");
	}else if(integralSecuential!=-1 && integral!=integralSecuential){
		printf("-> ERROR!!\n");
	}
	printf("\n\n");
}