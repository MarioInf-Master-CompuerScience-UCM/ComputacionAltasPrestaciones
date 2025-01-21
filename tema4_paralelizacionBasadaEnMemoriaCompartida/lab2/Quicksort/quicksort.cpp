#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <stdio.h>

using namespace std;

void init(int arr[], int n){
	srand(0);
	#pragma omp parallel for schedule(guided, 8) shared(arr)
	#pragma ivdep
	for(int i=0;i<n;i++){
		arr[i]=rand()%1234 + 1;
	}
}

// PARTITION
// Versión original, no macha el contrenido y el resulatdo esta ordenado, pro no aplica paralelziación.
// Es la versión más reapida debido a que no se producen esperar de sincronización.
int partition(int arr[], int low, int high, int pivot){
	int j = low, i=low;
	int temp=0;

	for(i=low;i<=high-1;i++){
		temp = arr[i];
		if(temp<=pivot){
			arr[i] = arr[j];
			arr[j] = temp;
			j++;
		}
	}
	temp=arr[high];
	arr[high] = arr[j];
	arr[j] = temp;
	j++;
	return j-1;
}


// PARTITION
// Versión apralelizada de partition que cumple la condición de ordenar los valores de menor a mayor pero que machaca
// el contenido de algunas posiciones.
// Es la versión más lenta debido a que no se producen esperas de sincronización.
/* int partition(int arr[], int low, int high, int pivot){
	int j = low, i=low;
	int temp=0;

	#pragma omp parallel for private(i) schedule(guided) firstprivate(low, high, pivot, temp) shared(j, arr)
	for(i=low;i<=high-1;i++){
		temp = arr[i];
		if(temp<=pivot){
			#pragma omp critical(CSpactition_swap)
			{
				arr[i] = arr[j];
				arr[j] = temp;
				j++;
			}
		}
	}
	temp=arr[high];
	arr[high] = arr[j];
	arr[j] = temp;
	j++;
	return j-1;
} */



// PARTITION
// Intento de realziar la versión de partition apta para la paralelizacion sin machacar valores
// El algoritmo resulta totalmente ineficiente respectoa  asu versión sin apralizar
/* int partition(int arr[], int low, int high, int pivot){
	int j = low, numMayores=0;
	int temp=0, i=0;
	int *arr_sec = NULL;
	int *arr_mayores = NULL;
	arr_sec=(int *)calloc(high-low+1, sizeof(int));
	arr_mayores=(int *)calloc(high-low+1, sizeof(int));
 	for(i=low; i<=high; i++){
		arr_sec[i-low]=arr[i];	
	}

	#pragma omp parallel for \
	schedule(guided) \
	private(i, temp) \
	firstprivate(pivot) \
	shared(j, arr, arr_sec, low, high, numMayores)
	for(i=low; i<high; i++){
		temp=arr_sec[i-low];
		if(temp<=pivot){
			#pragma omp critical(CSpactition_swap)
			{
				arr[i] = arr[j];
				arr[j] = temp;
				j++;
			}
		}else{
			#pragma omp critical(CSpactition_mayores)
			{
				arr_mayores[numMayores]=temp;
				numMayores++;
			}
		}
	}

	temp=arr[high];
	arr[high] = arr[j];
	arr[j] = temp;
	j++;
	for(i=0 ; i<numMayores ; i++){
		arr[j]=arr_mayores[i];
		j++;
	}

	//fprintf(stderr,"low: %d    high: %d    num: %d    pt: %p \n",low, high,num, arr_sec);
	free(arr_sec);
	free(arr_mayores);

	return j-1;
}
 */

		
void quickSort(int arr[], int low, int high){
	if(low < high){
		int pivot = arr[high];
		int pos = partition(arr, low, high, pivot);

		#pragma omp task
		{
			quickSort(arr, low, pos-1);

		}
		#pragma omp task
		{
			quickSort(arr, pos+1, high);

		}
		#pragma omp taskwait
	}
	return;
}

bool checkFn(int * arr,int n){
	
	bool flag=true;
	
	#pragma omp parallel for schedule(guided) reduction(&:flag)
	#pragma ivdep
	for(int i=0;i<n-1;i++){
		if(arr[i]>arr[i+1]){
			cout << "array[" << i << "] > array[" << i+1 << "]" << endl;
			flag=false;
		}
	}
	return flag;
}

int main(int argc, char *argv[]){
	int n;
	int *arr;
	double t1,t2;
	int debug=0;
	if(argc < 2){
		n=6000000;
	}else{
		n=atoi(argv[1]);
	}
	arr = new int[n];
	init(arr, n);

	t1=omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single 
		quickSort(arr, 0 , n-1);
	}
	t2=omp_get_wtime()-t1;

	cout << "quicksort took " << t2 << " sec. to complete" << endl;
	if (!checkFn(arr, n)) {
		cout << "validation failed!" << endl;
	}

	if (debug) {
		cout<<"The sorted array is: ";
		for( int i = 0 ; i < n; i++){
			cout<< arr[i]<<" ";
		}
		cout << endl;
	}

	delete [] arr;
	
}
