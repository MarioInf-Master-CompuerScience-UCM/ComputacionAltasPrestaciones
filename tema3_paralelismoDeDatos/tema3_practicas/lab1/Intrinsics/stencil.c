#include "stencil.h"
#include <immintrin.h>
#include <stdio.h>

void ApplyStencil(unsigned char *img_in, unsigned char *img_out, int width, int height) {
  
	short val;
	unsigned char val_out;
	unsigned char *cast=NULL;

	for (int i = 1; i < height-1; i++){
		for (int j = 1; j < width-1; j++) {
			
			__m256i valueTemp_11 = _mm256_loadu_si256((__m256i *) &(img_in[(i-1)*width + j-1]));
			__m256i valueTemp_12 = _mm256_loadu_si256((__m256i *) &(img_in[(i-1)*width + j]));
			__m256i valueTemp_13 = _mm256_loadu_si256((__m256i *) &(img_in[(i-1)*width + j+1]));

			__m256i valueTemp_21 = _mm256_loadu_si256((__m256i *) &(img_in[(i)*width + j-1]));
			__m256i valueTemp_22 = _mm256_loadu_si256((__m256i *) &(img_in[(i)*width + j]));
			__m256i valueTemp_23 = _mm256_loadu_si256((__m256i *) &(img_in[(i)*width + j+1]));

			__m256i valueTemp_31 = _mm256_loadu_si256((__m256i *) &(img_in[(i+1)*width + j-1]));
			__m256i valueTemp_32 = _mm256_loadu_si256((__m256i *) &(img_in[(i+1)*width + j]));
			__m256i valueTemp_33 = _mm256_loadu_si256((__m256i *) &(img_in[(i+1)*width + j+1]));

			__m256i valueTemp;
			__m256i result = _mm256_set1_epi8(0);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_11);
			//cast=(unsigned char*)&valueTemp; printf("RESTA 1: %-10d", cast[0]);
			result = _mm256_adds_epu8(result, valueTemp);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_12);
			result = _mm256_adds_epu8(result, valueTemp);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_13);
			result = _mm256_adds_epu8(result, valueTemp);

			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_21);
			result = _mm256_adds_epu8(result, valueTemp);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_22);
			result = _mm256_adds_epu8(result, valueTemp);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_23);
			result = _mm256_adds_epu8(result, valueTemp);

			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_31);
			result = _mm256_adds_epu8(result, valueTemp);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_32);
			result = _mm256_adds_epu8(result, valueTemp);
			valueTemp = _mm256_subs_epu8(valueTemp_22, valueTemp_33);
			result = _mm256_adds_epu8(result, valueTemp);
			cast=(unsigned char*)&result;
			img_out[i*width + j] = *cast; 



 			//Comprobación de cálculos
			//============================================
/*  			printf("\n**CONTROL DE DATOS -> FILA=%d  COLUMNA=%d**\n",i,j);
			printf("%-4d - ", img_in[(i-1)*width + j-1]);	cast=(unsigned char*)&valueTemp_11; printf("%-10d", cast[0]);
			printf("%-4d - ", img_in[(i-1)*width + j]);		cast=(unsigned char*)&valueTemp_12;	printf("%-10d", cast[0]);
			printf("%-4d - ", img_in[(i-1)*width + j+1]);	cast=(unsigned char*)&valueTemp_13;	printf("%-10d\n", cast[0]);
			printf("%-4d - ", img_in[(i)*width + j-1]);		cast=(unsigned char*)&valueTemp_21;	printf("%-10d", cast[0]);
			printf("%-4d - ", img_in[(i)*width + j]);		cast=(unsigned char*)&valueTemp_22;	printf("%-10d", cast[0]);
			printf("%-4d - ", img_in[(i)*width + j+1]);		cast=(unsigned char*)&valueTemp_23;	printf("%-10d\n", cast[0]);
			printf("%-4d - ", img_in[(i+1)*width + j-1]);	cast=(unsigned char*)&valueTemp_31;	printf("%-10d", cast[0]);
			printf("%-4d - ", img_in[(i+1)*width + j]);		cast=(unsigned char*)&valueTemp_32;	printf("%-10d", cast[0]);
			printf("%-4d - ", img_in[(i+1)*width + j+1]);	cast=(unsigned char*)&valueTemp_33;	printf("%-10d\n", cast[0]);
			cast=(unsigned char*)&result;
			printf("Resultado calculado:%-10d\n", cast[0]);
			printf("Valor devuelto:%-10d\n", img_out[i*width + j]);  */
		}
	}

}

void CopyImage( unsigned char *img_in,  unsigned char *img_out, int width, int height) {

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img_in[i*width + j] = img_out[i*width + j];
}




//FUNCION ORIGINAL ApplyStencil
//==================================================
void ApplyStencil_original(unsigned char *img_in, unsigned char *img_out, int width, int height) {
  
	short val;
	unsigned char val_out;

	for (int i = 1; i < height-1; i++)
		for (int j = 1; j < width-1; j++) {
			val = img_in[(i  )*width + j];
			val +=	-img_in[(i-1)*width + j-1] -   img_in[(i-1)*width + j] - img_in[(i-1)*width + j+1] 
					-img_in[(i  )*width + j-1] + 7*img_in[(i  )*width + j] - img_in[(i  )*width + j+1] 
					-img_in[(i+1)*width + j-1] -   img_in[(i+1)*width + j] - img_in[(i+1)*width + j+1];
			if (val<0){
				val=0;
			}else val_out = (unsigned char)val;
			if (val>255){
				val_out=255;
			}else val_out = (unsigned char)val;

			img_out[i*width + j] = (unsigned char)(val_out);
		}
}