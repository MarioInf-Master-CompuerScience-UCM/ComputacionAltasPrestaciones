
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>

#include "io_routines.h"
#include "steano_routines.h"

# define M_PI           3.14159265358979323846  /* pi */

	#pragma omp declare target
	//Called in encoder and decodrer
	void im2imRGB(uint8_t *im, int w, int h, t_sRGB *imRGB){

		imRGB->w = w;
		imRGB->h = h;
		int i, j;

		#pragma omp target teams distribute parallel for collapse(2) private(i,j)
		for (i=0; i<h; i++)
			for (j=0; j<w; j++)
			{
				imRGB->R[i*w+j] = im[3*(i*w+j)  ];
				imRGB->G[i*w+j] = im[3*(i*w+j)+1];  
				imRGB->B[i*w+j] = im[3*(i*w+j)+2];    
			}                    
	}

	//Called in encoder
	void imRGB2im(t_sRGB *imRGB, uint8_t *im, int *w, int *h)
	{
		int w_ = imRGB->w;
		int i, j;
		*w = imRGB->w;
		*h = imRGB->h;

		#pragma omp target teams distribute parallel for collapse(2) private(i,j)
		for (i=0; i<*h; i++)
			for (j=0; j<*w; j++)
			{
				im[3*(i*w_+j)  ] = imRGB->R[i*w_+j];
				im[3*(i*w_+j)+1] = imRGB->G[i*w_+j];  
				im[3*(i*w_+j)+2] = imRGB->B[i*w_+j];    
			}                    
	}

	//funtion for translate from RGB to YCbCr. Called un encoder an decoder
	void rgb2ycbcr(t_sRGB *in, t_sYCrCb *out)
	{

		int w = in->w;
		int i, j;
		out->w = in->w;
		out->h = in->h;

		#pragma omp target teams distribute parallel for collapse(2) private(i,j)
		for (i = 0; i < in->h; i++) {
			for (j = 0; j < in->w; j++) {
				out->Y[i*w+j]  =         0.299*in->R[i*w+j]     + 0.587*in->G[i*w+j]      + 0.114*in->B[i*w+j];
				out->Cr[i*w+j] = 128.0 - 0.168736*in->R[i*w+j]  - 0.3331264*in->G[i*w+j]  + 0.5*in->B[i*w+j] ;
				out->Cb[i*w+j] = 128.0 + 0.5*in->R[i*w+j]       - 0.418688*in->G[i*w+j]   - 0.081312*in->B[i*w+j];
			}
		}

	}

	//function for translate YCbCr to RGB. Called in encoder
	void ycbcr2rgb(t_sYCrCb *in, t_sRGB *out){

		int w = in->w;
		int i, j;
		out->w = in->w;
		out->h = in->h;

		#pragma omp target teams distribute parallel for collapse(2) private(i,j)
		for (i = 0; i < in->h; i++) {
			for (j = 0; j < in->w; j++) {

				// Use standard coeficient
				out->R[i*w+j] = in->Y[i*w+j]                                 + 1.402*(in->Cb[i*w+j]-128.0);
				out->G[i*w+j] = in->Y[i*w+j] - 0.34414*(in->Cr[i*w+j]-128.0) - 0.71414*(in->Cb[i*w+j]-128.0); 
				out->B[i*w+j] = in->Y[i*w+j] + 1.772*(in->Cr[i*w+j]-128.0);
				
				// After translate we must check if RGB component is in [0...255]
				if (out->R[i*w+j] < 0){
					out->R[i*w+j] = 0;
				}else if (out->R[i*w+j] > 255){
					out->R[i*w+j] = 255;
				}

				if (out->G[i*w+j] < 0){
					out->G[i*w+j] = 0;
				}else if (out->G[i*w+j] > 255){
					out->G[i*w+j] = 255;
				}

				if (out->B[i*w+j] < 0){
					out->B[i*w+j]= 0;
				}else if (out->B[i*w+j] > 255){
					out->B[i*w+j] = 255;
				}
			}
		}
	}

	//Called in encoder and decoder
	void get_dct8x8_params(float *mcosine, float *alpha)
	{
		int bM = 8;
		int bN = 8;
		int i, j;

		#pragma omp target teams distribute parallel for collapse(2) private(i,j)
		for (i = 0; i < bM; i++)
			for (j = 0; j < bN; j++)
				mcosine[i*bN+j] = cos(((2*i+1)*M_PI*j)/(2*bM));

		alpha[0] = 1 / sqrt(bM * 1.0f);
		#pragma omp target teams distribute parallel for private(i)
		for (i = 1; i < bM; i++)
			alpha[i] = sqrt(2.0f) / sqrt(bM * 1.0f);
	}


	//function for DCT. Picture divide block size 8x8. Called in encoder and decoder
	void dct8x8_2d(float *in, float *out, int width, int height, float *mcosine, float *alpha)
	{
		int bM=8;
		int bN=8;

		#pragma omp target teams distribute parallel for collapse(6)
		for(int bi=0; bi<height/bM; bi++)
		{
			int stride_i = bi * bM;
			for(int bj=0; bj<width/bN; bj++)
			{
				int stride_j = bj * bN;
				for (int i=0; i<bM; i++)
				{
					for (int j=0; j<bN; j++)
					{
						float tmp = 0.0;
						for (int ii=0; ii < bM; ii++) 
						{
							for (int jj=0; jj < bN; jj++)
								tmp += in[(stride_i+ii)*width + stride_j+jj] * mcosine[ii*bN+i]*mcosine[jj*bN+j];
						}
						out[(stride_i+i)*width + stride_j+j] = tmp*alpha[i]*alpha[j];
					}
				}
			}
		}
	}

	//Called in encoder
	void idct8x8_2d(float *in, float *out, int width, int height, float *mcosine, float *alpha)
	{
		int bM=8;
		int bN=8;

		#pragma omp target teams distribute parallel for collapse(6)
		for(int bi=0; bi<height/bM; bi++)
		{
			int stride_i = bi * bM;
			for(int bj=0; bj<width/bN; bj++)
			{
				int stride_j = bj * bN;
				for (int i=0; i<bM; i++)
				{
					for (int j=0; j<bN; j++)
					{
						float tmp = 0.0;
						for (int ii=0; ii < bM; ii++) 
						{
							for (int jj=0; jj < bN; jj++)
								tmp += in[(stride_i+ii)*width + stride_j+jj] * mcosine[i*bN+ii]*mcosine[j*bN+jj]*alpha[ii]*alpha[jj];
						}
						out[(stride_i+i)*width + stride_j+j] = tmp;
					}
				}
			}
		}
	}

	//Called in encoder
	void insert_msg(float *img, int width, int height, char *msg, int msg_length)
	{
		int i_insert=3;
		int j_insert=4;

		int bM=8;
		int bN=8;
			
		int bsI = height/bM;
		int bsJ = width/bN;
		int bi = 0;
		int bj = 0;
		
		if(bsI*bsJ<msg_length*8)
			printf("Image not enough to save message!!!\n");

		#pragma omp target teams distribute parallel for collapse(2) shared(bj, bi) 
		for(int c=0; c<msg_length; c++)
			for(int b=0; b<8; b++)
			{
				char ch = msg[c];
				char bit = (ch&(1<<b))>>b;
				
				int stride_i = bi * bM;
				int stride_j = bj * bN;
				float tmp = 0.0;
				for (int ii=0; ii < bM; ii++) 
				{
					for (int jj=0; jj < bN; jj++)
						tmp += img[(stride_i+ii)*width + stride_j+jj];
				}
				float mean = tmp/(bM*bN);
				
	//			img[(bi+i_insert)*width + bj+j_insert] = (float)(bit)*img[(bi+i_insert)*width + bj+j_insert];

				if (bit) 
					img[(stride_i+i_insert)*width + stride_j+j_insert] = fabsf(mean); //+
				else
					img[(stride_i+i_insert)*width + stride_j+j_insert] = -1.0f*fabsf(mean); //-

					bj++;
					if (bj>=bsJ){
						bj=0;
						bi++;
					}

			}
	}

	//Called in decoder
	void extract_msg(float *img, int width, int height, char *msg, int msg_length)
	{
		int i_insert=3;
		int j_insert=4;

		int bM=8;
		int bN=8;
			
		int bsI = height/bM;
		int bsJ = width/bN;
		int bi = 0;
		int bj = 0;
		
		#pragma omp target teams distribute parallel for collapse(2) shared(bj, bi)
		for(int c=0; c<msg_length; c++){
			char ch=0;
			for(int b=0; b<8; b++){
				int bit; 

				int stride_i = bi * bM;
				int stride_j = bj * bN;
				float tmp = 0.0;
				for (int ii=0; ii < bM; ii++) 
				{
					for (int jj=0; jj < bN; jj++)
						tmp += img[(stride_i+ii)*width + stride_j+jj];
				}
				float mean = tmp/(bM*bN);


				if (img[(stride_i+i_insert)*width + stride_j+j_insert]>0.5f*mean)
					bit = 1;
				else
					bit = 0;

				ch = (bit<<b)|ch;


					bj++;
					if (bj>=bsJ){
						bj=0;
						bi++;
					}

			}
			msg[c] = ch;
		}
	}
#pragma omp end declare target


//*****************************************************
//		ENCODER FUNCTION
//*****************************************************
void encoder(char *file_in, char *file_out, char *msg, int msg_len)
{

	int w, h, bytesPerPixel;

	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t*)loadPNG(file_in, &w, &h);
    
	// Create imRGB & imYCrCb
	uint8_t *im_out = malloc(3*w*h*sizeof(uint8_t));
	t_sRGB imRGB;
	imRGB.R = malloc(w*h*sizeof(float));
	imRGB.G = malloc(w*h*sizeof(float));
	imRGB.B = malloc(w*h*sizeof(float));
	t_sYCrCb imYCrCb;
	imYCrCb.Y  = malloc(w*h*sizeof(float));
	imYCrCb.Cr = malloc(w*h*sizeof(float));
	imYCrCb.Cb = malloc(w*h*sizeof(float));
	float *Ydct= malloc(w*h*sizeof(float));

	float *mcosine = malloc(8*8*sizeof(float));
	float *alpha = malloc(8*sizeof(float));


	/*Necessary Parameters:
		float *mcosine, *alpha, *Ydct
		char *msg
		int w, h, msg_len
		uint8_t *im, *im_out
		t_sRGB imRGB
		t_sYCrCb imYCrCb
	*/
    #pragma omp target enter data map(to: mcosine[0 : 8*8], alpha[0 : 8], Ydct[0 : w*h], msg[0 : msg_len], w, h, msg_len, im[0 : w*h], im_out[0 : 3*w*h], imRGB, imYCrCb)
	get_dct8x8_params(mcosine, alpha);	//Obetenmos los parámetros de entrada
	double start = omp_get_wtime();

		im2imRGB(im, w, h, &imRGB);											//Convertimos la imgen a RGB
		rgb2ycbcr(&imRGB, &imYCrCb);										//Convertimos la imagen a YCBCR
		dct8x8_2d(imYCrCb.Y, Ydct, imYCrCb.w, imYCrCb.h, mcosine, alpha);	//Aplicamos transformada
			
		insert_msg(Ydct, imYCrCb.w, imYCrCb.h, msg, msg_len);				// Insert Message	

		idct8x8_2d(Ydct, imYCrCb.Y, imYCrCb.w, imYCrCb.h, mcosine, alpha);	//Aplicamos transformada inversa
		ycbcr2rgb(&imYCrCb, &imRGB);										//Convertimos la imgen a RGB
		imRGB2im(&imRGB, im_out, &w, &h);									//Convertimos al formato original		

	double stop = omp_get_wtime();
	printf("Encoding time=%f sec.\n", stop-start);
	#pragma omp target exit data map(from : mcosine, alpha, Ydct, im_out,imRGB, imYCrCb)


	savePNG(file_out, im_out, w, h);
	free(imRGB.R); free(imRGB.G); free(imRGB.B);
	free(imYCrCb.Y); free(imYCrCb.Cr); free(imYCrCb.Cb);
	free(Ydct);
	free(mcosine); free(alpha);
}


//*****************************************************
//		DECODER FUNCTION
//*****************************************************
void decoder(char *file_in, char *msg_decoded, int msg_len)
{

	int w, h, bytesPerPixel;

	// im corresponds to R0G0B0,R1G1B1.....
	uint8_t *im = (uint8_t*)loadPNG(file_in, &w, &h);
    
	// Create imRGB & imYCrCb
	uint8_t *im_out = malloc(3*w*h*sizeof(uint8_t));
	t_sRGB imRGB;
	imRGB.R = malloc(w*h*sizeof(float));
	imRGB.G = malloc(w*h*sizeof(float));
	imRGB.B = malloc(w*h*sizeof(float));
	t_sYCrCb imYCrCb;
	imYCrCb.Y  = malloc(w*h*sizeof(float));
	imYCrCb.Cr = malloc(w*h*sizeof(float));
	imYCrCb.Cb = malloc(w*h*sizeof(float));
	float *Ydct= malloc(w*h*sizeof(float));

	float *mcosine = malloc(8*8*sizeof(float));
	float *alpha = malloc(8*sizeof(float));


	/*Necessary Parameters:
		float *mcosine, *alpha, *Ydct
		char *msg_decoded
		int w, h, msg_len
		uint8_t *im
		t_sRGB imRGB
		t_sYCrCb imYCrCb
	*/
    #pragma omp target enter data map(to: mcosine[0 : 8*8], alpha[0 : 8], Ydct[0 : w*h], msg_decoded[0 : msg_len], w, h, msg_len, im, imRGB, imYCrCb)
	get_dct8x8_params(mcosine, alpha);		//Obetenmos los parámetros de entrada
	double start = omp_get_wtime();

		im2imRGB(im, w, h, &imRGB);												//Convertimos la imgen a RGB
		rgb2ycbcr(&imRGB, &imYCrCb);											//Convertimos la imagen a YCBCR
		dct8x8_2d(imYCrCb.Y, Ydct, imYCrCb.w, imYCrCb.h, mcosine, alpha);		//Aplicamos transformada
			
		extract_msg(Ydct, imYCrCb.w, imYCrCb.h, msg_decoded, msg_len);			//Extraemos el mensaje

	double stop = omp_get_wtime();
	printf("Decoding time=%f sec.\n", stop-start);
	#pragma omp target exit data map(from : mcosine, alpha, Ydct,imRGB, imYCrCb)

	free(imRGB.R); free(imRGB.G); free(imRGB.B);
	free(imYCrCb.Y); free(imYCrCb.Cr); free(imYCrCb.Cb);
	free(Ydct);
	free(mcosine); free(alpha);
}
