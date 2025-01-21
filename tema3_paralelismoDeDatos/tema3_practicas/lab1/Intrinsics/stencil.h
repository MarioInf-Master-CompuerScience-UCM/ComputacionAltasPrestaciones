#ifndef STENCIL_H
#define STENCIL_H
#include <immintrin.h>

void ApplyStencil( unsigned char *img_in,  unsigned char *img_out, int width, int height);
void ApplyStencil_original( unsigned char *img_in,  unsigned char *img_out, int width, int height);
void CopyImage( unsigned char *img_in,  unsigned char *img_out, int width, int height);

#endif
