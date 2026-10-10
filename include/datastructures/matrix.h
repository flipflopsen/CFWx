#ifndef CFWX_MATRIX_H
#define CFWX_MATRIX_H
#include "cfwx.h"
typedef struct {size_t rows,cols; double *data;} cfwx_matrix;
cfwx_status cfwx_matrix_init(cfwx_matrix *m,size_t rows,size_t cols);
void cfwx_matrix_free(cfwx_matrix *m);
cfwx_status cfwx_matrix_multiply(const cfwx_matrix *a,const cfwx_matrix *b,cfwx_matrix *out);
cfwx_status cfwx_matrix_transpose(const cfwx_matrix *a,cfwx_matrix *out);
#endif
