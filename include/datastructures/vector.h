#ifndef CFWX_VECTOR_H
#define CFWX_VECTOR_H
#include "cfwx.h"
typedef struct {size_t size; double *data;} cfwx_vector;
cfwx_status cfwx_vector_init(cfwx_vector *v,size_t n);
void cfwx_vector_free(cfwx_vector *v);
cfwx_status cfwx_vector_dot(const cfwx_vector *a,const cfwx_vector *b,double *out);
cfwx_status cfwx_vector_add(const cfwx_vector *a,const cfwx_vector *b,cfwx_vector *out);
cfwx_status cfwx_vector_scale(cfwx_vector *v,double scale);
#endif
