/* Public contracts for tensor. */

#ifndef CFWX_TENSOR_H
#define CFWX_TENSOR_H
#include "cfwx.h"
#define CFWX_MAX_RANK 8
typedef struct
{
    size_t rank, size, shape[CFWX_MAX_RANK], stride[CFWX_MAX_RANK];
    double *data;
} cfwx_tensor;
cfwx_status cfwx_tensor_init(cfwx_tensor *t, size_t rank, const size_t *shape);
void cfwx_tensor_free(cfwx_tensor *t);
cfwx_status cfwx_tensor_offset(const cfwx_tensor *t, const size_t *indices, size_t *offset);
cfwx_status cfwx_tensor_reshape(cfwx_tensor *t, size_t rank, const size_t *shape);
#endif
