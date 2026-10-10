#ifndef CFWX_ANN_H
#define CFWX_ANN_H
#include "datastructures/matrix.h"
#include "datastructures/vector.h"
/* W [inputs,outputs], x [batch,inputs], y [batch,outputs]. */
typedef struct {cfwx_matrix weights; cfwx_vector bias;} cfwx_dense;
cfwx_status cfwx_dense_init(cfwx_dense *l,size_t inputs,size_t outputs,uint64_t seed);
void cfwx_dense_free(cfwx_dense *l);
cfwx_status cfwx_dense_forward(const cfwx_dense *l,const cfwx_matrix *x,cfwx_matrix *y);
/* Overwrites dX=dY*W^T, dW=X^T*dY, db=sum_batch(dY).
 * No implicit averaging; reduction belongs to the loss. All outputs disjoint. */
cfwx_status cfwx_dense_backward(const cfwx_dense *l,const cfwx_matrix *x,const cfwx_matrix *dy,cfwx_matrix *dx,cfwx_matrix *dw,cfwx_vector *db);
cfwx_status cfwx_sgd_step(double *p,const double *g,size_t n,double rate);
#endif
