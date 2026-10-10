#ifndef CFWX_NEURON_H
#define CFWX_NEURON_H
#include "datastructures/vector.h"
typedef struct {cfwx_vector weights; double bias;} cfwx_neuron;
cfwx_status cfwx_neuron_init(cfwx_neuron *n,size_t inputs);
void cfwx_neuron_free(cfwx_neuron *n);
/* Affine output; activation selected by caller. */
cfwx_status cfwx_neuron_forward(const cfwx_neuron *n,const cfwx_vector *x,double *out);
#endif
