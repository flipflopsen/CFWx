#ifndef CFWX_ATTENTION_H
#define CFWX_ATTENTION_H
#include <stdbool.h>
#include "datastructures/matrix.h"
/* Q [queries,d], K [keys,d], V [keys,value_dim], output [queries,value_dim].
 * Optional probabilities [queries,keys]. Causal mode requires queries==keys;
 * masks key index > query index. Finite inputs, disjoint outputs. Single head. */
cfwx_status cfwx_attention(const cfwx_matrix *q,const cfwx_matrix *k,const cfwx_matrix *v,bool causal,cfwx_matrix *out,cfwx_matrix *p);
/* Row-wise population variance. gamma/beta each have cols entries.
 * x/out may alias exactly; gamma/beta must be disjoint. epsilon > 0. */
cfwx_status cfwx_layer_norm(const cfwx_matrix *x,const double *gamma,const double *beta,double eps,cfwx_matrix *out);
#endif
