#ifndef CFWX_ACTIVATIONS_H
#define CFWX_ACTIVATIONS_H
#include "cfwx.h"
double cfwx_sigmoid(double x);
double cfwx_sigmoid_derivative(double x);
double cfwx_relu(double x);
double cfwx_relu_derivative(double x);
double cfwx_tanh_derivative(double x);
double cfwx_gelu(double x);
double cfwx_gelu_derivative(double x);
/* Finite logits; exact input/output aliasing allowed. */
cfwx_status cfwx_softmax(const double *x,size_t n,double *out);
#endif
