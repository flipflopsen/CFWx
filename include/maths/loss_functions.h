/* Public contracts for loss functions. */

#ifndef CFWX_LOSSES_H
#define CFWX_LOSSES_H
#include "cfwx.h"
/* Optional gradients have n entries and may not overlap inputs.
 * MSE/BCE reduction is mean over scalar elements. CE is one categorical label. */
cfwx_status cfwx_mse(const double *p, const double *t, size_t n, double *loss, double *g);
cfwx_status cfwx_cross_entropy_logits(const double *x, size_t n, size_t label, double *loss,
                                      double *g);
cfwx_status cfwx_bce_logits(const double *x, const double *t, size_t n, double *loss, double *g);
#endif
