#ifndef CFWX_REWARDS_H
#define CFWX_REWARDS_H
#include "cfwx.h"
/* G[t]=r[t]+gamma*G[t+1], terminal bootstrap zero; exact in-place supported.
 * Split episode boundaries in the caller. */
cfwx_status cfwx_discounted_returns(const double *r,size_t n,double gamma,double *out);
#endif
