#include "maths/rewards.h"
cfwx_status cfwx_discounted_returns(const double *r,size_t n,double gamma,double *out){
    if(!r || !out || !n || !isfinite(gamma) || gamma<0 || gamma>1) return CFWX_INVALID;
    for(size_t i=0;i<n;i++) if(!isfinite(r[i])) return CFWX_NUMERIC;
    double g=0; for(size_t i=n;i-->0;){g=r[i]+gamma*g; if(!isfinite(g)) return CFWX_NUMERIC; out[i]=g;}
    return CFWX_OK;
}
