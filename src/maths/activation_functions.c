#include "maths/activation_functions.h"
double cfwx_sigmoid(double x){if(x>=0) return 1/(1+exp(-x)); double e=exp(x); return e/(1+e);}
double cfwx_sigmoid_derivative(double x){double s=cfwx_sigmoid(x); return s*(1-s);}
double cfwx_relu(double x){return isnan(x)?x:(x>0?x:0);}
double cfwx_relu_derivative(double x){return x>0?1:0;}
double cfwx_tanh_derivative(double x){double t=tanh(x); return 1-t*t;}
double cfwx_gelu(double x){if(isinf(x)) return x>0?x:0; return .5*x*(1+erf(x/sqrt(2.0)));}
double cfwx_gelu_derivative(double x){if(isinf(x)) return x>0?1:0; return .5*(1+erf(x/sqrt(2.0)))+x*exp(-.5*x*x)/sqrt(6.2831853071795864769);}
cfwx_status cfwx_softmax(const double *x,size_t n,double *out){
    if(!x || !out || !n) return CFWX_INVALID;
    double max=x[0]; for(size_t i=0;i<n;i++){if(!isfinite(x[i])) return CFWX_NUMERIC; if(x[i]>max) max=x[i];}
    double sum=0; for(size_t i=0;i<n;i++){out[i]=exp(x[i]-max); sum+=out[i];}
    for(size_t i=0;i<n;i++) out[i]/=sum;
    return CFWX_OK;
}
