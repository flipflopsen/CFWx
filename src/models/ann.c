#include "models/ann.h"
static uint64_t rng(uint64_t *s){*s+=UINT64_C(0x9e3779b97f4a7c15); uint64_t z=*s; z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9); z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb); return z^(z>>31);}
cfwx_status cfwx_dense_init(cfwx_dense *l,size_t inputs,size_t outputs,uint64_t seed){
    if(!l || l->weights.data || l->weights.rows || l->weights.cols || l->bias.data || l->bias.size) return CFWX_INVALID;
    cfwx_dense t={0}; cfwx_status s=cfwx_matrix_init(&t.weights,inputs,outputs); if(s) return s;
    s=cfwx_vector_init(&t.bias,outputs); if(s){cfwx_dense_free(&t); return s;}
    double limit=sqrt(6.0/((double)inputs+(double)outputs));
    for(size_t i=0;i<inputs*outputs;i++) t.weights.data[i]=(2*(double)(rng(&seed)>>11)*0x1.0p-53-1)*limit;
    *l=t; return CFWX_OK;
}
void cfwx_dense_free(cfwx_dense *l){if(l){cfwx_matrix_free(&l->weights); cfwx_vector_free(&l->bias);}}
cfwx_status cfwx_dense_forward(const cfwx_dense *l,const cfwx_matrix *x,cfwx_matrix *y){
    if(!l || !l->bias.data || l->bias.size!=l->weights.cols) return CFWX_INVALID;
    cfwx_status s=cfwx_matrix_multiply(x,&l->weights,y); if(s) return s;
    for(size_t i=0;i<y->rows;i++) for(size_t j=0;j<y->cols;j++){y->data[i*y->cols+j]+=l->bias.data[j]; if(!isfinite(y->data[i*y->cols+j])) return CFWX_NUMERIC;}
    return CFWX_OK;
}
cfwx_status cfwx_dense_backward(const cfwx_dense *l,const cfwx_matrix *x,const cfwx_matrix *dy,cfwx_matrix *dx,cfwx_matrix *dw,cfwx_vector *db){
    if(!l || !x || !dy || !dx || !dw || !db || !l->weights.data || !x->data || !dy->data || !dx->data || !dw->data || !db->data) return CFWX_INVALID;
    size_t in=l->weights.rows,out=l->weights.cols,batch=x->rows;
    if(x->cols!=in || dy->rows!=batch || dy->cols!=out || dx->rows!=batch || dx->cols!=in || dw->rows!=in || dw->cols!=out || db->size!=out) return CFWX_SHAPE;
    const double *read[]={x->data,dy->data,l->weights.data,l->bias.data}; double *write[]={dx->data,dw->data,db->data};
    for(size_t a=0;a<3;a++){for(size_t b=0;b<4;b++) if(write[a]==read[b]) return CFWX_INVALID; for(size_t b=0;b<a;b++) if(write[a]==write[b]) return CFWX_INVALID;}
    for(size_t i=0;i<batch;i++) for(size_t k=0;k<in;k++){double s=0; for(size_t j=0;j<out;j++) s+=dy->data[i*out+j]*l->weights.data[k*out+j]; if(!isfinite(s)) return CFWX_NUMERIC; dx->data[i*in+k]=s;}
    for(size_t k=0;k<in;k++) for(size_t j=0;j<out;j++){double s=0; for(size_t i=0;i<batch;i++) s+=x->data[i*in+k]*dy->data[i*out+j]; if(!isfinite(s)) return CFWX_NUMERIC; dw->data[k*out+j]=s;}
    for(size_t j=0;j<out;j++){double s=0; for(size_t i=0;i<batch;i++) s+=dy->data[i*out+j]; if(!isfinite(s)) return CFWX_NUMERIC; db->data[j]=s;}
    return CFWX_OK;
}
cfwx_status cfwx_sgd_step(double *p,const double *g,size_t n,double rate){
    if(!p || !g || !n || !isfinite(rate) || rate<0 || p==g) return CFWX_INVALID;
    for(size_t i=0;i<n;i++) if(!isfinite(p[i]) || !isfinite(g[i]) || !isfinite(p[i]-rate*g[i])) return CFWX_NUMERIC;
    for(size_t i=0;i<n;i++) p[i]-=rate*g[i];
    return CFWX_OK;
}
