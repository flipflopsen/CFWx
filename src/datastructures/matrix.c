#include "datastructures/matrix.h"
cfwx_status cfwx_matrix_init(cfwx_matrix *m,size_t rows,size_t cols){
    size_t n;
    if(!m || m->data || m->rows || m->cols || !cfwx_count(rows,cols,&n)) return CFWX_INVALID;
    double *p=calloc(n,sizeof(double)); if(!p) return CFWX_ALLOC;
    *m=(cfwx_matrix){rows,cols,p}; return CFWX_OK;
}
void cfwx_matrix_free(cfwx_matrix *m){if(m){free(m->data); *m=(cfwx_matrix){0};}}
cfwx_status cfwx_matrix_multiply(const cfwx_matrix *a,const cfwx_matrix *b,cfwx_matrix *out){
    if(!a || !b || !out || !a->data || !b->data || !out->data || out->data==a->data || out->data==b->data) return CFWX_INVALID;
    if(a->cols!=b->rows || out->rows!=a->rows || out->cols!=b->cols) return CFWX_SHAPE;
    for(size_t i=0;i<a->rows;i++) for(size_t j=0;j<b->cols;j++){
        double s=0; for(size_t k=0;k<a->cols;k++) s+=a->data[i*a->cols+k]*b->data[k*b->cols+j];
        if(!isfinite(s)) return CFWX_NUMERIC;
        out->data[i*out->cols+j]=s;
    }
    return CFWX_OK;
}
cfwx_status cfwx_matrix_transpose(const cfwx_matrix *a,cfwx_matrix *out){
    if(!a || !out || !a->data || !out->data || a->data==out->data) return CFWX_INVALID;
    if(out->rows!=a->cols || out->cols!=a->rows) return CFWX_SHAPE;
    for(size_t i=0;i<a->rows;i++) for(size_t j=0;j<a->cols;j++) out->data[j*out->cols+i]=a->data[i*a->cols+j];
    return CFWX_OK;
}
