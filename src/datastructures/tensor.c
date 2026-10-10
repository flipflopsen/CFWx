#include "datastructures/tensor.h"
static cfwx_status layout(cfwx_tensor *t,size_t rank,const size_t *shape){
    if(!shape || !rank || rank>CFWX_MAX_RANK) return CFWX_INVALID;
    t->size=1; t->rank=rank;
    for(size_t i=rank;i-->0;){t->stride[i]=t->size; t->shape[i]=shape[i]; if(!cfwx_count(t->size,shape[i],&t->size)) return CFWX_INVALID;}
    return CFWX_OK;
}
cfwx_status cfwx_tensor_init(cfwx_tensor *t,size_t rank,const size_t *shape){
    if(!t || t->data || t->rank || t->size) return CFWX_INVALID;
    cfwx_tensor tmp={0}; cfwx_status s=layout(&tmp,rank,shape); if(s) return s;
    tmp.data=calloc(tmp.size,sizeof(double)); if(!tmp.data) return CFWX_ALLOC;
    *t=tmp; return CFWX_OK;
}
void cfwx_tensor_free(cfwx_tensor *t){if(t){free(t->data); *t=(cfwx_tensor){0};}}
cfwx_status cfwx_tensor_offset(const cfwx_tensor *t,const size_t *idx,size_t *offset){
    if(!t || !t->data || !idx || !offset) return CFWX_INVALID;
    size_t n=0; for(size_t i=0;i<t->rank;i++){if(idx[i]>=t->shape[i]) return CFWX_SHAPE; n+=idx[i]*t->stride[i];}
    *offset=n; return CFWX_OK;
}
cfwx_status cfwx_tensor_reshape(cfwx_tensor *t,size_t rank,const size_t *shape){
    if(!t || !t->data) return CFWX_INVALID;
    cfwx_tensor tmp={0}; cfwx_status s=layout(&tmp,rank,shape); if(s) return s;
    if(tmp.size!=t->size) return CFWX_SHAPE;
    tmp.data=t->data; *t=tmp; return CFWX_OK;
}
