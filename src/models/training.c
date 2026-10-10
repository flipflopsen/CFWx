/* Local analytic reverse-mode tape. Each node owns activation and adjoint.
 * Fan-out accumulates adjoints, including shared self-attention Q/K/V inputs.
 * Recompute trades speed for a simple API with no stale external cache. */
#include "models/training.h"
#include <string.h>
#include <stdlib.h>

typedef enum {LEAF, ADD, MUL, MATMUL, BIAS, ACT, NORM, HEAD, SLICE, JOIN, EMBED} operation;
typedef struct node node;
struct node
{
    cfwx_matrix value, grad;
    operation op;
    node *a, *b, *c, *d;
    double *destination;
    size_t start;
    double epsilon;
    bool rms, causal;
    cfwx_activation activation;
    const bool *mask;
    const size_t *tokens;
};
#define MAX_NODES 2048
typedef struct
{
    node *nodes[MAX_NODES];
    size_t count;
    cfwx_status status;
} tape;

static bool shape(const cfwx_matrix *x, size_t rows, size_t cols)
{
    size_t n;
    return x && x->data && x->rows == rows && x->cols == cols && cfwx_count(rows, cols, &n);
}

static node *allocate(tape *t, size_t rows, size_t cols, operation op)
{
    if (t->status != CFWX_OK) return NULL;
    if (t->count == MAX_NODES) {t->status = CFWX_ALLOC; return NULL;}
    node *n = calloc(1, sizeof(*n));
    if (!n) {t->status = CFWX_ALLOC; return NULL;}
    t->nodes[t->count++] = n;
    t->status = cfwx_matrix_init(&n->value, rows, cols);
    if (t->status == CFWX_OK) t->status = cfwx_matrix_init(&n->grad, rows, cols);
    n->op = op;
    return t->status == CFWX_OK ? n : NULL;
}

static void release(tape *t)
{
    for (size_t i = 0; i < t->count; i++)
    {
        cfwx_matrix_free(&t->nodes[i]->value);
        cfwx_matrix_free(&t->nodes[i]->grad);
        free(t->nodes[i]);
    }
}

static node *leaf(tape *t, const double *data, size_t rows, size_t cols, double *gradient)
{
    if (!data) {t->status = CFWX_INVALID; return NULL;}
    node *n = allocate(t, rows, cols, LEAF);
    if (!n) return NULL;
    for (size_t i = 0; i < rows * cols; i++)
    {
        if (!isfinite(data[i])) {t->status = CFWX_NUMERIC; return NULL;}
        n->value.data[i] = data[i];
    }
    n->destination = gradient;
    return n;
}

static node *binary(tape *t, node *a, node *b, operation op)
{
    if (!a || !b) return NULL;
    size_t rows = a->value.rows, cols = op == MATMUL ? b->value.cols : a->value.cols;
    if ((op == MATMUL && a->value.cols != b->value.rows) ||
        (op != MATMUL && !shape(&b->value, rows, cols))) {t->status = CFWX_SHAPE; return NULL;}
    node *n = allocate(t, rows, cols, op);
    if (!n) return NULL;
    n->a = a; n->b = b;
    if (op == MATMUL) t->status = cfwx_matrix_multiply(&a->value, &b->value, &n->value);
    else for (size_t i = 0; i < rows * cols; i++)
        n->value.data[i] = op == ADD ? a->value.data[i] + b->value.data[i] : a->value.data[i] * b->value.data[i];
    return n;
}

static node *activation(tape *t, node *x, cfwx_activation kind)
{
    if (!x) return NULL;
    node *n = allocate(t, x->value.rows, x->value.cols, ACT);
    if (!n) return NULL;
    n->a = x; n->activation = kind;
    t->status = cfwx_activate(&x->value, kind, &n->value);
    return n;
}

static node *dense(tape *t, node *x, const cfwx_dense *layer, cfwx_dense *g)
{
    if (!x || !layer) return NULL;
    if (g && (!shape(&g->weights, layer->weights.rows, layer->weights.cols) ||
              !g->bias.data || g->bias.size != layer->bias.size)) {t->status = CFWX_SHAPE; return NULL;}
    node *w = leaf(t, layer->weights.data, layer->weights.rows, layer->weights.cols, g ? g->weights.data : NULL);
    node *b = leaf(t, layer->bias.data, 1, layer->bias.size, g ? g->bias.data : NULL);
    node *product = binary(t, x, w, MATMUL);
    if (!product || !b || b->value.cols != product->value.cols) {if(t->status == CFWX_OK)t->status = CFWX_SHAPE; return NULL;}
    node *n = allocate(t, product->value.rows, product->value.cols, BIAS);
    if (!n) return NULL;
    n->a = product; n->b = b;
    for (size_t i = 0; i < n->value.rows * n->value.cols; i++)
        n->value.data[i] = product->value.data[i] + b->value.data[i % n->value.cols];
    return n;
}

static node *normalization(tape *t, node *x, node *gamma, node *beta, double eps, bool rms)
{
    if (!x || !gamma || (!rms && !beta)) return NULL;
    if (gamma->value.cols != x->value.cols || (!rms && beta->value.cols != x->value.cols)) {t->status = CFWX_SHAPE; return NULL;}
    node *n = allocate(t, x->value.rows, x->value.cols, NORM);
    if (!n) return NULL;
    n->a = x; n->b = gamma; n->c = beta; n->epsilon = eps; n->rms = rms;
    t->status = rms ? cfwx_rms_norm(&x->value, gamma->value.data, eps, &n->value) :
        cfwx_layer_norm(&x->value, gamma->value.data, beta->value.data, eps, &n->value);
    return n;
}

static node *head(tape *t, node *q, node *k, node *v, node *bias, const bool *mask, bool causal)
{
    if (!q || !k || !v) return NULL;
    node *n = allocate(t, q->value.rows, v->value.cols, HEAD);
    if (!n) return NULL;
    n->a = q; n->b = k; n->c = v; n->d = bias; n->mask = mask; n->causal = causal;
    t->status = cfwx_masked_attention(&q->value, &k->value, &v->value, mask,
                                    bias ? &bias->value : NULL, causal, &n->value, NULL);
    return n;
}

static node *slice(tape *t, node *x, size_t start, size_t width)
{
    if (!x) return NULL;
    node *n = allocate(t, x->value.rows, width, SLICE);
    if (!n) return NULL;
    n->a = x; n->start = start;
    for (size_t i = 0; i < x->value.rows; i++)
        memcpy(n->value.data + i * width, x->value.data + i * x->value.cols + start, width * sizeof(double));
    return n;
}

static node *join(tape *t, node *a, node *b)
{
    if (!a || !b) return NULL;
    node *n = allocate(t, a->value.rows, a->value.cols + b->value.cols, JOIN);
    if (!n) return NULL;
    n->a = a; n->b = b;
    for (size_t i = 0; i < n->value.rows; i++)
    {
        memcpy(n->value.data + i * n->value.cols, a->value.data + i * a->value.cols, a->value.cols * sizeof(double));
        memcpy(n->value.data + i * n->value.cols + a->value.cols, b->value.data + i * b->value.cols, b->value.cols * sizeof(double));
    }
    return n;
}

static node *multihead(tape *t, node *query, node *context, const cfwx_multihead *layer,
                       cfwx_multihead *g, const bool *mask, bool causal)
{
    if (!layer || !layer->width || !layer->heads || layer->width % layer->heads) {t->status = CFWX_INVALID; return NULL;}
    node *q = dense(t, query, &layer->query, g ? &g->query : NULL);
    node *k = dense(t, context, &layer->key, g ? &g->key : NULL);
    node *v = dense(t, context, &layer->value, g ? &g->value : NULL);
    node *joined = NULL;
    size_t width = layer->width / layer->heads;
    for (size_t i = 0; i < layer->heads && t->status == CFWX_OK; i++)
    {
        node *h = head(t, slice(t,q,i*width,width), slice(t,k,i*width,width), slice(t,v,i*width,width), NULL, mask, causal);
        joined = joined ? join(t, joined, h) : h;
    }
    return dense(t, joined, &layer->output, g ? &g->output : NULL);
}

static node *feedforward(tape *t, node *x, const cfwx_feedforward *layer, cfwx_feedforward *g)
{
    if (!layer) {t->status = CFWX_INVALID; return NULL;}
    if (g && (g->gated != layer->gated || g->activation != layer->activation)) {t->status = CFWX_SHAPE; return NULL;}
    node *up = dense(t, x, &layer->up, g ? &g->up : NULL);
    node *a;
    if (layer->gated)
    {
        node *gate = dense(t, x, &layer->gate, g ? &g->gate : NULL);
        a = binary(t, activation(t, gate, layer->activation), up, MUL);
    }
    else a = activation(t, up, layer->activation);
    return dense(t, a, &layer->down, g ? &g->down : NULL);
}

static node *block(tape *t, node *x, const cfwx_transformer_block *layer,
                   cfwx_transformer_block *g, const bool *mask, bool causal)
{
    if (!layer) {t->status = CFWX_INVALID; return NULL;}
    const cfwx_vector *vectors[] = {&layer->gamma1,&layer->beta1,&layer->gamma2,&layer->beta2};
    cfwx_vector *grads[] = {g ? &g->gamma1 : NULL,g ? &g->beta1 : NULL,g ? &g->gamma2 : NULL,g ? &g->beta2 : NULL};
    node *n[4];
    for (size_t i=0;i<4;i++)
    {
        if (g && (!grads[i]->data || grads[i]->size != vectors[i]->size)) {t->status=CFWX_SHAPE;return NULL;}
        n[i]=leaf(t,vectors[i]->data,1,vectors[i]->size,g?grads[i]->data:NULL);
    }
    node *ln1=normalization(t,x,n[0],n[1],layer->epsilon,false);
    node *residual=binary(t,x,multihead(t,ln1,ln1,&layer->attention,g?&g->attention:NULL,mask,causal),ADD);
    node *ln2=normalization(t,residual,n[2],n[3],layer->epsilon,false);
    return binary(t,residual,feedforward(t,ln2,&layer->feedforward,g?&g->feedforward:NULL),ADD);
}

/* Reverse one node, accumulating into all parents (never overwrite fan-out). */
static void reverse_node(tape *t, node *n)
{
    size_t rows=n->value.rows, cols=n->value.cols;
    node *a=n->a,*b=n->b,*c=n->c;
    if(n->op==LEAF)return;
    if(n->op==ADD || n->op==MUL)
        for(size_t i=0;i<rows*cols;i++)
        {
            a->grad.data[i]+=n->grad.data[i]*(n->op==MUL?b->value.data[i]:1.0);
            b->grad.data[i]+=n->grad.data[i]*(n->op==MUL?a->value.data[i]:1.0);
        }
    else if(n->op==MATMUL)
        for(size_t i=0;i<rows;i++)for(size_t j=0;j<cols;j++)for(size_t k=0;k<a->value.cols;k++)
        {
            a->grad.data[i*a->value.cols+k]+=n->grad.data[i*cols+j]*b->value.data[k*cols+j];
            b->grad.data[k*cols+j]+=a->value.data[i*a->value.cols+k]*n->grad.data[i*cols+j];
        }
    else if(n->op==BIAS)
        for(size_t i=0;i<rows*cols;i++){a->grad.data[i]+=n->grad.data[i];b->grad.data[i%cols]+=n->grad.data[i];}
    else if(n->op==ACT)
    {
        cfwx_matrix temp={0};t->status=cfwx_matrix_init(&temp,rows,cols);
        if(t->status==CFWX_OK)t->status=cfwx_activate_backward(&a->value,&n->grad,n->activation,&temp);
        if(t->status==CFWX_OK)for(size_t i=0;i<rows*cols;i++)a->grad.data[i]+=temp.data[i];
        cfwx_matrix_free(&temp);
    }
    else if(n->op==SLICE)
        for(size_t i=0;i<rows;i++)for(size_t j=0;j<cols;j++)a->grad.data[i*a->value.cols+n->start+j]+=n->grad.data[i*cols+j];
    else if(n->op==JOIN)
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<a->value.cols;j++)a->grad.data[i*a->value.cols+j]+=n->grad.data[i*cols+j];
            for(size_t j=0;j<b->value.cols;j++)b->grad.data[i*b->value.cols+j]+=n->grad.data[i*cols+a->value.cols+j];
        }
    else if(n->op==EMBED)
        for(size_t i=0;i<rows;i++)for(size_t j=0;j<cols;j++)a->grad.data[n->tokens[i]*cols+j]+=n->grad.data[i*cols+j];
    else if(n->op==NORM)
        for(size_t i=0;i<rows;i++)
        {
            double mean=0.0,m2=0.0;
            if(!n->rms)
                for(size_t j=0;j<cols;j++){double delta=a->value.data[i*cols+j]-mean;mean+=delta/(double)(j+1);m2+=delta*(a->value.data[i*cols+j]-mean);}
            else for(size_t j=0;j<cols;j++)m2+=a->value.data[i*cols+j]*a->value.data[i*cols+j];
            double inv=1.0/sqrt(m2/(double)cols+n->epsilon),sum=0.0,dot=0.0;
            for(size_t j=0;j<cols;j++)
            {
                double z=(a->value.data[i*cols+j]-mean)*inv;
                double u=n->grad.data[i*cols+j]*b->value.data[j];
                sum+=u;dot+=u*z;
                b->grad.data[j]+=n->grad.data[i*cols+j]*z;
                if(c)c->grad.data[j]+=n->grad.data[i*cols+j];
            }
            /* LayerNorm projects out mean and variance directions. RMSNorm
             * removes only the radial component, not the mean component. */
            for(size_t j=0;j<cols;j++)
            {
                double z=(a->value.data[i*cols+j]-mean)*inv;
                double u=n->grad.data[i*cols+j]*b->value.data[j];
                a->grad.data[i*cols+j]+=inv*(u-(n->rms?0.0:sum/(double)cols)-z*dot/(double)cols);
            }
        }
    else if(n->op==HEAD)
    {
        size_t keys=b->value.rows,depth=a->value.cols;
        cfwx_matrix p={0},out={0};
        t->status=cfwx_matrix_init(&p,rows,keys);
        if(t->status==CFWX_OK)t->status=cfwx_matrix_init(&out,rows,cols);
        if(t->status==CFWX_OK)t->status=cfwx_masked_attention(&a->value,&b->value,&c->value,n->mask,n->d?&n->d->value:NULL,n->causal,&out,&p);
        if(t->status==CFWX_OK)
            for(size_t i=0;i<rows;i++)
            {
                double weighted=0.0;
                for(size_t j=0;j<keys;j++)
                {
                    double dp=0.0;
                    for(size_t d=0;d<cols;d++)dp+=n->grad.data[i*cols+d]*c->value.data[j*cols+d];
                    weighted+=p.data[i*keys+j]*dp;
                }
                for(size_t j=0;j<keys;j++)
                {
                    if(p.data[i*keys+j]==0.0)continue;
                    double dp=0.0;
                    for(size_t d=0;d<cols;d++)
                    {
                        dp+=n->grad.data[i*cols+d]*c->value.data[j*cols+d];
                        c->grad.data[j*cols+d]+=p.data[i*keys+j]*n->grad.data[i*cols+d];
                    }
                    /* Softmax Jacobian-vector product: p*(dp-dot(p,dp)).
                     * Masked entries have exactly zero score gradients. */
                    double ds=p.data[i*keys+j]*(dp-weighted);
                    if(n->d)n->d->grad.data[i*keys+j]+=ds;
                    ds/=sqrt((double)depth);
                    for(size_t d=0;d<depth;d++)
                    {
                        a->grad.data[i*depth+d]+=ds*b->value.data[j*depth+d];
                        b->grad.data[j*depth+d]+=ds*a->value.data[i*depth+d];
                    }
                }
            }
        cfwx_matrix_free(&p);cfwx_matrix_free(&out);
    }
}

static cfwx_status backward(tape *t,node *output,const cfwx_matrix *upstream)
{
    if(t->status!=CFWX_OK)return t->status;
    if(!output || !shape(upstream,output->value.rows,output->value.cols))return CFWX_SHAPE;
    memcpy(output->grad.data,upstream->data,upstream->rows*upstream->cols*sizeof(double));
    for(size_t i=t->count;i-->0 && t->status==CFWX_OK;)reverse_node(t,t->nodes[i]);
    /* Validate before exporting gradient leaves to caller storage. */
    for(size_t i=0;i<t->count;i++)for(size_t j=0;j<t->nodes[i]->grad.rows*t->nodes[i]->grad.cols;j++)
        if(!isfinite(t->nodes[i]->grad.data[j]))return CFWX_NUMERIC;
    if(t->status!=CFWX_OK)return t->status;
    for(size_t i=0;i<t->count;i++)
    {
        node *n=t->nodes[i];
        if(n->destination)memcpy(n->destination,n->grad.data,n->grad.rows*n->grad.cols*sizeof(double));
    }
    return CFWX_OK;
}

cfwx_status cfwx_attention_backward(const cfwx_matrix *q,const cfwx_matrix *k,const cfwx_matrix *v,
    const bool *mask,const cfwx_matrix *bias,bool causal,const cfwx_matrix *dy,
    cfwx_matrix *dq,cfwx_matrix *dk,cfwx_matrix *dv,cfwx_matrix *dbias)
{
    if(!q || !k || !v || !shape(dq,q->rows,q->cols) || !shape(dk,k->rows,k->cols) || !shape(dv,v->rows,v->cols) ||
       (dbias && (!bias || !shape(dbias,bias->rows,bias->cols))))return CFWX_SHAPE;
    tape t={0};
    node *a=leaf(&t,q->data,q->rows,q->cols,dq->data),*b=leaf(&t,k->data,k->rows,k->cols,dk->data),*c=leaf(&t,v->data,v->rows,v->cols,dv->data);
    node *d=bias?leaf(&t,bias->data,bias->rows,bias->cols,dbias?dbias->data:NULL):NULL;
    node *out=head(&t,a,b,c,d,mask,causal);
    cfwx_status s=backward(&t,out,dy);release(&t);return s;
}

static cfwx_status norm_backward(const cfwx_matrix *x,const double *gamma,double eps,
    const cfwx_matrix *dy,cfwx_matrix *dx,double *dg,double *db,bool rms)
{
    if(!x || !gamma || !dg || (!rms && !db) || !shape(dx,x->rows,x->cols))return CFWX_INVALID;
    tape t={0};node *a=leaf(&t,x->data,x->rows,x->cols,dx->data),*g=leaf(&t,gamma,1,x->cols,dg);
    node *b=NULL;
    if(!rms){b=allocate(&t,1,x->cols,LEAF);if(b)b->destination=db;}
    node *out=normalization(&t,a,g,b,eps,rms);
    cfwx_status s=backward(&t,out,dy);release(&t);return s;
}
cfwx_status cfwx_layer_norm_backward(const cfwx_matrix *x,const double *gamma,double eps,
    const cfwx_matrix *dy,cfwx_matrix *dx,double *dg,double *db)
{
    return norm_backward(x,gamma,eps,dy,dx,dg,db,false);
}
cfwx_status cfwx_rms_norm_backward(const cfwx_matrix *x,const double *gamma,double eps,
    const cfwx_matrix *dy,cfwx_matrix *dx,double *dg)
{
    return norm_backward(x,gamma,eps,dy,dx,dg,NULL,true);
}
cfwx_status cfwx_rope_backward(const cfwx_matrix *dy,size_t offset,double base,cfwx_matrix *dx)
{
    if(!dy || !shape(dx,dy->rows,dy->cols) || dy->cols%2 || !isfinite(base) || base<=1.0 || offset>SIZE_MAX-(dy->rows-1))return CFWX_INVALID;
    for(size_t i=0;i<dy->rows;i++)for(size_t j=0;j<dy->cols;j+=2)
    {
        double angle=(double)(offset+i)/pow(base,(double)j/(double)dy->cols);
        double a=dy->data[i*dy->cols+j],b=dy->data[i*dy->cols+j+1];
        double u=a*cos(angle)+b*sin(angle),v=-a*sin(angle)+b*cos(angle);
        if(!isfinite(u)||!isfinite(v))return CFWX_NUMERIC;
        dx->data[i*dx->cols+j]=u;dx->data[i*dx->cols+j+1]=v;
    }
    return CFWX_OK;
}
cfwx_status cfwx_multihead_backward(const cfwx_multihead *layer,const cfwx_matrix *q,const cfwx_matrix *context,
    const bool *mask,bool causal,const cfwx_matrix *dy,cfwx_matrix *dq,cfwx_matrix *dc,cfwx_multihead *g)
{
    if(!q || !context || !g || !shape(dq,q->rows,q->cols)||!shape(dc,context->rows,context->cols))return CFWX_INVALID;
    tape t={0};node *a=leaf(&t,q->data,q->rows,q->cols,dq->data),*b=leaf(&t,context->data,context->rows,context->cols,dc->data);
    node *out=multihead(&t,a,b,layer,g,mask,causal);
    cfwx_status s=backward(&t,out,dy);release(&t);return s;
}
cfwx_status cfwx_feedforward_backward(const cfwx_feedforward *layer,const cfwx_matrix *x,
    const cfwx_matrix *dy,cfwx_matrix *dx,cfwx_feedforward *g)
{
    if(!x||!g||!shape(dx,x->rows,x->cols))return CFWX_INVALID;
    tape t={0};node *a=leaf(&t,x->data,x->rows,x->cols,dx->data),*out=feedforward(&t,a,layer,g);
    cfwx_status s=backward(&t,out,dy);release(&t);return s;
}
cfwx_status cfwx_transformer_block_backward(const cfwx_transformer_block *layer,const cfwx_matrix *x,
    const bool *mask,bool causal,const cfwx_matrix *dy,cfwx_matrix *dx,cfwx_transformer_block *g)
{
    if(!x||!g||!shape(dx,x->rows,x->cols))return CFWX_INVALID;
    tape t={0};node *a=leaf(&t,x->data,x->rows,x->cols,dx->data),*out=block(&t,a,layer,g,mask,causal);
    cfwx_status s=backward(&t,out,dy);release(&t);return s;
}

static cfwx_status parameter(cfwx_parameters *p,double *data,size_t n)
{
    if(!p||!data||!n||p->count>=CFWX_MAX_PARAMETERS)return CFWX_INVALID;
    p->data[p->count]=data;p->size[p->count++]=n;return CFWX_OK;
}
cfwx_status cfwx_parameters_dense(cfwx_parameters *p,cfwx_dense *l)
{
    if(!l)return CFWX_INVALID;
    cfwx_status s=parameter(p,l->weights.data,l->weights.rows*l->weights.cols);
    if(s==CFWX_OK)s=parameter(p,l->bias.data,l->bias.size);return s;
}
cfwx_status cfwx_parameters_block(cfwx_parameters *p,cfwx_transformer_block *b)
{
    if(!b)return CFWX_INVALID;
    cfwx_dense *d[]={&b->attention.query,&b->attention.key,&b->attention.value,&b->attention.output,&b->feedforward.up,&b->feedforward.down};
    cfwx_status s=CFWX_OK;
    for(size_t i=0;i<6 && s==CFWX_OK;i++)s=cfwx_parameters_dense(p,d[i]);
    if(s==CFWX_OK && b->feedforward.gated)s=cfwx_parameters_dense(p,&b->feedforward.gate);
    cfwx_vector *v[]={&b->gamma1,&b->beta1,&b->gamma2,&b->beta2};
    for(size_t i=0;i<4 && s==CFWX_OK;i++)s=parameter(p,v[i]->data,v[i]->size);return s;
}
cfwx_status cfwx_clip_gradients(cfwx_parameters *g,double maximum,double *norm)
{
    if(!g||!norm||!isfinite(maximum)||maximum<=0||g->count>CFWX_MAX_PARAMETERS)return CFWX_INVALID;
    double total=0.0;
    for(size_t i=0;i<g->count;i++)for(size_t j=0;j<g->size[i];j++)
    {
        if(!isfinite(g->data[i][j]))return CFWX_NUMERIC;
        total=hypot(total,g->data[i][j]);
    }
    if(!isfinite(total))return CFWX_NUMERIC;
    *norm=total;double scale=total>maximum?maximum/total:1.0;
    for(size_t i=0;i<g->count;i++)for(size_t j=0;j<g->size[i];j++)g->data[i][j]*=scale;
    return CFWX_OK;
}
void cfwx_adamw_free(cfwx_adamw *o)
{
    if(!o)return;
    for(size_t i=0;i<o->count;i++){free(o->first[i]);free(o->second[i]);}
    *o=(cfwx_adamw){0};
}
cfwx_status cfwx_adamw_init(cfwx_adamw *o,const cfwx_parameters *p)
{
    if(!o||o->count||!p||!p->count||p->count>CFWX_MAX_PARAMETERS)return CFWX_INVALID;
    cfwx_adamw temp={0};temp.count=p->count;
    for(size_t i=0;i<p->count;i++)
    {
        size_t n;if(!cfwx_count(p->size[i],1,&n)){cfwx_adamw_free(&temp);return CFWX_INVALID;}
        temp.size[i]=n;temp.first[i]=calloc(n,sizeof(double));temp.second[i]=calloc(n,sizeof(double));
        if(!temp.first[i]||!temp.second[i]){cfwx_adamw_free(&temp);return CFWX_ALLOC;}
    }
    *o=temp;return CFWX_OK;
}
cfwx_status cfwx_adamw_step(cfwx_adamw *o,cfwx_parameters *p,const cfwx_parameters *g,
    double rate,double b1,double b2,double eps,double decay)
{
    if(!o||!p||!g||!o->count||o->count!=p->count||p->count!=g->count||o->step==SIZE_MAX||
       !isfinite(rate)||rate<=0||!isfinite(b1)||b1<0||b1>=1||!isfinite(b2)||b2<0||b2>=1||
       !isfinite(eps)||eps<=0||!isfinite(decay)||decay<0)return CFWX_INVALID;
    double c1=1.0-pow(b1,(double)(o->step+1)),c2=1.0-pow(b2,(double)(o->step+1));
    /* Validate the complete candidate update before changing any moments or weights. */
    for(size_t i=0;i<p->count;i++)
    {
        if(p->size[i]!=g->size[i]||p->size[i]!=o->size[i])return CFWX_SHAPE;
        for(size_t j=0;j<p->size[i];j++)
        {
            double d=g->data[i][j],m=b1*o->first[i][j]+(1-b1)*d,v=b2*o->second[i][j]+(1-b2)*d*d;
            double next=p->data[i][j]-rate*(m/c1/(sqrt(v/c2)+eps)+decay*p->data[i][j]);
            if(!isfinite(d)||!isfinite(m)||!isfinite(v)||v<0||!isfinite(next))return CFWX_NUMERIC;
        }
    }
    for(size_t i=0;i<p->count;i++)for(size_t j=0;j<p->size[i];j++)
    {
        double d=g->data[i][j];o->first[i][j]=b1*o->first[i][j]+(1-b1)*d;
        o->second[i][j]=b2*o->second[i][j]+(1-b2)*d*d;
        p->data[i][j]-=rate*(o->first[i][j]/c1/(sqrt(o->second[i][j]/c2)+eps)+decay*p->data[i][j]);
    }
    o->step++;return CFWX_OK;
}

void cfwx_language_model_free(cfwx_language_model *m)
{
    if(!m)return;
    cfwx_matrix_free(&m->embeddings);
    for(size_t i=0;i<CFWX_MAX_BLOCKS;i++)cfwx_transformer_block_free(&m->blocks[i]);
    cfwx_vector_free(&m->gamma);cfwx_vector_free(&m->beta);cfwx_dense_free(&m->head);
    *m=(cfwx_language_model){0};
}
cfwx_status cfwx_language_model_init(cfwx_language_model *m,size_t vocab,size_t width,size_t heads,size_t hidden,size_t depth,uint64_t seed)
{
    if(!m||m->vocabulary||m->embeddings.data||!vocab||!depth||depth>CFWX_MAX_BLOCKS)return CFWX_INVALID;
    cfwx_language_model temp={0};cfwx_dense embedding={0};
    cfwx_status s=cfwx_dense_init(&embedding,vocab,width,seed);
    if(s==CFWX_OK){temp.embeddings=embedding.weights;embedding.weights=(cfwx_matrix){0};}
    cfwx_dense_free(&embedding);
    for(size_t i=0;i<depth && s==CFWX_OK;i++)s=cfwx_transformer_block_init(&temp.blocks[i],width,heads,hidden,seed+1+i*10);
    if(s==CFWX_OK)s=cfwx_vector_init(&temp.gamma,width);
    if(s==CFWX_OK)s=cfwx_vector_init(&temp.beta,width);
    if(s==CFWX_OK)s=cfwx_dense_init(&temp.head,width,vocab,seed+100);
    if(s!=CFWX_OK){cfwx_language_model_free(&temp);return s;}
    for(size_t i=0;i<width;i++)temp.gamma.data[i]=1.0;
    temp.vocabulary=vocab;temp.width=width;temp.depth=depth;*m=temp;return CFWX_OK;
}
cfwx_status cfwx_parameters_language_model(cfwx_parameters *p,cfwx_language_model *m)
{
    if(!m||!m->depth||m->depth>CFWX_MAX_BLOCKS)return CFWX_INVALID;
    cfwx_status s=parameter(p,m->embeddings.data,m->embeddings.rows*m->embeddings.cols);
    for(size_t i=0;i<m->depth && s==CFWX_OK;i++)s=cfwx_parameters_block(p,&m->blocks[i]);
    if(s==CFWX_OK)s=parameter(p,m->gamma.data,m->gamma.size);
    if(s==CFWX_OK)s=parameter(p,m->beta.data,m->beta.size);
    if(s==CFWX_OK)s=cfwx_parameters_dense(p,&m->head);return s;
}

static node *language(tape *t,const cfwx_language_model *m,const size_t *ids,size_t length,cfwx_language_model *g)
{
    if(!m||!ids||!length||!m->depth||m->depth>CFWX_MAX_BLOCKS){t->status=CFWX_INVALID;return NULL;}
    if(g && (g->depth!=m->depth||g->width!=m->width||g->vocabulary!=m->vocabulary ||
       !shape(&g->embeddings,m->vocabulary,m->width)||g->gamma.size!=m->width||g->beta.size!=m->width)){t->status=CFWX_SHAPE;return NULL;}
    for(size_t i=0;i<length;i++)if(ids[i]>=m->vocabulary){t->status=CFWX_SHAPE;return NULL;}
    node *table=leaf(t,m->embeddings.data,m->vocabulary,m->width,g?g->embeddings.data:NULL);
    node *embedded=allocate(t,length,m->width,EMBED),*positions=allocate(t,length,m->width,LEAF);
    if(!table||!embedded||!positions)return NULL;
    embedded->a=table;embedded->tokens=ids;
    t->status=cfwx_embedding(&table->value,ids,length,&embedded->value);
    if(t->status==CFWX_OK)t->status=cfwx_sinusoidal_positions(0,&positions->value);
    node *x=binary(t,embedded,positions,ADD);
    for(size_t i=0;i<m->depth && t->status==CFWX_OK;i++)x=block(t,x,&m->blocks[i],g?&g->blocks[i]:NULL,NULL,true);
    node *gamma=leaf(t,m->gamma.data,1,m->width,g?g->gamma.data:NULL),*beta=leaf(t,m->beta.data,1,m->width,g?g->beta.data:NULL);
    return dense(t,normalization(t,x,gamma,beta,1e-5,false),&m->head,g?&g->head:NULL);
}
cfwx_status cfwx_language_model_forward(const cfwx_language_model *m,const size_t *ids,size_t length,cfwx_matrix *logits)
{
    if(!m||!shape(logits,length,m->vocabulary))return CFWX_SHAPE;
    tape t={0};node *out=language(&t,m,ids,length,NULL);cfwx_status s=t.status;
    if(s==CFWX_OK && out)memcpy(logits->data,out->value.data,length*m->vocabulary*sizeof(double));
    release(&t);return s;
}
cfwx_status cfwx_language_model_loss(const cfwx_language_model *m,const size_t *ids,const size_t *targets,
    size_t length,double *loss,cfwx_language_model *g)
{
    if(!targets||!loss)return CFWX_INVALID;
    tape t={0};node *out=language(&t,m,ids,length,g);cfwx_status s=t.status;
    cfwx_matrix upstream={0};
    if(s==CFWX_OK && g)s=cfwx_matrix_init(&upstream,length,m->vocabulary);
    if(s==CFWX_OK && out)s=cfwx_token_cross_entropy(&out->value,targets,SIZE_MAX,0.0,loss,g?&upstream:NULL);
    if(s==CFWX_OK && g)s=backward(&t,out,&upstream);
    cfwx_matrix_free(&upstream);release(&t);return s;
}
