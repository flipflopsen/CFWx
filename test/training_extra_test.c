/* Cross-attention, residual block input, RMS constant-row and AdamW regressions. */
#include "models/training.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s\n",__LINE__,#x);exit(1);}}while(0)
#define OK(x) CHECK((x)==CFWX_OK)
#define CLOSE(a,b) CHECK(isfinite(a)&&fabs((a)-(b))<2e-5*(1+fabs(b)))
static double dot(cfwx_matrix *a,cfwx_matrix *b)
{double s=0;for(size_t i=0;i<a->rows*a->cols;i++)s+=a->data[i]*b->data[i];return s;}
int main(void)
{
    cfwx_multihead m={0},g={0};OK(cfwx_multihead_init(&m,4,2,6));OK(cfwx_multihead_init(&g,4,2,0));
    cfwx_matrix q={0},c={0},dy={0},dq={0},dc={0},out={0};cfwx_matrix *buffers[]={&q,&c,&dy,&dq,&dc,&out};
    for(size_t i=0;i<6;i++){OK(cfwx_matrix_init(buffers[i],(i==1||i==4)?3:2,4));for(size_t j=0;j<buffers[i]->rows*4;j++)buffers[i]->data[j]=sin((double)(j+i));}
    bool mask[]={true,true,false,true,false,true};
    OK(cfwx_multihead_backward(&m,&q,&c,mask,false,&dy,&dq,&dc,&g));
    cfwx_parameters p={0},grad={0};cfwx_dense *parts[]={&m.query,&m.key,&m.value,&m.output},*gp[]={&g.query,&g.key,&g.value,&g.output};
    for(size_t i=0;i<4;i++){OK(cfwx_parameters_dense(&p,parts[i]));OK(cfwx_parameters_dense(&grad,gp[i]));}
    for(size_t a=0;a<p.count+2;a++)for(size_t j=0;j<(a<p.count?p.size[a]:a==p.count?8:12);j++)
    {
        double *v=a<p.count?&p.data[a][j]:a==p.count?&q.data[j]:&c.data[j];double old=*v,h=1e-6;
        *v=old+h;OK(cfwx_multihead_forward(&m,&q,&c,mask,false,&out));double plus=dot(&out,&dy);
        *v=old-h;OK(cfwx_multihead_forward(&m,&q,&c,mask,false,&out));double minus=dot(&out,&dy);*v=old;
        double analytic=a<p.count?grad.data[a][j]:a==p.count?dq.data[j]:dc.data[j];CLOSE(analytic,(plus-minus)/(2*h));
    }
    cfwx_transformer_block block={0},bg={0};OK(cfwx_transformer_block_init(&block,4,2,5,9));OK(cfwx_transformer_block_init(&bg,4,2,5,0));
    OK(cfwx_transformer_block_backward(&block,&q,NULL,true,&dy,&dq,&bg));
    for(size_t j=0;j<8;j++)
    {
        double old=q.data[j],h=1e-6;q.data[j]=old+h;OK(cfwx_transformer_block_forward(&block,&q,NULL,true,&out));double plus=dot(&out,&dy);
        q.data[j]=old-h;OK(cfwx_transformer_block_forward(&block,&q,NULL,true,&out));double minus=dot(&out,&dy);q.data[j]=old;CLOSE(dq.data[j],(plus-minus)/(2*h));
    }
    /* A constant row is not a division-by-zero case with epsilon>0. */
    double gamma[]={1,1,1,1},dg[4],db[4];for(size_t j=0;j<8;j++)q.data[j]=2;
    OK(cfwx_layer_norm_backward(&q,gamma,1e-5,&dy,&dq,dg,db));
    for(size_t j=0;j<4;j++)CLOSE(dg[j],0);
    cfwx_parameters scalar={0},sg={0};double value=1.0,d=2.0;
    scalar.count=sg.count=1;scalar.data[0]=&value;sg.data[0]=&d;scalar.size[0]=sg.size[0]=1;
    cfwx_adamw optimizer={0};OK(cfwx_adamw_init(&optimizer,&scalar));
    OK(cfwx_adamw_step(&optimizer,&scalar,&sg,0.1,0.9,0.999,1e-8,0.01));CLOSE(value,1.0-0.1*(2.0/(2.0+1e-8)+0.01));CHECK(optimizer.step==1);
    double saved=value;d=NAN;CHECK(cfwx_adamw_step(&optimizer,&scalar,&sg,0.1,0.9,0.999,1e-8,0)==CFWX_NUMERIC);CHECK(value==saved&&optimizer.step==1);
    d=3;double norm;OK(cfwx_clip_gradients(&sg,1,&norm));CLOSE(norm,3);CLOSE(d,1);
    cfwx_adamw_free(&optimizer);cfwx_transformer_block_free(&block);cfwx_transformer_block_free(&bg);
    cfwx_multihead_free(&m);cfwx_multihead_free(&g);for(size_t i=0;i<6;i++)cfwx_matrix_free(buffers[i]);
    puts("PASS cross-attention parameters/inputs, block inputs, constant normalization, AdamW and clipping");return 0;
}
