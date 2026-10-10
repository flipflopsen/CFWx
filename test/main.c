#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "datastructures/tensor.h"
#include "datastructures/neuron.h"
#include "models/ann.h"
#include "models/attention.h"
#include "maths/activation_functions.h"
#include "maths/loss_functions.h"
#include "maths/rewards.h"
/* Checks remain active under NDEBUG. */
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(EXIT_FAILURE);}}while(0)
#define OK(x) CHECK((x)==CFWX_OK)
#define NEAR(a,b,t) CHECK(isfinite(a) && fabs((a)-(b))<=(t))
static void containers(void){
    cfwx_vector a={0},b={0},sum={0}; OK(cfwx_vector_init(&a,3)); OK(cfwx_vector_init(&b,3)); OK(cfwx_vector_init(&sum,3));
    for(size_t i=0;i<3;i++){a.data[i]=(double)(i+1); b.data[i]=2;}
    double dot; OK(cfwx_vector_dot(&a,&b,&dot)); NEAR(dot,12,1e-12);
    OK(cfwx_vector_add(&a,&b,&sum)); NEAR(sum.data[2],5,1e-12); OK(cfwx_vector_scale(&sum,2)); NEAR(sum.data[2],10,1e-12);
    CHECK(cfwx_vector_init(&a,3)==CFWX_INVALID); cfwx_vector bad={0}; CHECK(cfwx_vector_init(&bad,SIZE_MAX)==CFWX_INVALID);
    cfwx_neuron neuron={0}; OK(cfwx_neuron_init(&neuron,3)); memcpy(neuron.weights.data,b.data,3*sizeof(double)); neuron.bias=1;
    OK(cfwx_neuron_forward(&neuron,&a,&dot)); NEAR(dot,13,1e-12); cfwx_neuron_free(&neuron);
    cfwx_vector_free(&a); cfwx_vector_free(&a); cfwx_vector_free(&b); cfwx_vector_free(&sum);
    cfwx_matrix x={0},y={0},z={0},tr={0}; OK(cfwx_matrix_init(&x,2,3)); OK(cfwx_matrix_init(&y,3,2)); OK(cfwx_matrix_init(&z,2,2)); OK(cfwx_matrix_init(&tr,3,2));
    double xv[]={1,2,3,4,5,6},yv[]={7,8,9,10,11,12}; memcpy(x.data,xv,sizeof xv); memcpy(y.data,yv,sizeof yv);
    OK(cfwx_matrix_multiply(&x,&y,&z)); double expected[]={58,64,139,154}; for(size_t i=0;i<4;i++) NEAR(z.data[i],expected[i],1e-12);
    OK(cfwx_matrix_transpose(&x,&tr)); NEAR(tr.data[1],4,1e-12); CHECK(cfwx_matrix_multiply(&x,&x,&z)==CFWX_SHAPE); CHECK(cfwx_matrix_transpose(&x,&x)==CFWX_INVALID);
    cfwx_matrix_free(&x); cfwx_matrix_free(&y); cfwx_matrix_free(&z); cfwx_matrix_free(&tr);
    cfwx_tensor tensor={0}; size_t shape[]={2,3,4},idx[]={1,2,3},offset; OK(cfwx_tensor_init(&tensor,3,shape)); OK(cfwx_tensor_offset(&tensor,idx,&offset)); CHECK(offset==23);
    idx[0]=2; CHECK(cfwx_tensor_offset(&tensor,idx,&offset)==CFWX_SHAPE);
    size_t newshape[]={6,4}; OK(cfwx_tensor_reshape(&tensor,2,newshape)); CHECK(tensor.size==24 && tensor.stride[0]==4);
    size_t wrong[]={25}; CHECK(cfwx_tensor_reshape(&tensor,1,wrong)==CFWX_SHAPE); cfwx_tensor_free(&tensor); cfwx_tensor_free(&tensor);
}
static void maths(void){
    NEAR(cfwx_sigmoid(1000),1,1e-15); NEAR(cfwx_sigmoid(-1000),0,1e-15); NEAR(cfwx_relu(-1),0,1e-15);
    double x[]={1000,1001,1002},p[3]; OK(cfwx_softmax(x,3,p)); NEAR(p[0]+p[1]+p[2],1,1e-14); NEAR(p[2],0.6652409557748218,1e-12);
    OK(cfwx_softmax(x,3,x)); NEAR(x[2],p[2],1e-14); double invalid[]={NAN}; CHECK(cfwx_softmax(invalid,1,p)==CFWX_NUMERIC);
    double h=1e-6;
    for(int i=-4;i<=4;i++){double z=(double)i/2; NEAR(cfwx_gelu_derivative(z),(cfwx_gelu(z+h)-cfwx_gelu(z-h))/(2*h),1e-8); NEAR(cfwx_sigmoid_derivative(z),(cfwx_sigmoid(z+h)-cfwx_sigmoid(z-h))/(2*h),1e-8); NEAR(cfwx_tanh_derivative(z),(tanh(z+h)-tanh(z-h))/(2*h),1e-8);}
    double logits[]={.3,-.7,1.2},g[3],loss;
    OK(cfwx_cross_entropy_logits(logits,3,1,&loss,g));
    for(size_t i=0;i<3;i++){double old=logits[i],plus,minus; logits[i]=old+h; OK(cfwx_cross_entropy_logits(logits,3,1,&plus,NULL)); logits[i]=old-h; OK(cfwx_cross_entropy_logits(logits,3,1,&minus,NULL)); logits[i]=old; NEAR(g[i],(plus-minus)/(2*h),1e-8);}
    double extreme[]={1000,-1000},target[]={1,0}; OK(cfwx_bce_logits(extreme,target,2,&loss,g)); NEAR(loss,0,1e-12);
    double pred[]={1,3},truth[]={0,1}; OK(cfwx_mse(pred,truth,2,&loss,g)); NEAR(loss,2.5,1e-12); NEAR(g[1],2,1e-12);
    double r[]={1,2,3}; OK(cfwx_discounted_returns(r,3,.5,r)); NEAR(r[0],2.75,1e-12); CHECK(cfwx_discounted_returns(r,3,2,r)==CFWX_INVALID);
}
static double objective(cfwx_dense *l,cfwx_matrix *x,cfwx_matrix *y,cfwx_matrix *dy){OK(cfwx_dense_forward(l,x,y)); double s=0; for(size_t i=0;i<y->rows*y->cols;i++) s+=y->data[i]*dy->data[i]; return s;}
static void dense(void){
    cfwx_dense l={0},same={0}; OK(cfwx_dense_init(&l,2,3,42)); OK(cfwx_dense_init(&same,2,3,42)); CHECK(memcmp(l.weights.data,same.weights.data,6*sizeof(double))==0); cfwx_dense_free(&same);
    cfwx_matrix x={0},y={0},dy={0},dx={0},dw={0}; cfwx_vector db={0};
    OK(cfwx_matrix_init(&x,2,2)); OK(cfwx_matrix_init(&y,2,3)); OK(cfwx_matrix_init(&dy,2,3)); OK(cfwx_matrix_init(&dx,2,2)); OK(cfwx_matrix_init(&dw,2,3)); OK(cfwx_vector_init(&db,3));
    for(size_t i=0;i<4;i++) x.data[i]=.2+(double)i*.1;
    for(size_t i=0;i<6;i++) dy.data[i]=-.2+(double)i*.15;
    OK(cfwx_dense_backward(&l,&x,&dy,&dx,&dw,&db));
    double *params[]={l.weights.data,l.bias.data,x.data}; double *grads[]={dw.data,db.data,dx.data}; size_t sizes[]={6,3,4}; double h=1e-6;
    for(size_t a=0;a<3;a++) for(size_t i=0;i<sizes[a];i++){double old=params[a][i]; params[a][i]=old+h; double plus=objective(&l,&x,&y,&dy); params[a][i]=old-h; double minus=objective(&l,&x,&y,&dy); params[a][i]=old; NEAR(grads[a][i],(plus-minus)/(2*h),1e-8);}
    double before=objective(&l,&x,&y,&dy); OK(cfwx_sgd_step(l.weights.data,dw.data,6,.01)); CHECK(objective(&l,&x,&y,&dy)<before);
    double par[]={1,2},grad[]={1,NAN}; CHECK(cfwx_sgd_step(par,grad,2,.1)==CFWX_NUMERIC); CHECK(par[0]==1);
    cfwx_dense_free(&l); cfwx_matrix_free(&x); cfwx_matrix_free(&y); cfwx_matrix_free(&dy); cfwx_matrix_free(&dx); cfwx_matrix_free(&dw); cfwx_vector_free(&db);
}
static void attention(void){
    cfwx_matrix q={0},k={0},v={0},o={0},p={0}; OK(cfwx_matrix_init(&q,2,2)); OK(cfwx_matrix_init(&k,2,2)); OK(cfwx_matrix_init(&v,2,1)); OK(cfwx_matrix_init(&o,2,1)); OK(cfwx_matrix_init(&p,2,2));
    v.data[0]=2; v.data[1]=6;
    OK(cfwx_attention(&q,&k,&v,false,&o,&p)); NEAR(o.data[0],4,1e-12); NEAR(p.data[0],.5,1e-12);
    OK(cfwx_attention(&q,&k,&v,true,&o,&p)); NEAR(o.data[0],2,1e-12); CHECK(p.data[1]==0); NEAR(o.data[1],4,1e-12);
    q.data[0]=k.data[0]=1; q.data[3]=k.data[3]=1; OK(cfwx_attention(&q,&k,&v,false,&o,&p)); double w=1/(1+exp(-1/sqrt(2.0))); NEAR(p.data[0],w,1e-12); NEAR(o.data[0],2*w+6*(1-w),1e-12);
    k.data[0]=NAN; CHECK(cfwx_attention(&q,&k,&v,false,&o,NULL)==CFWX_NUMERIC);
    double gamma[]={1,1},beta[]={0,0}; q.data[0]=1; q.data[1]=3; q.data[2]=2; q.data[3]=2;
    OK(cfwx_layer_norm(&q,gamma,beta,1e-5,&q)); NEAR(q.data[0],-1/sqrt(1+1e-5),1e-12); NEAR(q.data[3],0,1e-12);
    CHECK(cfwx_layer_norm(&q,gamma,beta,0,&q)==CFWX_INVALID);
    cfwx_matrix_free(&q); cfwx_matrix_free(&k); cfwx_matrix_free(&v); cfwx_matrix_free(&o); cfwx_matrix_free(&p);
}
int main(void){containers(); puts("PASS containers"); maths(); puts("PASS activations/losses/rewards"); dense(); puts("PASS dense gradients/SGD"); attention(); puts("PASS attention/layer norm"); return EXIT_SUCCESS;}
