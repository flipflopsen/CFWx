#include <stdio.h>
#include <stdlib.h>
#include "util/args.h"
#include "maths/activation_functions.h"
int main(int argc,char **argv){
    options_t opts={0}; options_parser(argc,argv,&opts);
    double x[]={1,2,3},p[3];
    if(cfwx_softmax(x,3,p)!=CFWX_OK) return EXIT_FAILURE;
    printf("CFWx numerical playground\nsoftmax([1,2,3]) = [%.6f, %.6f, %.6f]\n",p[0],p[1],p[2]);
    return EXIT_SUCCESS;
}
