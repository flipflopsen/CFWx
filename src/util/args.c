#include <stdlib.h>
#include <string.h>
#include "util/messages.h"
#include "util/args.h"
void options_parser(int argc,char **argv,options_t *opts){
    if(!opts) return;
    *opts=(options_t){.use_colors=true};
    static const struct option long_options[]={
        {"help",no_argument,NULL,'h'},{"version",no_argument,NULL,'v'},
        {"no-colors",no_argument,NULL,'n'},{NULL,0,NULL,0}};
    int c;
    while((c=getopt_long(argc,argv,"hvn",long_options,NULL))!=-1){
        switch(c){case 'h':opts->help=true; help(); exit(EXIT_SUCCESS);
        case 'v':opts->version=true; version(); exit(EXIT_SUCCESS);
        case 'n':opts->use_colors=false; break;
        default:usage(); exit(EXIT_FAILURE);}
    }
    const char *name=optind<argc?argv[optind++]:"-";
    if(strlen(name)>=sizeof opts->file_name || optind<argc){usage(); exit(EXIT_FAILURE);}
    memcpy(opts->file_name,name,strlen(name)+1);
}
