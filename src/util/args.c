#include <stdlib.h>
#include <string.h>


#include "util/messages.h"
#include "util/args.h"




static void 
set_default_options(options_t* options)
{
    options->help = false;
    options->version = false;
    options->use_colors = true;
}


void
switch_options (int arg, options_t* options)
{
    switch (arg)
    {
        case 'h':
            options->help = true;
            help();
            exit(EXIT_SUCCESS);

        case 'v':
            options->version = true;
            version();
            exit(EXIT_SUCCESS);

        case 0:
            options->use_colors = false;
            break;

        case '?':
            usage();
            exit(EXIT_FAILURE);

        default:
            usage();
            abort();
    }
}

/*
 * Tries to get the file name. Otherwise, gets stdin
 */
void
get_file_name (int argc, char* argv[], options_t* options)
{

    /* If there is more arguments, probably, it is an input file */
    if (optind < argc) {
        strncpy(options->file_name, argv[optind++], FILE_NAME_SIZE - 1);

    /* Otherwise, assumes stdin as the input file */
    } else {
        strncpy(options->file_name, "-", FILE_NAME_SIZE - 1);
    }
}

/*
 * Public function that loops until command line options were parsed
 */
void
options_parser (int argc, char* argv[], options_t* options)
{
    set_default_options(options);

    int arg; 

    static struct option long_options[] =
    {
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'v'},
        {"no-colors", no_argument, 0, 0},
    };

    while (true) {
        int option_index = 0;
        arg = getopt_long(argc, argv, "hvt:", long_options, &option_index);

        if (arg == -1) break;

        switch_options(arg, options);
    }

    get_file_name(argc, argv, options);
}
