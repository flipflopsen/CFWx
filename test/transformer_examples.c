/* Example driver supplies identities and owns the borrowed logging contexts. */
#include "experiments/transformer_runners.h"
#include <stdlib.h>

int main(void)
{
    experiment_logger language, cross;
    if (experiment_logger_init(&language, "transformer.causal_lm", LOG_DEBUG, NULL, NULL) != 0 ||
        experiment_logger_init(&cross, "transformer.cross_attention", LOG_DEBUG, NULL, NULL) != 0)
        return EXIT_FAILURE;
    if (transformer_language_runner(&language) != EXIT_SUCCESS) return EXIT_FAILURE;
    return transformer_cross_attention_runner(&cross);
}
