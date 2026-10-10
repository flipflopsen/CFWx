/* Reference CPU batching, full evaluation, incremental KV decode and sampling. */
#include "models/runtime.h"
#include <stdlib.h>
#include <string.h>

static uint64_t random_next(uint64_t *state)
{
    *state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
uint64_t cfwx_dataset_hash(const unsigned char *data, size_t count)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < count; i++) { h ^= data[i]; h *= UINT64_C(1099511628211); }
    return h;
}

cfwx_status cfwx_language_model_batch(const cfwx_language_model *model,
    const size_t *const *tokens, const size_t *const *targets, const size_t *lengths,
    size_t batch, size_t context, double dropout, uint64_t *rng,
    double *loss, cfwx_language_model *gradient)
{
    if (!model || !tokens || !targets || !batch || !loss || !gradient || model == gradient ||
        !isfinite(dropout) || dropout < 0.0 || dropout >= 1.0 || (dropout > 0.0 && !rng)) return CFWX_INVALID;
    size_t total = 0;
    for (size_t b = 0; b < batch; b++)
    {
        size_t n = lengths ? lengths[b] : context;
        if (!n || !tokens[b] || !targets[b]) return CFWX_INVALID;
        for (size_t i = 0; i < n; i++) if (targets[b][i] != SIZE_MAX)
        {
            if (total == SIZE_MAX) return CFWX_INVALID;
            total++;
        }
    }
    if (!total) return CFWX_INVALID;
    cfwx_parameters destination = {0}, scratch = {0};
    cfwx_status s = cfwx_parameters_language_model(&destination, gradient);
    cfwx_language_model temp = {0};
    if (s == CFWX_OK) s = cfwx_language_model_init(&temp, model->vocabulary, model->width,
        model->blocks[0].attention.heads, model->blocks[0].feedforward.up.weights.cols, model->depth, 0);
    if (s == CFWX_OK) s = cfwx_parameters_language_model(&scratch, &temp);
    if (s == CFWX_OK && scratch.count != destination.count) s = CFWX_SHAPE;
    for (size_t a = 0; a < scratch.count && s == CFWX_OK; a++)
        if (scratch.size[a] != destination.size[a]) s = CFWX_SHAPE;
    if (s != CFWX_OK) { cfwx_language_model_free(&temp); return s; }
    for (size_t a = 0; a < destination.count; a++) memset(destination.data[a], 0, destination.size[a] * sizeof(double));
    double sum = 0.0;
    for (size_t b = 0; b < batch && s == CFWX_OK; b++)
    {
        size_t n = lengths ? lengths[b] : context, scored = 0;
        for (size_t i = 0; i < n; i++) if (targets[b][i] != SIZE_MAX) scored++;
        if (!scored) continue;
        double sample_loss = 0.0;
        s = cfwx_language_model_loss_training(model, tokens[b], targets[b], n, dropout, rng, &sample_loss, &temp);
        if (s != CFWX_OK) break;
        double weight = (double)scored / (double)total;
        sum += weight * sample_loss;
        for (size_t a = 0; a < destination.count; a++)
            for (size_t i = 0; i < destination.size[a]; i++) destination.data[a][i] += weight * scratch.data[a][i];
    }
    if (s == CFWX_OK) *loss = sum;
    cfwx_language_model_free(&temp);
    return s;
}

cfwx_status cfwx_language_model_evaluate(const cfwx_language_model *model,
    const size_t *tokens, size_t count, size_t context, double *loss, size_t *scored)
{
    if (!model || !tokens || count < 2 || !context || !loss || !scored) return CFWX_INVALID;
    double sum = 0.0;
    for (size_t start = 0; start < count - 1;)
    {
        size_t length = count - 1 - start;
        if (length > context) length = context;
        double value;
        cfwx_status s = cfwx_language_model_loss(model, tokens + start, tokens + start + 1, length, &value, NULL);
        if (s != CFWX_OK) return s;
        sum += value * (double)length;
        start += length;
    }
    *scored = count - 1; *loss = sum / (double)(count - 1);
    return isfinite(*loss) ? CFWX_OK : CFWX_NUMERIC;
}

void cfwx_kv_cache_free(cfwx_kv_cache *cache)
{
    if (!cache) return;
    for (size_t i = 0; i < CFWX_MAX_BLOCKS; i++)
    { cfwx_matrix_free(&cache->keys[i]); cfwx_matrix_free(&cache->values[i]); }
    *cache = (cfwx_kv_cache){0};
}
void cfwx_kv_cache_reset(cfwx_kv_cache *cache)
{
    if (cache) cache->length = 0;
}
cfwx_status cfwx_kv_cache_init(cfwx_kv_cache *cache, const cfwx_language_model *model, size_t capacity)
{
    if (!cache || cache->model || !model || !model->depth || model->depth > CFWX_MAX_BLOCKS || !capacity) return CFWX_INVALID;
    cfwx_kv_cache temp = {0};
    for (size_t i = 0; i < model->depth; i++)
    {
        cfwx_status s = cfwx_matrix_init(&temp.keys[i], capacity, model->width);
        if (s == CFWX_OK) s = cfwx_matrix_init(&temp.values[i], capacity, model->width);
        if (s != CFWX_OK) { cfwx_kv_cache_free(&temp); return s; }
    }
    temp.model = model; temp.capacity = capacity; *cache = temp;
    return CFWX_OK;
}

cfwx_status cfwx_language_model_decode(cfwx_kv_cache *cache, size_t token, cfwx_matrix *logits)
{
    if (!cache || !cache->model || cache->length >= cache->capacity) return CFWX_INVALID;
    const cfwx_language_model *m = cache->model;
    if (token >= m->vocabulary || !logits || !logits->data || logits->rows != 1 || logits->cols != m->vocabulary) return CFWX_SHAPE;
    cfwx_matrix x={0},positions={0},norm={0},q={0},k={0},v={0},joined={0},branch={0},residual={0};
    cfwx_matrix *buffers[]={&x,&positions,&norm,&q,&k,&v,&joined,&branch,&residual};
    cfwx_status s=CFWX_OK;
    for(size_t i=0;i<9 && s==CFWX_OK;i++)s=cfwx_matrix_init(buffers[i],1,m->width);
    if(s!=CFWX_OK)goto cleanup;
    s=cfwx_embedding(&m->embeddings,&token,1,&x);
    if(s==CFWX_OK)s=cfwx_sinusoidal_positions(cache->length,&positions);
    if(s==CFWX_OK)s=cfwx_residual_add(&x,&positions,&x);
    for(size_t layer=0;layer<m->depth && s==CFWX_OK;layer++)
    {
        const cfwx_transformer_block *b=&m->blocks[layer];
        const cfwx_multihead *a=&b->attention;
        s=cfwx_layer_norm(&x,b->gamma1.data,b->beta1.data,b->epsilon,&norm);
        if(s==CFWX_OK)s=cfwx_dense_forward(&a->query,&norm,&q);
        if(s==CFWX_OK)s=cfwx_dense_forward(&a->key,&norm,&k);
        if(s==CFWX_OK)s=cfwx_dense_forward(&a->value,&norm,&v);
        if(s!=CFWX_OK)break;
        memcpy(cache->keys[layer].data+cache->length*m->width,k.data,m->width*sizeof(double));
        memcpy(cache->values[layer].data+cache->length*m->width,v.data,m->width*sizeof(double));
        size_t depth=m->width/a->heads;
        double *scores=calloc(cache->length+1,sizeof(double));
        if(!scores){s=CFWX_ALLOC;break;}
        for(size_t h=0;h<a->heads && s==CFWX_OK;h++)
        {
            double maximum=-INFINITY;
            for(size_t i=0;i<=cache->length;i++)
            {
                double score=0.0;
                for(size_t d=0;d<depth;d++)score+=q.data[h*depth+d]*cache->keys[layer].data[i*m->width+h*depth+d];
                scores[i]=score/sqrt((double)depth);
                if(!isfinite(scores[i])){s=CFWX_NUMERIC;break;}
                if(scores[i]>maximum)maximum=scores[i];
            }
            if(s!=CFWX_OK)break;
            double sum=0.0;
            for(size_t i=0;i<=cache->length;i++){scores[i]=exp(scores[i]-maximum);sum+=scores[i];}
            for(size_t d=0;d<depth;d++)
            {
                double value=0.0;
                for(size_t i=0;i<=cache->length;i++)value+=scores[i]/sum*cache->values[layer].data[i*m->width+h*depth+d];
                joined.data[h*depth+d]=value;
            }
        }
        free(scores);
        if(s==CFWX_OK)s=cfwx_dense_forward(&a->output,&joined,&branch);
        if(s==CFWX_OK)s=cfwx_residual_add(&x,&branch,&residual);
        if(s==CFWX_OK)s=cfwx_layer_norm(&residual,b->gamma2.data,b->beta2.data,b->epsilon,&norm);
        if(s==CFWX_OK)s=cfwx_feedforward_forward(&b->feedforward,&norm,&branch);
        if(s==CFWX_OK)s=cfwx_residual_add(&residual,&branch,&x);
    }
    if(s==CFWX_OK)s=cfwx_layer_norm(&x,m->gamma.data,m->beta.data,1e-5,&norm);
    if(s==CFWX_OK)s=cfwx_dense_forward(&m->head,&norm,logits);
    if(s==CFWX_OK)cache->length++;
cleanup:
    for(size_t i=0;i<9;i++)cfwx_matrix_free(buffers[i]);
    return s;
}

cfwx_status cfwx_sample_logits(const double *logits,size_t vocabulary,double temperature,
    size_t top_k,uint64_t *rng,size_t *token)
{
    if(!logits||!vocabulary||!token||!isfinite(temperature)||temperature<0||top_k>vocabulary)return CFWX_INVALID;
    size_t best=0;
    for(size_t i=0;i<vocabulary;i++)
    {
        if(!isfinite(logits[i]))return CFWX_NUMERIC;
        if(logits[i]>logits[best])best=i;
    }
    if(temperature==0.0){*token=best;return CFWX_OK;}
    if(!rng)return CFWX_INVALID;
    double *probabilities=calloc(vocabulary,sizeof(double));
    if(!probabilities)return CFWX_ALLOC;
    double sum=0.0;
    for(size_t i=0;i<vocabulary;i++)
    {
        size_t rank=0;
        if(top_k)for(size_t j=0;j<vocabulary;j++)
            if(logits[j]>logits[i] || (logits[j]==logits[i] && j<i))rank++;
        if(!top_k||rank<top_k)probabilities[i]=exp((logits[i]-logits[best])/temperature);
        sum+=probabilities[i];
    }
    double sample=(double)(random_next(rng)>>11)*0x1.0p-53*sum;
    *token=best;
    for(size_t i=0;i<vocabulary;i++)
    {
        sample-=probabilities[i];
        if(probabilities[i]>0.0 && sample<0.0){*token=i;break;}
    }
    free(probabilities);return CFWX_OK;
}
