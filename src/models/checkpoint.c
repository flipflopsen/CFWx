/* Versioned portable checkpoint with checksum and transactional load.
 * Atomic rename protects against incomplete replacement, not power-loss durability. */
#include "models/runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <unistd.h>
#include <fcntl.h>

typedef struct
{
    FILE *file;
    uint64_t hash;
    bool write, ok;
} stream;
static void bytes(stream *s, unsigned char *data, size_t size)
{
    if (!s->ok)
        return;
    size_t n = s->write ? fwrite(data, 1, size, s->file) : fread(data, 1, size, s->file);
    if (n != size)
    {
        s->ok = false;
        return;
    }
    for (size_t i = 0; i < size; i++)
    {
        s->hash ^= data[i];
        s->hash *= UINT64_C(1099511628211);
    }
}

static void integer(stream *s, uint64_t *value)
{
    unsigned char b[8];
    if (s->write)
        for (size_t i = 0; i < 8; i++)
            b[i] = (unsigned char)(*value >> (8 * i));
    bytes(s, b, 8);
    if (!s->write && s->ok)
    {
        *value = 0;
        for (size_t i = 0; i < 8; i++)
            *value |= (uint64_t)b[i] << (8 * i);
    }
}

static void count(stream *s, size_t *value)
{
    uint64_t v = (uint64_t)*value;
    integer(s, &v);
    if (!s->write && s->ok)
    {
        if (v > SIZE_MAX)
            s->ok = false;
        else
            *value = (size_t)v;
    }
}

static void real(stream *s, double *value)
{
    uint64_t bits = 0;
    if (s->write)
    {
        if (!isfinite(*value))
        {
            s->ok = false;
            return;
        }
        memcpy(&bits, value, 8);
    }
    integer(s, &bits);
    if (!s->write && s->ok)
    {
        memcpy(value, &bits, 8);
        if (!isfinite(*value))
            s->ok = false;
    }
}

static bool compatible(void)
{
    return sizeof(double) == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024;
}

static void state_io(stream *s, cfwx_training_state *state, size_t vocabulary)
{
    integer(s, &state->rng);
    integer(s, &state->dataset_hash);
    count(s, &state->context);
    count(s, &state->batch);
    count(s, &state->unknown);
    for (size_t i = 0; i < 256; i++)
    {
        uint64_t v = state->byte_map[i] == SIZE_MAX ? UINT64_MAX : state->byte_map[i];
        integer(s, &v);
        if (!s->write)
            state->byte_map[i] = v == UINT64_MAX ? SIZE_MAX : (size_t)v;
        if (v != UINT64_MAX && v >= vocabulary)
            s->ok = false;
    }
    double *values[] = {&state->dropout, &state->rate,  &state->beta1, &state->beta2,
                        &state->epsilon, &state->decay, &state->clip};
    for (size_t i = 0; i < 7; i++)
        real(s, values[i]);
    if (!state->context || state->context > 1048576 || !state->batch || state->batch > 4096 ||
        state->unknown >= vocabulary || state->dropout < 0 || state->dropout >= 1 ||
        state->rate <= 0 || state->beta1 < 0 || state->beta1 >= 1 || state->beta2 < 0 ||
        state->beta2 >= 1 || state->epsilon <= 0 || state->decay < 0 || state->clip <= 0)
        s->ok = false;
}

static void payload(stream *s, cfwx_parameters *p, cfwx_adamw *o)
{
    size_t stored = p->count;
    count(s, &stored);
    if (stored != p->count || o->count != p->count)
    {
        s->ok = false;
        return;
    }
    count(s, &o->step);
    for (size_t i = 0; i < p->count && s->ok; i++)
    {
        size_t size = p->size[i];
        count(s, &size);
        if (size != p->size[i] || size != o->size[i])
        {
            s->ok = false;
            break;
        }
        for (size_t j = 0; j < size; j++)
            real(s, &p->data[i][j]);
        for (size_t j = 0; j < size; j++)
            real(s, &o->first[i][j]);
        for (size_t j = 0; j < size; j++)
        {
            real(s, &o->second[i][j]);
            if (o->second[i][j] < 0)
                s->ok = false;
        }
    }
}

static bool footer(stream *s)
{
    uint64_t expected = s->hash, checksum = s->hash;
    integer(s, &checksum);
    if (!s->write && checksum != expected)
        s->ok = false;
    if (!s->write && fgetc(s->file) != EOF)
        s->ok = false;
    if (ferror(s->file))
        s->ok = false;
    return s->ok;
}

static bool architecture(size_t v, size_t w, size_t h, size_t hidden, size_t d)
{
    if (!v || v > 65536 || !w || w > 4096 || !h || h > w || w % h || !hidden || hidden > 65536 ||
        !d || d > CFWX_MAX_BLOCKS)
        return false;
    /* Conservative size bound before any allocations: parameters+Adam moments. */
    long double parameters = (long double)v * w * 2 + v + 2 * w +
                             (long double)d * (4 * ((long double)w * w + w) +
                                               2 * (long double)w * hidden + hidden + w + 4 * w);
    return parameters * 24 < 268435456;
}

/* Rename persistence requires syncing the directory entry as well as payload. */
static bool fsync_parent(const char *path)
{
    size_t n = strlen(path);
    char *parent = malloc(n + 2);
    if (!parent)
        return false;
    memcpy(parent, path, n + 1);
    char *slash = strrchr(parent, '/');
    if (!slash)
        strcpy(parent, ".");
    else if (slash == parent)
        slash[1] = '\0';
    else
        *slash = '\0';
    int fd = open(parent, O_RDONLY | O_DIRECTORY);
    free(parent);
    if (fd < 0)
        return false;
    bool ok = fsync(fd) == 0;
    if (close(fd) != 0)
        ok = false;
    return ok;
}

cfwx_status cfwx_checkpoint_save(const char *path, cfwx_language_model *m,
                                 const cfwx_adamw *optimizer, const cfwx_training_state *state)
{
    if (!path || !path[0] || !m || !optimizer || !state || !compatible())
        return CFWX_INVALID;
    size_t heads = m->blocks[0].attention.heads, hidden = m->blocks[0].feedforward.up.weights.cols;
    if (!architecture(m->vocabulary, m->width, heads, hidden, m->depth))
        return CFWX_INVALID;
    for (size_t i = 0; i < m->depth; i++)
        if (m->blocks[i].attention.heads != heads || m->blocks[i].feedforward.gated ||
            m->blocks[i].feedforward.activation != CFWX_ACT_GELU ||
            m->blocks[i].feedforward.up.weights.cols != hidden || m->blocks[i].epsilon != 1e-5)
            return CFWX_INVALID;
    cfwx_parameters p = {0};
    cfwx_status status = cfwx_parameters_language_model(&p, m);
    if (status != CFWX_OK)
        return status;
    size_t n = strlen(path);
    char *temp = malloc(n + 12);
    if (!temp)
        return CFWX_ALLOC;
    snprintf(temp, n + 12, "%s.tmp.XXXXXX", path);
    int descriptor = mkstemp(temp);
    FILE *file = descriptor >= 0 ? fdopen(descriptor, "wb") : NULL;
    if (!file && descriptor >= 0)
    {
        close(descriptor);
        remove(temp);
    }
    if (!file)
    {
        free(temp);
        return CFWX_INVALID;
    }
    stream s = {file, UINT64_C(14695981039346656037), true, true};
    unsigned char magic[8] = {'C', 'F', 'W', 'X', 'C', 'K', '0', '1'};
    bytes(&s, magic, 8);
    uint64_t version = 1;
    integer(&s, &version);
    size_t vocab = m->vocabulary, width = m->width, depth = m->depth;
    count(&s, &vocab);
    count(&s, &width);
    count(&s, &heads);
    count(&s, &hidden);
    count(&s, &depth);
    cfwx_training_state state_copy = *state;
    state_io(&s, &state_copy, vocab);
    /* payload writes moments without modifying them. */
    payload(&s, &p, (cfwx_adamw *)optimizer);
    bool ok = footer(&s);
    if (fflush(file) != 0)
        ok = false;
    if (ok && fsync(fileno(file)) != 0)
        ok = false;
    if (fclose(file) != 0)
        ok = false;
    bool renamed = false;
    if (ok)
    {
        renamed = rename(temp, path) == 0;
        ok = renamed;
    }
    /* Failure here may leave the new checkpoint visible, but durability uncertain. */
    if (renamed && !fsync_parent(path))
        ok = false;
    if (!ok)
        remove(temp);
    free(temp);
    return ok ? CFWX_OK : CFWX_INVALID;
}

cfwx_status cfwx_checkpoint_load(const char *path, cfwx_language_model *m, cfwx_adamw *o,
                                 cfwx_training_state *state)
{
    if (!path || !m || m->vocabulary || m->embeddings.data || !o || o->count || !state ||
        !compatible())
        return CFWX_INVALID;
    FILE *file = fopen(path, "rb");
    if (!file)
        return CFWX_INVALID;
    if (fseek(file, 0, SEEK_END) != 0)
    {
        fclose(file);
        return CFWX_INVALID;
    }
    long size = ftell(file);
    if (size < 0 || size > 268435456 || fseek(file, 0, SEEK_SET) != 0)
    {
        fclose(file);
        return CFWX_INVALID;
    }
    stream s = {file, UINT64_C(14695981039346656037), false, true};
    unsigned char magic[8];
    bytes(&s, magic, 8);
    uint64_t version = 0;
    integer(&s, &version);
    size_t v = 0, w = 0, h = 0, hidden = 0, d = 0;
    count(&s, &v);
    count(&s, &w);
    count(&s, &h);
    count(&s, &hidden);
    count(&s, &d);
    if (!s.ok || memcmp(magic, "CFWXCK01", 8) || version != 1 || !architecture(v, w, h, hidden, d))
    {
        fclose(file);
        return CFWX_INVALID;
    }
    cfwx_training_state temp_state = {0};
    state_io(&s, &temp_state, v);
    cfwx_language_model temp = {0};
    cfwx_adamw optimizer = {0};
    cfwx_parameters p = {0};
    cfwx_status result =
        s.ok ? cfwx_language_model_init(&temp, v, w, h, hidden, d, 0) : CFWX_INVALID;
    if (result == CFWX_OK)
        result = cfwx_parameters_language_model(&p, &temp);
    if (result == CFWX_OK)
        result = cfwx_adamw_init(&optimizer, &p);
    if (result == CFWX_OK)
    {
        payload(&s, &p, &optimizer);
        if (!footer(&s))
            result = CFWX_INVALID;
    }
    fclose(file);
    if (result == CFWX_OK)
    {
        *m = temp;
        *o = optimizer;
        *state = temp_state;
    }
    else
    {
        cfwx_language_model_free(&temp);
        cfwx_adamw_free(&optimizer);
    }
    return result;
}
