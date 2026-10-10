"""Idempotent CPU sliding-cache, durable publication and stable checkpoint-name integration."""
from pathlib import Path
p=Path('include/models/runtime.h');s=p.read_text()
if 'CFWX_CACHE_SLIDING' not in s:
 s=s.replace('typedef struct\n{\n    const cfwx_language_model *model;', 'typedef enum { CFWX_CACHE_FIXED, CFWX_CACHE_SLIDING } cfwx_cache_mode;\n\ntypedef struct\n{\n    const cfwx_language_model *model;')
 s=s.replace('    size_t capacity, length;', '    size_t capacity, length;\n    size_t position; /* Absolute position; does not reset on eviction. */\n    cfwx_cache_mode mode;')
 s=s.replace('void cfwx_kv_cache_reset(', '''/* Configurable mode; legacy init defaults to fixed. Sliding evicts oldest K/V
 * only when full. On failed full append committed entries are restored. */
cfwx_status cfwx_kv_cache_init_mode(cfwx_kv_cache *cache, const cfwx_language_model *model,
                                  size_t capacity, cfwx_cache_mode mode);
void cfwx_kv_cache_reset(''')
 p.write_text(s)
p=Path('src/models/runtime.c');s=p.read_text()
if 'cfwx_kv_cache_init_mode' not in s:
 s=s.replace('    if (cache)\n        cache->length = 0;', '    if (cache)\n    {\n        cache->length = 0;\n        cache->position = 0;\n    }')
 start=s.index('cfwx_status cfwx_kv_cache_init(');end=s.index('cfwx_status cfwx_language_model_decode(',start)
 old=s[start:end]
 new=old.replace('cfwx_status cfwx_kv_cache_init(', 'cfwx_status cfwx_kv_cache_init_mode(',1).replace('size_t capacity)', 'size_t capacity, cfwx_cache_mode mode)',1)
 new=new.replace('!capacity)', '!capacity || (mode != CFWX_CACHE_FIXED && mode != CFWX_CACHE_SLIDING))')
 new=new.replace('    temp.capacity = capacity;', '    temp.capacity = capacity;\n    temp.mode = mode;')
 new+='''cfwx_status cfwx_kv_cache_init(cfwx_kv_cache *cache, const cfwx_language_model *model,
                               size_t capacity)
{
    return cfwx_kv_cache_init_mode(cache, model, capacity, CFWX_CACHE_FIXED);
}

'''
 s=s[:start]+new+s[end:]
 s=s.replace('if (!cache || !cache->model || cache->length >= cache->capacity)', 'if (!cache || !cache->model || cache->position == SIZE_MAX ||\n        (cache->mode == CFWX_CACHE_FIXED && cache->length >= cache->capacity))')
 marker='    cfwx_matrix x = {0}'
 index=s.index(marker,s.index('cfwx_status cfwx_language_model_decode'))
 s=s[:index]+'''    bool evict = cache->length == cache->capacity;
    double *backup = NULL;
    size_t cache_elements = cache->capacity * m->width;
    if (evict)
    {
        size_t total;
        if (!cfwx_count(cache_elements, 2 * m->depth, &total)) return CFWX_INVALID;
        backup = malloc(total * sizeof(double));
        if (!backup) return CFWX_ALLOC;
        for (size_t layer = 0; layer < m->depth; layer++)
        {
            memcpy(backup + 2 * layer * cache_elements, cache->keys[layer].data,
                   cache_elements * sizeof(double));
            memcpy(backup + (2 * layer + 1) * cache_elements, cache->values[layer].data,
                   cache_elements * sizeof(double));
            memmove(cache->keys[layer].data, cache->keys[layer].data + m->width,
                    (cache_elements - m->width) * sizeof(double));
            memmove(cache->values[layer].data, cache->values[layer].data + m->width,
                    (cache_elements - m->width) * sizeof(double));
        }
        cache->length--;
    }
'''+s[index:]
 s=s.replace('cfwx_sinusoidal_positions(cache->length, &positions)', 'cfwx_sinusoidal_positions(cache->position, &positions)')
 s=s.replace('    if (s == CFWX_OK)\n        cache->length++;\ncleanup:', '''    if (s == CFWX_OK)
    {
        cache->length++;
        cache->position++;
    }
cleanup:
    if (evict && s != CFWX_OK)
    {
        for (size_t layer = 0; layer < m->depth; layer++)
        {
            memcpy(cache->keys[layer].data, backup + 2 * layer * cache_elements,
                   cache_elements * sizeof(double));
            memcpy(cache->values[layer].data, backup + (2 * layer + 1) * cache_elements,
                   cache_elements * sizeof(double));
        }
        cache->length++;
    }
    free(backup);''')
 p.write_text(s)
p=Path('src/models/checkpoint.c');s=p.read_text()
if 'fsync_parent' not in s:
 s=s.replace('#include <float.h>', '#include <float.h>\n#include <unistd.h>\n#include <fcntl.h>')
 index=s.index('cfwx_status cfwx_checkpoint_save(')
 s=s[:index]+'''/* Rename persistence requires syncing the directory entry as well as payload. */
static bool fsync_parent(const char *path)
{
    size_t n = strlen(path);
    char *parent = malloc(n + 2);
    if (!parent) return false;
    memcpy(parent, path, n + 1);
    char *slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) slash[1] = '\\0';
    else *slash = '\\0';
    int fd = open(parent, O_RDONLY | O_DIRECTORY);
    free(parent);
    if (fd < 0) return false;
    bool ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = false;
    return ok;
}

'''+s[index:]
 s=s.replace('malloc(n + 5)', 'malloc(n + 12)').replace('snprintf(temp, n + 5, "%s.tmp", path);', 'snprintf(temp, n + 12, "%s.tmp.XXXXXX", path);')
 s=s.replace('    FILE *file = fopen(temp, "wb");', '''    int descriptor = mkstemp(temp);
    FILE *file = descriptor >= 0 ? fdopen(descriptor, "wb") : NULL;
    if (!file && descriptor >= 0) { close(descriptor); remove(temp); }''')
 s=s.replace('    if (fclose(file) != 0)\n        ok = false;', '    if (ok && fsync(fileno(file)) != 0)\n        ok = false;\n    if (fclose(file) != 0)\n        ok = false;',1)
 s=s.replace('    if (ok && rename(temp, path) != 0)\n        ok = false;', '''    bool renamed = false;
    if (ok)
    {
        renamed = rename(temp, path) == 0;
        ok = renamed;
    }
    /* Failure here may leave the new checkpoint visible, but durability uncertain. */
    if (renamed && !fsync_parent(path)) ok = false;''')
 p.write_text(s)
p=Path('test/runtime_driver.c');s=p.read_text()
if '--checkpoint-name' not in s:
 index=s.index('int main(')
 s=s[:index]+'''/* Stable identifier flag resolves to NAME.ckpt in the working directory. */
static bool checkpoint_name(const char *name, char *path, size_t capacity)
{
    size_t n = strlen(name);
    if (!n || n > 100 || n + 6 > capacity) return false;
    for (size_t i = 0; i < n; i++)
        if (!((name[i] >= 'a' && name[i] <= 'z') || (name[i] >= 'A' && name[i] <= 'Z') ||
              (name[i] >= '0' && name[i] <= '9') || name[i] == '_' || name[i] == '-')) return false;
    snprintf(path, capacity, "%s.ckpt", name);
    return true;
}

'''+s[index:]
 s=s.replace('    size_t count;', '''    size_t count;
    if (argc == 7 && (!strcmp(argv[1], "train") || !strcmp(argv[1], "resume")) &&
        !strcmp(argv[3], "--checkpoint-name") && number(argv[5], &count) && !strcmp(argv[6], "cpu"))
    {
        char path[128];
        if (!checkpoint_name(argv[4], path, sizeof(path))) return 1;
        return train(&logger, argv[2], path, count, !strcmp(argv[1], "resume"));
    }''',1)
 s=s.replace('Only the tested CPU backend', 'Stable-name syntax: train/resume DATA --checkpoint-name NAME STEPS cpu.\\nOnly the tested CPU backend')
 # Keep default generation fixed; add optional mode/window to driver.
 s=s.replace('const char *prompt, size_t count)', 'const char *prompt, size_t count, size_t window, cfwx_cache_mode mode)')
 s=s.replace('cfwx_kv_cache_init(&cache, &m, length + count)', 'cfwx_kv_cache_init_mode(&cache, &m, window ? window : length + count, mode)')
 s=s.replace('return generate(&logger, argv[2], argv[3], count);', 'return generate(&logger, argv[2], argv[3], count, 0, CFWX_CACHE_FIXED);')
 needle='    if (argc == 5 && !strcmp(argv[1], "generate")'
 index=s.index(needle)
 s=s[:index]+'''    if (argc == 7 && !strcmp(argv[1], "generate") && number(argv[4], &count))
    {
        size_t window;
        if (!number(argv[6], &window)) return 1;
        cfwx_cache_mode mode;
        if (!strcmp(argv[5], "sliding")) mode = CFWX_CACHE_SLIDING;
        else if (!strcmp(argv[5], "fixed")) mode = CFWX_CACHE_FIXED;
        else return 1;
        return generate(&logger, argv[2], argv[3], count, window, mode);
    }
'''+s[index:]
 p.write_text(s)
import subprocess
subprocess.run(['clang-format','-i','include/models/runtime.h','src/models/runtime.c','src/models/checkpoint.c','test/runtime_driver.c'],check=True)
