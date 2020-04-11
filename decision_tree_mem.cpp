#include "decision_tree_def.h"


void *safe_malloc(size_t size) {
    void *p;

    if ((p = malloc(size)) == NULL) {
        fprintf(stderr, "Out of memory, "
                        "failed to allocate %zd bytes\n", size);
        exit(1);
    }

    return p;
}


void *safe_alloc(size_t size) {
    void *p;

    if ((p = allocate(size)) == NULL) {
        fprintf(stderr, "Out of memory, [safe_alloc] "
                        "failed to allocate %zd bytes\n", size);
        exit(1);
    }

    return p;
}


void *reshape(void *p, size_t size) {

    if ((p = realloc(p, size)) == NULL) {
        fprintf(stderr, "Out of memory, "
                        "failed to reallocate %zd bytes\n", size);
        exit(1);
    }

    return p;
}


data_t **safe_malloc_2d_dt(int dim_1, int dim_2) {
    data_t **p = (data_t**) safe_malloc(dim_1 * sizeof(data_t*));

    for (int i = 0; i < dim_1; i++)
        p[i] = (data_t*) safe_malloc(dim_2 * sizeof(data_t));

    return p;
}


cont_t **safe_malloc_2d_ct(int dim_1, int dim_2) {
    cont_t **p = (cont_t**) safe_malloc(dim_1 * sizeof(cont_t*));

    for (int i = 0; i < dim_1; i++)
        p[i] = (cont_t*) safe_malloc(dim_2 * sizeof(cont_t));

    return p;
}


void *safe_allocin(int len, size_t size, const unsigned char val) {
    void *p = val ? malloc(len * size) : calloc(len, size);

    if (p == NULL) {
        fprintf(stderr, "Out of memory, "
                        "failed to allocate %zd bytes\n", size);
        exit(1);
    }

    return val ? memset(p, val, len * size) : p;
}
