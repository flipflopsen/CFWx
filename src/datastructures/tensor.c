/**
 * @file src/datastructures/tensor.c
 * @brief tensor interfaces and implementation.
 * @ingroup containers
 * @details See the module guide for shape, ownership, failure and execution-policy
 * constraints. Source listings include implementation helpers and ordinary
 * explanatory comments; generated source documentation does not execute code.
 */

/** Implementation and checks for tensor. */

#include "datastructures/tensor.h"

/**
 * @anchor api_src_datastructures_tensor_c_layout
 * @brief Perform the layout operation.
 * @ingroup containers
 * @details Read the adjacent operation-specific contract and linked module guide
 * for exact dimensions, aliasing, reduction and error behavior. Parameters
 * are borrowed unless the constructor/free contract explicitly transfers ownership.
 * @param t t argument as declared; use the adjacent contract for storage, count and input/output role.
 * @param rank Tensor rank; supported range is 1 through CFWX_MAX_RANK.
 * @param shape Dimension array with rank entries; every dimension must be positive.
 * @return CFWX_OK on success; otherwise an invalid-argument, shape, allocation
 * or numerical status as checked by the implementation. Do not consume failed outputs.
 */
static cfwx_status layout(cfwx_tensor *t, size_t rank, const size_t *shape)
{
    /* Reverse dimension traversal constructs contiguous row-major strides. */

    if (!shape || !rank || rank > CFWX_MAX_RANK)
        return CFWX_INVALID;
    t->size = 1;
    t->rank = rank;
    for (size_t i = rank; i-- > 0;)
    {
        t->stride[i] = t->size;
        t->shape[i] = shape[i];
        if (!cfwx_count(t->size, shape[i], &t->size))
            return CFWX_INVALID;
    }
    return CFWX_OK;
}

/**
 * @anchor api_src_datastructures_tensor_c_cfwx_tensor_init
 * @brief Construct and initialize tensor state.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_tensor_init include/datastructures/tensor.h
 */
cfwx_status cfwx_tensor_init(cfwx_tensor *t, size_t rank, const size_t *shape)
{
    /* Temporary metadata leaves ownership unchanged on allocation failure. */

    if (!t || t->data || t->rank || t->size)
        return CFWX_INVALID;
    cfwx_tensor tmp = {0};
    cfwx_status s = layout(&tmp, rank, shape);
    if (s != CFWX_OK)
        return s;
    tmp.data = calloc(tmp.size, sizeof(double));
    if (!tmp.data)
        return CFWX_ALLOC;
    *t = tmp;
    return CFWX_OK;
}

/**
 * @anchor api_src_datastructures_tensor_c_cfwx_tensor_free
 * @brief Release owned tensor storage and reset its owner.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_tensor_free include/datastructures/tensor.h
 */
void cfwx_tensor_free(cfwx_tensor *t)
{
    if (t)
    {
        free(t->data);
        *t = (cfwx_tensor){0};
    }
}

/**
 * @anchor api_src_datastructures_tensor_c_cfwx_tensor_offset
 * @brief Check tensor indices and compute a contiguous-storage offset.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_tensor_offset include/datastructures/tensor.h
 */
cfwx_status cfwx_tensor_offset(const cfwx_tensor *t, const size_t *idx, size_t *offset)
{
    if (!t || !t->data || !idx || !offset)
        return CFWX_INVALID;
    size_t n = 0;
    for (size_t i = 0; i < t->rank; i++)
    {
        if (idx[i] >= t->shape[i])
            return CFWX_SHAPE;
        n += idx[i] * t->stride[i];
    }
    *offset = n;
    return CFWX_OK;
}

/**
 * @anchor api_src_datastructures_tensor_c_cfwx_tensor_reshape
 * @brief Replace tensor metadata while preserving element count and allocation.
 * @ingroup containers
 * @details Implementation of the public declaration; its canonical contract
 * remains in the corresponding header and is merged by Doxygen.
 * @see cfwx_tensor_reshape include/datastructures/tensor.h
 */
cfwx_status cfwx_tensor_reshape(cfwx_tensor *t, size_t rank, const size_t *shape)
{
    /* Reshape preserves the allocation and total element count. */

    if (!t || !t->data)
        return CFWX_INVALID;
    cfwx_tensor tmp = {0};
    cfwx_status s = layout(&tmp, rank, shape);
    if (s != CFWX_OK)
        return s;
    if (tmp.size != t->size)
        return CFWX_SHAPE;
    tmp.data = t->data;
    *t = tmp;
    return CFWX_OK;
}
