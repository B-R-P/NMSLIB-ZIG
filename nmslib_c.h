#ifndef NMSLIB_C_H
#define NMSLIB_C_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Data types
typedef enum {
    NMSLIB_DATATYPE_DENSE_VECTOR,
    NMSLIB_DATATYPE_SPARSE_VECTOR,
    NMSLIB_DATATYPE_DENSE_UINT8_VECTOR,
    NMSLIB_DATATYPE_OBJECT_AS_STRING
} nmslib_data_type_t;

// Distance types
typedef enum {
    NMSLIB_DISTTYPE_FLOAT,
    NMSLIB_DISTTYPE_INT
} nmslib_dist_type_t;

// Error codes
typedef enum {
    NMSLIB_SUCCESS = 0,
    NMSLIB_ERROR_NULL_POINTER = 1,
    NMSLIB_ERROR_INVALID_ARGUMENT = 2,
    NMSLIB_ERROR_OUT_OF_MEMORY = 3,
    NMSLIB_ERROR_BUFFER_TOO_SMALL = 4,
    NMSLIB_ERROR_SPACE_INCOMPATIBLE = 5,
    NMSLIB_ERROR_QUERY_TOO_LARGE = 6,
    NMSLIB_ERROR_INVALID_SPARSE_ELEMENT = 7,
    NMSLIB_ERROR_INDEX_BUILD_FAILED = 8,
    NMSLIB_ERROR_QUERY_EXECUTION_FAILED = 9,
    NMSLIB_ERROR_DATA_IO_FAILED = 10,
    NMSLIB_ERROR_PLUGIN_REGISTRATION_FAILED = 11,
    NMSLIB_ERROR_INTERNAL = 12,
    NMSLIB_ERROR_RUNTIME = 13,
    NMSLIB_ERROR_INDEX_NOT_BUILT = 14
} nmslib_error_t;

// Data mode for pointer-based batch addition
typedef enum {
    NMSLIB_DATA_MODE_DENSE_FLOAT = 0,
    NMSLIB_DATA_MODE_SPARSE = 1,
    NMSLIB_DATA_MODE_UINT8 = 2
} nmslib_data_mode_t;

// Sparse element structure
typedef struct {
    uint32_t id;
    float value;
} nmslib_sparse_elem_float_t;

// Result structure
typedef struct {
    int32_t* ids;
    float* distances;
    size_t size;
    size_t capacity;
} nmslib_result_t;

// Allocator structure for memory management
typedef struct {
    void* (*alloc)(size_t size, void* ctx);
    void (*free)(void* ptr, void* ctx);
    void* ctx;
} nmslib_allocator_t;

// Error detail structure for error introspection
typedef struct {
    nmslib_error_t code;
    const char* message;
    const char* file;
    int line;
} nmslib_error_detail_t;

typedef struct {
    nmslib_data_type_t data_type;
    nmslib_dist_type_t dist_type;
} nmslib_index_header_t;


// Opaque handle for the NMSLIB index
typedef struct nmslib_index_t* nmslib_index_handle_t;

// Opaque handle for the NMSLIB parameters
typedef struct nmslib_params_t* nmslib_params_handle_t;

// Ensure NMSLIB registration of spaces/methods is performed.
// Call from Zig if you want explicit control; the C++ wrapper also
// performs a one-time init before creating an index.
void nmslib_init(void);

/**
 * Creates a new NMSLIB index with the specified parameters.
 *
 * @param space The type of space for the index.
 * @param space_params Parameters for the space.
 * @param method The method to use for the index.
 * @param data_type The type of data.
 * @param dist_type The type of distance.
 * @param allocator The allocator to use for memory management.
 * @param out_handle Pointer to store the created index handle.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_index_create(
    const char* space,
    nmslib_params_handle_t space_params,
    const char* method,
    nmslib_data_type_t data_type,
    nmslib_dist_type_t dist_type,
    const nmslib_allocator_t* allocator,
    nmslib_index_handle_t* out_handle
);

/**
 * Destroys an NMSLIB index.
 *
 * @param handle The index handle to destroy.
 */
void nmslib_index_destroy(nmslib_index_handle_t handle);

/**
 * Creates an NMSLIB index.
 *
 * @param index The index handle.
 * @param index_params Parameters for the index.
 * @param print_progress Whether to print progress.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_create_index(
    nmslib_index_handle_t index,
    nmslib_params_handle_t index_params,
    int print_progress
);

/**
 * Resets an NMSLIB index.
 *
 * @param index The index handle.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_reset_index(nmslib_index_handle_t index);

/**
 * Creates a new NMSLIB parameters object.
 *
 * @param allocator The allocator to use for memory management.
 * @return Handle to the created parameters object.
 */
nmslib_params_handle_t nmslib_create_params(const nmslib_allocator_t* allocator);

/**
 * Adds a parameter to the NMSLIB parameters object.
 *
 * @param params The parameters handle.
 * @param name The name of the parameter.
 * @param type The type of the parameter (0=int, 1=double, 2=string).
 * @param value The value of the parameter.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_add_param(
    nmslib_params_handle_t params,
    const char* name,
    int type,
    const void* value
);

/**
 * Frees an NMSLIB parameters object.
 *
 * @param params The parameters handle.
 */
void nmslib_free_params(nmslib_params_handle_t params);

/**
 * Gets the space type of an NMSLIB index.
 *
 * @param index The index handle.
 * @param space_type Pointer to store the space type string.
 * @param space_type_len Pointer to store the length of the space type string.
 * @param allocator The allocator to use for memory management.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_space_type(
    nmslib_index_handle_t index,
    const char** space_type,
    size_t* space_type_len,
    const nmslib_allocator_t* allocator
);

/**
 * Gets the method of an NMSLIB index.
 *
 * @param index The index handle.
 * @param method Pointer to store the method string.
 * @param method_len Pointer to store the length of the method string.
 * @param allocator The allocator to use for memory management.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_method(
    nmslib_index_handle_t index,
    const char** method,
    size_t* method_len,
    const nmslib_allocator_t* allocator
);

/**
 * Frees a string allocated by NMSLIB.
 *
 * @param str The string to free.
 * @param allocator The allocator used to allocate the string.
 */
void nmslib_free_string(char* str, const nmslib_allocator_t* allocator);

/**
 * Gets the last error detail for the NMSLIB index.
 *
 * @param detail Pointer to store the error detail.
 * @param allocator The allocator to use for memory management.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_last_error_detail(
    nmslib_error_detail_t* detail,
    const nmslib_allocator_t* allocator
);

/**
 * Adds a data point to the NMSLIB index.
 *
 * @param index The index handle.
 * @param data The data point to add.
 * @param element_count The number of elements in the data point.
 * @param id The ID of the data point.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_add_data_point(
    nmslib_index_handle_t index,
    const void* data,
    size_t element_count,
    int32_t id
);

/**
 * Adds a batch of data points to the NMSLIB index.
 *
 * @param index The index handle.
 * @param data The data points to add (flat buffer).
 * @param count The number of data points to add.
 * @param element_count The number of elements in each data point (dim for dense).
 * @param ids The IDs of the data points (NULL for auto).
 * @param num_elements The number of elements in each sparse data point (NULL for dense).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_add_data_point_batch(
    nmslib_index_handle_t index,
    const void* data,
    size_t count,
    size_t element_count,
    const int32_t* ids,
    const size_t* num_elements
);

/**
 * Adds a batch of uint8 data points to the NMSLIB index.
 *
 * @param index The index handle.
 * @param data The data points to add (flat buffer).
 * @param count The number of data points to add.
 * @param element_count The number of elements in each data point.
 * @param ids The IDs of the data points (NULL for auto).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_add_data_point_batch_uint8(
    nmslib_index_handle_t index,
    const unsigned char* data,
    size_t count,
    size_t element_count,
    const int32_t* ids
);

/**
 * Adds a batch of string data points to the NMSLIB index.
 *
 * @param index The index handle.
 * @param data The data points to add (array of const char*).
 * @param count The number of data points to add.
 * @param ids The IDs of the data points (NULL for auto).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_add_data_point_batch_string(
    nmslib_index_handle_t index,
    const char* const* data,
    size_t count,
    const int32_t* ids
);

/**
 * Gets the size required for a k-NN query result.
 *
 * @param index The index handle.
 * @param query The query data point.
 * @param query_size_or_elem_count The size or element count of the query.
 * @param k The number of nearest neighbors to find.
 * @param out_size Pointer to store the required size.
 * @param num_elements The number of elements in the query (for sparse).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_knn_query_get_size(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    size_t* out_size,
    size_t num_elements
);

/**
 * Fills a k-NN query result.
 *
 * @param index The index handle.
 * @param query The query data point.
 * @param query_size_or_elem_count The size or element count of the query.
 * @param k The number of nearest neighbors to find.
 * @param result Pointer to store the result.
 * @param num_elements The number of elements in the query (for sparse).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_knn_query_fill(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* result,
    size_t num_elements
);

/**
 * Performs a batch k-NN query.
 *
 * @param index The index handle.
 * @param queries The query data points (flat buffer).
 * @param query_count The number of queries.
 * @param query_size_or_elem_count The size or element count of each query.
 * @param k The number of nearest neighbors to find.
 * @param results Pointer to store the results.
 * @param num_elements The number of elements in each query (for sparse).
 * @param thread_pool_size The size of the thread pool.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_knn_query_batch(
    nmslib_index_handle_t index,
    const void* queries,
    size_t query_count,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* results,
    const size_t* num_elements,
    size_t thread_pool_size
);

/**
 * Gets the size required for a range query result.
 *
 * @param index The index handle.
 * @param query The query data point.
 * @param query_size_or_elem_count The size or element count of the query.
 * @param radius The radius for the range query.
 * @param out_size Pointer to store the required size.
 * @param num_elements The number of elements in the query (for sparse).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_range_query_get_size(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    double radius,
    size_t* out_size,
    size_t num_elements
);

/**
 * Fills a range query result.
 *
 * @param index The index handle.
 * @param query The query data point.
 * @param query_size_or_elem_count The size or element count of the query.
 * @param radius The radius for the range query.
 * @param result Pointer to store the result.
 * @param num_elements The number of elements in the query (for sparse).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_range_query_fill(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    double radius,
    nmslib_result_t* result,
    size_t num_elements
);

/**
 * Gets the distance between two data points in the NMSLIB index.
 *
 * @param index The index handle.
 * @param pos1 The position of the first data point.
 * @param pos2 The position of the second data point.
 * @param distance Pointer to store the distance.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_distance(
    nmslib_index_handle_t index,
    size_t pos1,
    size_t pos2,
    float* distance
);

/**
 * Gets the size of a data point in the NMSLIB index.
 *
 * @param index The index handle.
 * @param position The position of the data point.
 * @param size Pointer to store the size.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_data_point_size(
    nmslib_index_handle_t index,
    size_t position,
    size_t* size
);

/**
 * Fills a data point from the NMSLIB index.
 *
 * @param index The index handle.
 * @param position The position of the data point.
 * @param data Pointer to store the data point.
 * @param size The size of the data point.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_data_point_fill(
    nmslib_index_handle_t index,
    size_t position,
    void* data,
    size_t size
);

/**
 * Gets a string data point from the NMSLIB index.
 *
 * @param index The index handle.
 * @param position The position of the data point.
 * @param data Pointer to store the data point.
 * @param data_len Pointer to store the length of the data point.
 * @param allocator The allocator to use for memory management.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_get_data_point_string(
    nmslib_index_handle_t index,
    size_t position,
    const char** data,
    size_t* data_len,
    const nmslib_allocator_t* allocator
);

/**
 * Borrows a dense data point from the NMSLIB index.
 *
 * @param index The index handle.
 * @param position The position of the data point.
 * @param data Pointer to store the data point.
 * @param size Pointer to store the size of the data point.
 * @param free_fn Pointer to store the free function.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_borrow_data_dense(
    nmslib_index_handle_t index,
    size_t position,
    void** data,
    size_t* size,
    void (**free_fn)(void*)
);

/**
 * Borrows a sparse data point from the NMSLIB index.
 *
 * @param index The index handle.
 * @param position The position of the data point.
 * @param data Pointer to store the data point.
 * @param size Pointer to store the size of the data point.
 * @param free_fn Pointer to store the free function.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_borrow_data_sparse(
    nmslib_index_handle_t index,
    size_t position,
    void** data,
    size_t* size,
    void (**free_fn)(void*)
);

/**
 * Saves an NMSLIB index to a file.
 *
 * @param index The index handle.
 * @param path The path to save the index.
 * @param save_data Whether to save the data.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_save_index(
    nmslib_index_handle_t index,
    const char* path,
    int save_data
);

/**
 * Loads an NMSLIB index from a file.
 *
 * @param path The path to load the index from.
 * @param data_type The type of data.
 * @param dist_type The type of distance.
 * @param allocator The allocator to use for memory management.
 * @param load_data Whether to load the data.
 * @param out_handle Pointer to store the loaded index handle.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_load_index(
    const char* path,
    nmslib_data_type_t data_type,
    nmslib_dist_type_t dist_type,
    const nmslib_allocator_t* allocator,
    int load_data,
    nmslib_index_handle_t* out_handle
);

/**
 * Sets the query time parameters for an NMSLIB index.
 *
 * @param index The index handle.
 * @param params The parameters handle.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_set_query_time_params(
    nmslib_index_handle_t index,
    nmslib_params_handle_t params
);

/**
 * Sets the thread pool size for an NMSLIB index.
 *
 * @param index The index handle.
 * @param size The size of the thread pool.
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_set_thread_pool_size(
    nmslib_index_handle_t index,
    size_t size
);

/**
 * Gets the thread pool size of an NMSLIB index.
 *
 * @param index The index handle.
 * @return The size of the thread pool.
 */
size_t nmslib_get_thread_pool_size(nmslib_index_handle_t index);

/**
 * Gets the number of data points in an NMSLIB index.
 *
 * @param index The index handle.
 * @return The number of data points.
 */
size_t nmslib_data_qty(nmslib_index_handle_t index);

/**
 * Gets an approximate memory usage of the NMSLIB index in bytes.
 * Includes data buffers and internal structures (e.g., graph edges).
 * Returns 0 for invalid or unbuilt indexes.
 *
 * @param handle The index handle.
 * @return Approximate memory usage in bytes.
 */
size_t nmslib_index_memory_usage(nmslib_index_handle_t handle);

/**
 * Adds a batch of data points to the NMSLIB index using borrowed pointers (zero-copy).
 * Call before nmslib_create_index. For strings, use the legacy batch_string function.
 *
 * @param handle The index handle.
 * @param data_mode The mode: NMSLIB_DATA_MODE_DENSE_FLOAT (0), SPARSE (1), UINT8 (2).
 * @param data_ptrs Array of pointers to each data point (cast to void**; e.g., const float** for dense).
 * @param count The number of data points.
 * @param element_count Dimension for dense/uint8 (0 for sparse).
 * @param ids The IDs (NULL for auto).
 * @param num_elements Per-point element counts for sparse (NULL otherwise).
 * @return Error code indicating success or failure.
 */
nmslib_error_t nmslib_add_data_point_batch_pointers(
    nmslib_index_handle_t handle,
    nmslib_data_mode_t data_mode,
    const void *const *data_ptrs,
    size_t count,
    size_t element_count,
    const int32_t* ids,
    const size_t* num_elements
);


// Ensure HNSW internal visited-list pool is initialized for the given index handle.
void nmslib_initialize_pool(nmslib_index_handle_t index);



#ifdef __cplusplus
}
#endif

#endif // NMSLIB_C_H