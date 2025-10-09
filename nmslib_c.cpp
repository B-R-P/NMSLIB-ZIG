#include "nmslib_c.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include "space.h"
#include "space/space_sparse_vector.h"
#include "space/space_l2sqr_sift.h"
#include "spacefactory.h"
#include "methodfactory.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "rangequery.h"
#include "object.h"
#include "thread_pool.h"

using namespace similarity;

// Thread-local error detail
thread_local struct {
    nmslib_error_t code;
    std::string message;
    std::string file;
    int line;
} last_error_detail = {NMSLIB_SUCCESS, "No error", __FILE__, __LINE__};
struct nmslib_borrowed_data_t {
    void* data;                    // The allocated data
    nmslib_allocator_t allocator;  // The allocator to free the data
};
static void nmslib_borrowed_data_free(void* ptr) {
    if (!ptr) return;
    nmslib_borrowed_data_t* borrowed = static_cast<nmslib_borrowed_data_t*>(ptr);
    if (borrowed->data && borrowed->allocator.free) {
        borrowed->allocator.free(borrowed->data, borrowed->allocator.ctx);
    }
    borrowed->allocator.free(borrowed, borrowed->allocator.ctx);
}
namespace NMSLIBUtil {
    inline void set_last_error(nmslib_error_t code, const std::string& msg, const char* file, int line) {
        last_error_detail = {code, msg.empty() ? "No error" : msg, file, line};
    }

    inline char* dup_string(const std::string& str, const nmslib_allocator_t* allocator) {
        size_t len = str.length() + 1;
        char* result = static_cast<char*>(allocator->alloc(len, allocator->ctx));
        if (!result) return nullptr;
        std::strncpy(result, str.c_str(), len);
        return result;
    }

    inline nmslib_error_t validate_common_inputs(nmslib_index_handle_t index, const void* data, size_t size, const void* out = nullptr) {
        if (!index) return NMSLIB_ERROR_INVALID_ARGUMENT;
        if (!data) return NMSLIB_ERROR_INVALID_ARGUMENT;
        if (size == 0) return NMSLIB_ERROR_INVALID_ARGUMENT;
        if (out == nullptr) return NMSLIB_ERROR_INVALID_ARGUMENT;
        return NMSLIB_SUCCESS;
    }

    inline nmslib_error_t validate_sparse_elements(const nmslib_sparse_elem_float_t* elements, size_t num_elements, bool sorted) {
        if (!elements) return NMSLIB_ERROR_INVALID_SPARSE_ELEMENT;
        if (num_elements == 0) return NMSLIB_ERROR_INVALID_SPARSE_ELEMENT;
        if (sorted) {
            for (size_t i = 1; i < num_elements; ++i) {
                if (elements[i].id <= elements[i - 1].id) {
                    return NMSLIB_ERROR_INVALID_SPARSE_ELEMENT;
                }
            }
        }
        return NMSLIB_SUCCESS;
    }
}

// Wrapper structure for NMSLIB parameters
struct nmslib_params_wrapper_t {
    std::vector<std::string> params;
    nmslib_allocator_t allocator;
};

// Wrapper structure for NMSLIB index
template <typename dist_t>
struct nmslib_internal_index_t {
    std::unique_ptr<Space<dist_t>> space;
    std::unique_ptr<Index<dist_t>> index_ptr;
    ObjectVector data;
    nmslib_data_type_t data_type;
    nmslib_dist_type_t dist_type;
    std::string method;
    std::string space_type;
    nmslib_allocator_t allocator;
    size_t thread_pool_size;

    nmslib_internal_index_t(const std::string& m, const std::string& st, nmslib_data_type_t dt, nmslib_dist_type_t dst, const nmslib_allocator_t* alloc)
        : method(m), space_type(st), data_type(dt), dist_type(dst), allocator(*alloc), thread_pool_size(std::thread::hardware_concurrency()) {}
};

static AnyParams load_params(const nmslib_params_wrapper_t* params) {
    if (!params) return AnyParams();
    return AnyParams(params->params);
}

extern "C" {

nmslib_error_t nmslib_index_create(
    const char* space,
    nmslib_params_handle_t space_params,
    const char* method,
    nmslib_data_type_t data_type,
    nmslib_dist_type_t dist_type,
    const nmslib_allocator_t* allocator,
    nmslib_index_handle_t* out_handle
) {
    if (!space || !method || !allocator || !allocator->alloc || !allocator->free || !out_handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        void* mem = allocator->alloc(sizeof(nmslib_internal_index_t<float>), allocator->ctx);
        if (!mem) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        nmslib_index_handle_t result = nullptr;
        switch (dist_type) {
            case NMSLIB_DISTTYPE_FLOAT: {
                auto idx = new (mem) nmslib_internal_index_t<float>(method, space, data_type, dist_type, allocator);
                idx->space.reset(SpaceFactoryRegistry<float>::Instance().CreateSpace(space, load_params(reinterpret_cast<nmslib_params_wrapper_t*>(space_params))));
                if (!idx->space) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid space type", __FILE__, __LINE__);
                    idx->~nmslib_internal_index_t();
                    allocator->free(mem, allocator->ctx);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                result = reinterpret_cast<nmslib_index_handle_t>(idx);
                break;
            }
            case NMSLIB_DISTTYPE_INT: {
                auto idx = new (mem) nmslib_internal_index_t<int>(method, space, data_type, dist_type, allocator);
                idx->space.reset(SpaceFactoryRegistry<int>::Instance().CreateSpace(space, load_params(reinterpret_cast<nmslib_params_wrapper_t*>(space_params))));
                if (!idx->space) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid space type", __FILE__, __LINE__);
                    idx->~nmslib_internal_index_t();
                    allocator->free(mem, allocator->ctx);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                result = reinterpret_cast<nmslib_index_handle_t>(idx);
                break;
            }
            default:
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid distance type", __FILE__, __LINE__);
                return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Index created successfully", __FILE__, __LINE__);
        *out_handle = result;
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to create index", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

void nmslib_index_destroy(nmslib_index_handle_t handle) {
    if (!handle) return;
    auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
    for (auto datum : idx->data) {
        delete datum;
    }
    idx->data.clear();
    idx->~nmslib_internal_index_t();
    idx->allocator.free(idx, idx->allocator.ctx);
}

nmslib_error_t nmslib_create_index(
    nmslib_index_handle_t handle,
    nmslib_params_handle_t index_params,
    int print_progress
) {
    if (!handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        auto factory = MethodFactoryRegistry<float>::Instance();
        idx->index_ptr.reset(factory.CreateMethod(print_progress, idx->method, idx->space_type, *idx->space, idx->data));
        idx->index_ptr->CreateIndex(load_params(reinterpret_cast<nmslib_params_wrapper_t*>(index_params)));
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Index created successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Failed to create index", __FILE__, __LINE__);
        return NMSLIB_ERROR_INDEX_BUILD_FAILED;
    }
}

nmslib_error_t nmslib_reset_index(nmslib_index_handle_t handle) {
    if (!handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        for (auto datum : idx->data) {
            delete datum;
        }
        idx->data.clear();
        idx->index_ptr.reset();
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Index reset successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to reset index", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_params_handle_t nmslib_create_params(const nmslib_allocator_t* allocator) {
    if (!allocator || !allocator->alloc || !allocator->free) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid allocator", __FILE__, __LINE__);
        return nullptr;
    }
    try {
        void* mem = allocator->alloc(sizeof(nmslib_params_wrapper_t), allocator->ctx);
        if (!mem) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for params", __FILE__, __LINE__);
            return nullptr;
        }
        auto params = new (mem) nmslib_params_wrapper_t{};
        params->params.reserve(4);
        params->allocator = *allocator;
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Parameters created successfully", __FILE__, __LINE__);
        return reinterpret_cast<nmslib_params_handle_t>(params);
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to create params", __FILE__, __LINE__);
        return nullptr;
    }
}

nmslib_error_t nmslib_add_param(
    nmslib_params_handle_t params,
    const char* name,
    int type,
    const void* value
) {
    if (!params || !name || !value) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto params_wrapper = reinterpret_cast<nmslib_params_wrapper_t*>(params);
        std::string param = std::string(name) + "=";
        switch (type) {
            case 0: param += std::to_string(*static_cast<const int*>(value)); break;
            case 1: param += std::to_string(*static_cast<const double*>(value)); break;
            case 2: param += static_cast<const char*>(value); break;
            default:
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid parameter type", __FILE__, __LINE__);
                return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
        params_wrapper->params.push_back(param);
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Parameter added successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to add parameter", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

void nmslib_free_params(nmslib_params_handle_t params) {
    if (!params || !reinterpret_cast<nmslib_params_wrapper_t*>(params)->allocator.free) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid params or allocator", __FILE__, __LINE__);
        return;
    }
    auto params_wrapper = reinterpret_cast<nmslib_params_wrapper_t*>(params);
    params_wrapper->~nmslib_params_wrapper_t();
    params_wrapper->allocator.free(params_wrapper, params_wrapper->allocator.ctx);
}

nmslib_error_t nmslib_get_space_type(
    nmslib_index_handle_t handle,
    const char** space_type,
    size_t* space_type_len,
    const nmslib_allocator_t* allocator
) {
    if (!handle || !space_type || !space_type_len || !allocator || !allocator->alloc || !allocator->free) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        *space_type_len = idx->space_type.length() + 1;
        *space_type = NMSLIBUtil::dup_string(idx->space_type, allocator);
        if (!*space_type) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for space type", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Space type retrieved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to get space type", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_get_method(
    nmslib_index_handle_t handle,
    const char** method,
    size_t* method_len,
    const nmslib_allocator_t* allocator
) {
    if (!handle || !method || !method_len || !allocator || !allocator->alloc || !allocator->free) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        *method_len = idx->method.length() + 1;
        *method = NMSLIBUtil::dup_string(idx->method, allocator);
        if (!*method) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for method", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Method retrieved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to get method", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

void nmslib_free_string(char* str, const nmslib_allocator_t* allocator) {
    if (!str || !allocator || !allocator->free) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid string or allocator", __FILE__, __LINE__);
        return;
    }
    allocator->free(str, allocator->ctx);
}

nmslib_error_t nmslib_get_last_error_detail(
    nmslib_error_detail_t* detail,
    const nmslib_allocator_t* allocator
) {
    if (!detail || !allocator || !allocator->alloc || !allocator->free) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid detail or allocator pointer", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    detail->code = last_error_detail.code;
    detail->message = NMSLIBUtil::dup_string(last_error_detail.message, allocator);
    if (!detail->message) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for error message", __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    }
    detail->file = NMSLIBUtil::dup_string(last_error_detail.file, allocator);
    if (!detail->file) {
        allocator->free(const_cast<char*>(detail->message), allocator->ctx);
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for error file", __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    }
    detail->line = last_error_detail.line;
    NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Error detail retrieved successfully", __FILE__, __LINE__);
    return NMSLIB_SUCCESS;
}

nmslib_error_t nmslib_add_data_point(
    nmslib_index_handle_t handle,
    const void* data,
    size_t element_count,
    int32_t id
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, data, element_count)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for adding data point", __FILE__, __LINE__);
        return err;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        std::unique_ptr<Object> obj;
        switch (idx->data_type) {
            case NMSLIB_DATATYPE_DENSE_VECTOR:
                if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                    auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                    if (!vectSpacePtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    std::vector<float> tempVect(static_cast<const float*>(data), static_cast<const float*>(data) + element_count);
                    obj.reset(vectSpacePtr->CreateObjFromVect(id, -1, tempVect));
                } else {
                    auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx->space.get());
                    if (!vectSpacePtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    std::vector<int> tempVect(static_cast<const int*>(data), static_cast<const int*>(data) + element_count);
                    obj.reset(vectSpacePtr->CreateObjFromVect(id, -1, tempVect));
                }
                break;
            case NMSLIB_DATATYPE_SPARSE_VECTOR:
                if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                    if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t*>(data), element_count, true)) {
                        NMSLIBUtil::set_last_error(err, "Invalid sparse elements", __FILE__, __LINE__);
                        return err;
                    }
                    auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                    if (!sparseSpacePtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    std::vector<SparseVectElem<float>> sparseVect;
                    for (size_t i = 0; i < element_count; ++i) {
                        sparseVect.emplace_back(static_cast<const nmslib_sparse_elem_float_t*>(data)[i].id, static_cast<const nmslib_sparse_elem_float_t*>(data)[i].value);
                    }
                    obj.reset(sparseSpacePtr->CreateObjFromVect(id, -1, sparseVect));
                } else {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                break;
            case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
                if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                    auto vectSiftPtr = dynamic_cast<SpaceL2SqrSift*>(idx->space.get());
                    if (!vectSiftPtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a uint8 vector space", __FILE__, __LINE__);
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    std::vector<uint8_t> tempVect(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + element_count);
                    obj.reset(vectSiftPtr->CreateObjFromUint8Vect(id, -1, tempVect));
                } else {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Uint8 vectors not supported with integer distances", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                break;
            case NMSLIB_DATATYPE_OBJECT_AS_STRING:
                if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                    obj.reset(idx->space->CreateObjFromStr(id, -1, static_cast<const char*>(data), nullptr).release());
                } else {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "String objects not supported with integer distances", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                break;
            default:
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        idx->data.push_back(obj.release());
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Data point added successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to add data point", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_add_data_point_batch(
    nmslib_index_handle_t handle,
    const void* data,
    size_t count,
    size_t element_count,
    const int32_t* ids,
    const size_t* num_elements
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, data, count)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for adding data points", __FILE__, __LINE__);
        return err;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        for (size_t i = 0; i < count; ++i) {
            std::unique_ptr<Object> obj;
            int32_t id = ids ? ids[i] : static_cast<int32_t>(idx->data.size());
            switch (idx->data_type) {
                case NMSLIB_DATATYPE_DENSE_VECTOR:
                    if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                        auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                        if (!vectSpacePtr) {
                            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        const float* row = static_cast<const float* const*>(data)[i];
                        std::vector<float> tempVect(row, row + element_count);
                        obj.reset(vectSpacePtr->CreateObjFromVect(id, -1, tempVect));
                    } else {
                        auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx->space.get());
                        if (!vectSpacePtr) {
                            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        const int* row = static_cast<const int* const*>(data)[i];
                        std::vector<int> tempVect(row, row + element_count);
                        obj.reset(vectSpacePtr->CreateObjFromVect(id, -1, tempVect));
                    }
                    break;
                case NMSLIB_DATATYPE_SPARSE_VECTOR:
                    if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                        if (!num_elements) {
                            NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_SPARSE_ELEMENT, "Missing num_elements for sparse vectors", __FILE__, __LINE__);
                            return NMSLIB_ERROR_INVALID_SPARSE_ELEMENT;
                        }
                        if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t* const*>(data)[i], num_elements[i], true)) {
                            NMSLIBUtil::set_last_error(err, "Invalid sparse elements", __FILE__, __LINE__);
                            return err;
                        }
                        auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                        if (!sparseSpacePtr) {
                            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        std::vector<SparseVectElem<float>> sparseVect;
                        const nmslib_sparse_elem_float_t* elements = static_cast<const nmslib_sparse_elem_float_t* const*>(data)[i];
                        for (size_t j = 0; j < num_elements[i]; ++j) {
                            sparseVect.emplace_back(elements[j].id, elements[j].value);
                        }
                        obj.reset(sparseSpacePtr->CreateObjFromVect(id, -1, sparseVect));
                    } else {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    break;
                default:
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Batch add not supported for this data type", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            idx->data.push_back(obj.release());  
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Data points added successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to add data points", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_add_data_point_batch_uint8(
    nmslib_index_handle_t handle,
    const unsigned char* data,
    size_t count,
    size_t element_count,
    const int32_t* ids
) {
    if (!handle || !data || count == 0 || element_count == 0 || !ids) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type != NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        for (size_t i = 0; i < count; ++i) {
            std::unique_ptr<Object> obj;
            int32_t id = ids[i];
            switch (idx->data_type) {
                case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
                    if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                        auto vectSiftPtr = dynamic_cast<SpaceL2SqrSift*>(idx->space.get());
                        if (!vectSiftPtr) {
                            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a uint8 vector space", __FILE__, __LINE__);
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        const unsigned char* row = static_cast<const unsigned char*>(data) + i * element_count;
                        std::vector<uint8_t> tempVect(row, row + element_count);
                        obj.reset(vectSiftPtr->CreateObjFromUint8Vect(id, -1, tempVect));
                    } else {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Uint8 vectors not supported with integer distances", __FILE__, __LINE__);
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    break;
                default:
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            idx->data.push_back(obj.release());
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Data points added successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to add data points", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_add_data_point_batch_string(
    nmslib_index_handle_t handle,
    const char* const* data,
    size_t count,
    const int32_t* ids
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, data, count)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for adding string data points", __FILE__, __LINE__);
        return err;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type != NMSLIB_DATATYPE_OBJECT_AS_STRING || idx->dist_type != NMSLIB_DISTTYPE_FLOAT) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type or distance type", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        for (size_t i = 0; i < count; ++i) {
            int32_t id = ids ? ids[i] : static_cast<int32_t>(idx->data.size());
            idx->data.push_back(idx->space->CreateObjFromStr(id, -1, data[i], nullptr).release());
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "String data points added successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to add string data points", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_knn_query_get_size(
    nmslib_index_handle_t handle,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    size_t* out_size,
    size_t num_elements
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, query, query_size_or_elem_count, out_size)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for k-NN query", __FILE__, __LINE__);
        return err;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        std::unique_ptr<Object> query_obj;
        if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<float> query_vector(static_cast<const float*>(query), static_cast<const float*>(query) + query_size_or_elem_count);
                query_obj.reset(vectSpacePtr->CreateObjFromVect(-1, -1, query_vector));
            } else {
                auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<int> query_vector(static_cast<const int*>(query), static_cast<const int*>(query) + query_size_or_elem_count);
                query_obj.reset(vectSpacePtr->CreateObjFromVect(-1, -1, query_vector));
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t*>(query), num_elements, true)) {
                    NMSLIBUtil::set_last_error(err, "Invalid sparse query elements", __FILE__, __LINE__);
                    return err;
                }
                auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                if (!sparseSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<SparseVectElem<float>> sparseVect;
                for (size_t i = 0; i < num_elements; ++i) {
                    sparseVect.emplace_back(static_cast<const nmslib_sparse_elem_float_t*>(query)[i].id, static_cast<const nmslib_sparse_elem_float_t*>(query)[i].value);
                }
                query_obj.reset(sparseSpacePtr->CreateObjFromVect(-1, -1, sparseVect));
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSiftPtr = dynamic_cast<SpaceL2SqrSift*>(idx->space.get());
                if (!vectSiftPtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a uint8 vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<uint8_t> tempVect(static_cast<const unsigned char*>(query), static_cast<const unsigned char*>(query) + query_size_or_elem_count);
                query_obj.reset(vectSiftPtr->CreateObjFromUint8Vect(-1, -1, tempVect));
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Uint8 vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_OBJECT_AS_STRING) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto obj = idx->space->CreateObjFromStr(-1, -1, static_cast<const char*>(query), nullptr);
                query_obj.reset(obj.release());
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "String objects not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        KNNQuery<float> knn(*idx->space, query_obj.get(), k);
        idx->index_ptr->Search(&knn, -1);
        *out_size = knn.Result()->Size();
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "k-NN query size retrieved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to retrieve k-NN query size", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_knn_query_fill(
    nmslib_index_handle_t handle,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* result,
    size_t num_elements
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, query, query_size_or_elem_count, result)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for k-NN query", __FILE__, __LINE__);
        return err;
    }
    if (!result->ids || !result->distances || result->capacity < k) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid result structure or insufficient capacity", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<float> tempVect(static_cast<const float*>(query), static_cast<const float*>(query) + query_size_or_elem_count);
                std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                KNNQuery<float> knn(*idx->space, queryObj.get(), k);
                idx->index_ptr->Search(&knn, -1);
                std::unique_ptr<KNNQueue<float>> results(knn.Result()->Clone());
                size_t result_size = results->Size();
                if (result_size > result->capacity) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                result->size = result_size;
                size_t i = 0;
                while (!results->Empty() && i < result->capacity) {
                    result->ids[i] = results->TopObject()->id();
                    result->distances[i] = results->TopDistance();
                    results->Pop();
                    ++i;
                }
            } else {
                auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<int> tempVect(static_cast<const int*>(query), static_cast<const int*>(query) + query_size_or_elem_count);
                std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                KNNQuery<int> knn(*idx_int->space, queryObj.get(), k);
                idx_int->index_ptr->Search(&knn, -1);
                std::unique_ptr<KNNQueue<int>> results(knn.Result()->Clone());
                size_t result_size = results->Size();
                if (result_size > result->capacity) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                result->size = result_size;
                size_t i = 0;
                while (!results->Empty() && i < result->capacity) {
                    result->ids[i] = results->TopObject()->id();
                    result->distances[i] = static_cast<float>(results->TopDistance());
                    results->Pop();
                    ++i;
                }
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t*>(query), query_size_or_elem_count, true)) {
                    NMSLIBUtil::set_last_error(err, "Invalid sparse query elements", __FILE__, __LINE__);
                    return err;
                }
                auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                if (!sparseSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<SparseVectElem<float>> sparseVect;
                for (size_t i = 0; i < query_size_or_elem_count; ++i) {
                    sparseVect.emplace_back(static_cast<const nmslib_sparse_elem_float_t*>(query)[i].id, static_cast<const nmslib_sparse_elem_float_t*>(query)[i].value);
                }
                std::unique_ptr<Object> queryObj(sparseSpacePtr->CreateObjFromVect(-1, -1, sparseVect));
                KNNQuery<float> knn(*idx->space, queryObj.get(), k);
                idx->index_ptr->Search(&knn, -1);
                std::unique_ptr<KNNQueue<float>> results(knn.Result()->Clone());
                size_t result_size = results->Size();
                if (result_size > result->capacity) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                result->size = result_size;
                size_t i = 0;
                while (!results->Empty() && i < result->capacity) {
                    result->ids[i] = results->TopObject()->id();
                    result->distances[i] = results->TopDistance();
                    results->Pop();
                    ++i;
                }
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for k-NN query", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "k-NN query executed successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to execute k-NN query", __FILE__, __LINE__);
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_knn_query_batch(
    nmslib_index_handle_t handle,
    const void* queries,
    size_t query_count,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* results,
    const size_t* num_elements,
    size_t thread_pool_size
) {
    if (!handle || !queries || query_count == 0 || !results || k == 0) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid inputs for batch k-NN query", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        ParallelFor(0, query_count, thread_pool_size, [&](size_t q, size_t threadId) {
            if (!results[q].ids || !results[q].distances || results[q].capacity < k) {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid result structure for query", __FILE__, __LINE__);
                throw std::runtime_error("Invalid result structure");
            }
            if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
                if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                    auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                    if (!vectSpacePtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                        throw std::runtime_error("Invalid space");
                    }
                    const float* query = static_cast<const float* const*>(queries)[q];
                    std::vector<float> tempVect(query, query + query_size_or_elem_count);
                    std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                    KNNQuery<float> knn(*idx->space, queryObj.get(), k);
                    idx->index_ptr->Search(&knn, -1);
                    std::unique_ptr<KNNQueue<float>> result_set(knn.Result()->Clone());
                    size_t result_size = result_set->Size();
                    if (result_size > results[q].capacity) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                        throw std::runtime_error("Buffer too small");
                    }
                    results[q].size = result_size;
                    size_t i = 0;
                    while (!result_set->Empty() && i < results[q].capacity) {
                        results[q].ids[i] = result_set->TopObject()->id();
                        results[q].distances[i] = result_set->TopDistance();
                        result_set->Pop();
                        ++i;
                    }
                } else {
                    auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                    auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                    if (!vectSpacePtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                        throw std::runtime_error("Invalid space");
                    }
                    const int* query = static_cast<const int* const*>(queries)[q];
                    std::vector<int> tempVect(query, query + query_size_or_elem_count);
                    std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                    KNNQuery<int> knn(*idx_int->space, queryObj.get(), k);
                    idx_int->index_ptr->Search(&knn, -1);
                    std::unique_ptr<KNNQueue<int>> result_set(knn.Result()->Clone());
                    size_t result_size = result_set->Size();
                    if (result_size > results[q].capacity) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                        throw std::runtime_error("Buffer too small");
                    }
                    results[q].size = result_size;
                    size_t i = 0;
                    while (!result_set->Empty() && i < results[q].capacity) {
                        results[q].ids[i] = result_set->TopObject()->id();
                        results[q].distances[i] = static_cast<float>(result_set->TopDistance());
                        result_set->Pop();
                        ++i;
                    }
                }
            } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
                if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                    if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t* const*>(queries)[q], num_elements ? num_elements[q] : query_size_or_elem_count, true)) {
                        NMSLIBUtil::set_last_error(err, "Invalid sparse query elements", __FILE__, __LINE__);
                        throw std::runtime_error("Invalid sparse elements");
                    }
                    auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                    if (!sparseSpacePtr) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                        throw std::runtime_error("Invalid space");
                    }
                    std::vector<SparseVectElem<float>> sparseVect;
                    const nmslib_sparse_elem_float_t* elements = static_cast<const nmslib_sparse_elem_float_t* const*>(queries)[q];
                    size_t elem_count = num_elements ? num_elements[q] : query_size_or_elem_count;
                    for (size_t i = 0; i < elem_count; ++i) {
                        sparseVect.emplace_back(elements[i].id, elements[i].value);
                    }
                    std::unique_ptr<Object> queryObj(sparseSpacePtr->CreateObjFromVect(-1, -1, sparseVect));
                    KNNQuery<float> knn(*idx->space, queryObj.get(), k);
                    idx->index_ptr->Search(&knn, -1);
                    std::unique_ptr<KNNQueue<float>> result_set(knn.Result()->Clone());
                    size_t result_size = result_set->Size();
                    if (result_size > results[q].capacity) {
                        NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                        throw std::runtime_error("Buffer too small");
                    }
                    results[q].size = result_size;
                    size_t i = 0;
                    while (!result_set->Empty() && i < results[q].capacity) {
                        results[q].ids[i] = result_set->TopObject()->id();
                        results[q].distances[i] = result_set->TopDistance();
                        result_set->Pop();
                        ++i;
                    }
                } else {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                    throw std::runtime_error("Sparse vectors not supported");
                }
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for batch k-NN query", __FILE__, __LINE__);
                throw std::runtime_error("Invalid data type");
            }
        });
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Batch k-NN query executed successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to execute batch k-NN query", __FILE__, __LINE__);
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_range_query_get_size(
    nmslib_index_handle_t handle,
    const void* query,
    size_t query_size_or_elem_count,
    double radius,
    size_t* out_size,
    size_t num_elements
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, query, query_size_or_elem_count, out_size)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for range query size", __FILE__, __LINE__);
        return err;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<float> tempVect(static_cast<const float*>(query), static_cast<const float*>(query) + query_size_or_elem_count);
                std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                RangeQuery<float> range(*idx->space, queryObj.get(), static_cast<float>(radius));
                idx->index_ptr->Search(&range, -1);
                *out_size = range.Result()->size();
            } else {
                auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<int> tempVect(static_cast<const int*>(query), static_cast<const int*>(query) + query_size_or_elem_count);
                std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                RangeQuery<int> range(*idx_int->space, queryObj.get(), static_cast<int>(radius));
                idx_int->index_ptr->Search(&range, -1);
                *out_size = range.Result()->size();
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t*>(query), query_size_or_elem_count, true)) {
                    NMSLIBUtil::set_last_error(err, "Invalid sparse query elements", __FILE__, __LINE__);
                    return err;
                }
                auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                if (!sparseSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<SparseVectElem<float>> sparseVect;
                for (size_t i = 0; i < query_size_or_elem_count; ++i) {
                    sparseVect.emplace_back(static_cast<const nmslib_sparse_elem_float_t*>(query)[i].id, static_cast<const nmslib_sparse_elem_float_t*>(query)[i].value);
                }
                std::unique_ptr<Object> queryObj(sparseSpacePtr->CreateObjFromVect(-1, -1, sparseVect));
                RangeQuery<float> range(*idx->space, queryObj.get(), static_cast<float>(radius));
                idx->index_ptr->Search(&range, -1);
                *out_size = range.Result()->size();
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for range query size", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Range query size retrieved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to execute range query size", __FILE__, __LINE__);
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_range_query_fill(
    nmslib_index_handle_t handle,
    const void* query,
    size_t query_size_or_elem_count,
    double radius,
    nmslib_result_t* result,
    size_t num_elements
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, query, query_size_or_elem_count, result)) {
        NMSLIBUtil::set_last_error(err, "Invalid inputs for range query", __FILE__, __LINE__);
        return err;
    }
    if (!result->ids || !result->distances || result->capacity == 0) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid result structure", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<float> tempVect(static_cast<const float*>(query), static_cast<const float*>(query) + query_size_or_elem_count);
                std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                RangeQuery<float> range(*idx->space, queryObj.get(), static_cast<float>(radius));
                idx->index_ptr->Search(&range, -1);
                const auto* results = range.Result();
                const auto* dists = range.ResultDists();
                size_t result_size = results->size();
                if (result_size > result->capacity) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                result->size = result_size;
                for (size_t i = 0; i < result_size && i < result->capacity; ++i) {
                    result->ids[i] = (*results)[i]->id();
                    result->distances[i] = (*dists)[i];
                }
            } else {
                auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<int> tempVect(static_cast<const int*>(query), static_cast<const int*>(query) + query_size_or_elem_count);
                std::unique_ptr<Object> queryObj(vectSpacePtr->CreateObjFromVect(-1, -1, tempVect));
                RangeQuery<int> range(*idx_int->space, queryObj.get(), static_cast<int>(radius));
                idx_int->index_ptr->Search(&range, -1);
                const auto* results = range.Result();
                const auto* dists = range.ResultDists();
                size_t result_size = results->size();
                if (result_size > result->capacity) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                result->size = result_size;
                for (size_t i = 0; i < result_size && i < result->capacity; ++i) {
                    result->ids[i] = (*results)[i]->id();
                    result->distances[i] = static_cast<float>((*dists)[i]);
                }
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(static_cast<const nmslib_sparse_elem_float_t*>(query), query_size_or_elem_count, true)) {
                    NMSLIBUtil::set_last_error(err, "Invalid sparse query elements", __FILE__, __LINE__);
                    return err;
                }
                auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                if (!sparseSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<SparseVectElem<float>> sparseVect;
                for (size_t i = 0; i < query_size_or_elem_count; ++i) {
                    sparseVect.emplace_back(static_cast<const nmslib_sparse_elem_float_t*>(query)[i].id, static_cast<const nmslib_sparse_elem_float_t*>(query)[i].value);
                }
                std::unique_ptr<Object> queryObj(sparseSpacePtr->CreateObjFromVect(-1, -1, sparseVect));
                RangeQuery<float> range(*idx->space, queryObj.get(), static_cast<float>(radius));
                idx->index_ptr->Search(&range, -1);
                const auto* results = range.Result();
                const auto* dists = range.ResultDists();
                size_t result_size = results->size();
                if (result_size > result->capacity) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffer too small", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                result->size = result_size;
                for (size_t i = 0; i < result_size && i < result->capacity; ++i) {
                    result->ids[i] = (*results)[i]->id();
                    result->distances[i] = (*dists)[i];
                }
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for range query", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Range query executed successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to execute range query", __FILE__, __LINE__);
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_get_distance(
    nmslib_index_handle_t handle,
    size_t pos1,
    size_t pos2,
    float* distance
) {
    if (!handle || !distance || pos1 >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size() || pos2 >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size()) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index or positions", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
            *distance = idx->space->IndexTimeDistance(idx->data[pos1], idx->data[pos2]);
        } else {
            auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
            *distance = static_cast<float>(idx_int->space->IndexTimeDistance(idx_int->data[pos1], idx_int->data[pos2]));
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Distance computed successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to compute distance", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_get_data_point_size(
    nmslib_index_handle_t handle,
    size_t position,
    size_t* size
) {
    if (!handle || !size || position >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size()) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index or position", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                *size = vectSpacePtr->GetElemQty(idx->data[position]);
            } else {
                auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                *size = vectSpacePtr->GetElemQty(idx_int->data[position]);
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
                if (!sparseSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                *size = sparseSpacePtr->GetElemQty(idx->data[position]);
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSiftPtr = dynamic_cast<SpaceL2SqrSift*>(idx->space.get());
                if (!vectSiftPtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a uint8 vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                *size = vectSiftPtr->GetElemQty(idx->data[position]);
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Uint8 vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_OBJECT_AS_STRING) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                *size = idx->data[position]->datalength();
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "String objects not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Data point size retrieved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to get data point size", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_get_data_point_fill(
    nmslib_index_handle_t handle,
    size_t position,
    void* data,
    size_t size
) {
    if (!handle || !data || position >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size()) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index, position, or output buffer", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                size_t elem_qty = vectSpacePtr->GetElemQty(idx->data[position]);
                if (size < elem_qty) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Buffer too small for dense vector", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                std::vector<float> tempVect(elem_qty);
                vectSpacePtr->CreateDenseVectFromObj(idx->data[position], tempVect.data(), elem_qty);
                std::copy(tempVect.begin(), tempVect.end(), static_cast<float*>(data));
            } else {
                auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
                if (!vectSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                size_t elem_qty = vectSpacePtr->GetElemQty(idx_int->data[position]);
                if (size < elem_qty) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Buffer too small for dense vector", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                std::vector<int> tempVect(elem_qty);
                vectSpacePtr->CreateDenseVectFromObj(idx_int->data[position], tempVect.data(), elem_qty);
                std::copy(tempVect.begin(), tempVect.end(), static_cast<int*>(data));
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto sparseSpacePtr = dynamic_cast<SpaceSparseVectorSimpleStorage<float>*>(idx->space.get());
                if (!sparseSpacePtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                const SparseVectElem<float>* beg = reinterpret_cast<const SparseVectElem<float>*>(idx->data[position]->data());
                const SparseVectElem<float>* end = reinterpret_cast<const SparseVectElem<float>*>(idx->data[position]->data() + idx->data[position]->datalength());
                size_t elem_qty = end - beg;
                if (size < elem_qty) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Buffer too small for sparse vector", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                for (size_t i = 0; i < elem_qty; ++i) {
                    static_cast<nmslib_sparse_elem_float_t*>(data)[i] = {beg[i].id_, beg[i].val_};
                }
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Sparse vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else if (idx->data_type == NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
            if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
                auto vectSiftPtr = dynamic_cast<SpaceL2SqrSift*>(idx->space.get());
                if (!vectSiftPtr) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a uint8 vector space", __FILE__, __LINE__);
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                size_t elem_qty = vectSiftPtr->GetElemQty(idx->data[position]);
                if (size < elem_qty) {
                    NMSLIBUtil::set_last_error(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Buffer too small for uint8 vector", __FILE__, __LINE__);
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }
                std::vector<int> tempVect(elem_qty);
                vectSiftPtr->CreateDenseVectFromObj(idx->data[position], tempVect.data(), elem_qty);
                for (size_t i = 0; i < elem_qty; ++i) {
                    static_cast<unsigned char*>(data)[i] = static_cast<uint8_t>(tempVect[i]);
                }
            } else {
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Uint8 vectors not supported with integer distances", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
        } else {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for data point fill", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Data point filled successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to fill data point", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_get_data_point_string(
    nmslib_index_handle_t handle,
    size_t position,
    const char** data,
    size_t* data_len,
    const nmslib_allocator_t* allocator
) {
    if (!handle || !data || !data_len || !allocator || !allocator->alloc || !allocator->free || position >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size()) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index, position, or allocator", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type != NMSLIB_DATATYPE_OBJECT_AS_STRING || idx->dist_type != NMSLIB_DISTTYPE_FLOAT) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type or distance type", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        *data_len = idx->data[position]->datalength() + 1;
        *data = NMSLIBUtil::dup_string(std::string(static_cast<const char*>(idx->data[position]->data()), idx->data[position]->datalength()), allocator);
        if (!*data) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for string data", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "String data point retrieved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to get string data point", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_borrow_data_dense(
    nmslib_index_handle_t handle,
    size_t position,
    void** data,
    size_t* size,
    void (**free_fn)(void*)
) {
    if (!handle || !data || !size || !free_fn || position >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size()) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index, position, or output pointers", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type != NMSLIB_DATATYPE_DENSE_VECTOR) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for dense borrow", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }

        // Allocate the wrapper structure
        nmslib_borrowed_data_t* borrowed = static_cast<nmslib_borrowed_data_t*>(
            idx->allocator.alloc(sizeof(nmslib_borrowed_data_t), idx->allocator.ctx));
        if (!borrowed) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for wrapper", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }

        if (idx->dist_type == NMSLIB_DISTTYPE_FLOAT) {
            auto vectSpacePtr = dynamic_cast<VectorSpace<float>*>(idx->space.get());
            if (!vectSpacePtr) {
                idx->allocator.free(borrowed, idx->allocator.ctx);
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            *size = vectSpacePtr->GetElemQty(idx->data[position]);
            float* tempData = static_cast<float*>(idx->allocator.alloc(*size * sizeof(float), idx->allocator.ctx));
            if (!tempData) {
                idx->allocator.free(borrowed, idx->allocator.ctx);
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for data copy", __FILE__, __LINE__);
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            std::vector<float> tempVect(*size);
            vectSpacePtr->CreateDenseVectFromObj(idx->data[position], tempVect.data(), *size);
            std::copy(tempVect.begin(), tempVect.end(), tempData);
            borrowed->data = tempData;
            borrowed->allocator = idx->allocator;
            *data = tempData;
        } else {
            auto idx_int = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
            auto vectSpacePtr = dynamic_cast<VectorSpace<int>*>(idx_int->space.get());
            if (!vectSpacePtr) {
                idx->allocator.free(borrowed, idx->allocator.ctx);
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a vector space", __FILE__, __LINE__);
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            *size = vectSpacePtr->GetElemQty(idx_int->data[position]);
            int* tempData = static_cast<int*>(idx_int->allocator.alloc(*size * sizeof(int), idx_int->allocator.ctx));
            if (!tempData) {
                idx->allocator.free(borrowed, idx->allocator.ctx);
                NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for data copy", __FILE__, __LINE__);
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            std::vector<int> tempVect(*size);
            vectSpacePtr->CreateDenseVectFromObj(idx_int->data[position], tempVect.data(), *size);
            std::copy(tempVect.begin(), tempVect.end(), tempData);
            borrowed->data = tempData;
            borrowed->allocator = idx_int->allocator;
            *data = tempData;
        }

        *free_fn = nmslib_borrowed_data_free;
        *data = borrowed->data;  // Return the data pointer
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Dense data borrowed successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to borrow dense data", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_borrow_data_sparse(
    nmslib_index_handle_t handle,
    size_t position,
    void** data,
    size_t* size,
    void (**free_fn)(void*)
) {
    if (!handle || !data || !size || !free_fn || position >= reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size()) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index, position, or output pointers", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type != NMSLIB_DATATYPE_SPARSE_VECTOR || idx->dist_type != NMSLIB_DISTTYPE_FLOAT) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type or distance type", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        auto sparseSpacePtr = dynamic_cast<SpaceSparseVector<float>*>(idx->space.get());
        if (!sparseSpacePtr) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Space is not a sparse vector space", __FILE__, __LINE__);
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }

        // Allocate the wrapper structure
        nmslib_borrowed_data_t* borrowed = static_cast<nmslib_borrowed_data_t*>(
            idx->allocator.alloc(sizeof(nmslib_borrowed_data_t), idx->allocator.ctx));
        if (!borrowed) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for wrapper", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }

        *size = sparseSpacePtr->GetElemQty(idx->data[position]);
        nmslib_sparse_elem_float_t* tempData = static_cast<nmslib_sparse_elem_float_t*>(
            idx->allocator.alloc(*size * sizeof(nmslib_sparse_elem_float_t), idx->allocator.ctx));
        if (!tempData) {
            idx->allocator.free(borrowed, idx->allocator.ctx);
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for sparse data copy", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        const SparseVectElem<float>* beg = reinterpret_cast<const SparseVectElem<float>*>(idx->data[position]->data());
        for (size_t i = 0; i < *size; ++i) {
            tempData[i] = {beg[i].id_, beg[i].val_};
        }
        borrowed->data = tempData;
        borrowed->allocator = idx->allocator;
        *data = tempData;
        *free_fn = nmslib_borrowed_data_free;
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Sparse data borrowed successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to borrow sparse data", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}


nmslib_error_t nmslib_save_index(
    nmslib_index_handle_t handle,
    const char* path,
    int save_data
) {
    if (!handle || !path) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index or path", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (!idx->index_ptr) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Index not created", __FILE__, __LINE__);
            return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
        if (save_data) {
            std::vector<std::string> dummy;
            idx->space->WriteObjectVectorBinData(idx->data, dummy, std::string(path) + ".dat");
        }
        idx->index_ptr->SaveIndex(path);
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Index saved successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_DATA_IO_FAILED, "Failed to save index", __FILE__, __LINE__);
        return NMSLIB_ERROR_DATA_IO_FAILED;
    }
}

nmslib_error_t nmslib_load_index(
    const char* path,
    nmslib_data_type_t data_type,
    nmslib_dist_type_t dist_type,
    const nmslib_allocator_t* allocator,
    int load_data,
    nmslib_index_handle_t* out_handle
) {
    if (!path || !allocator || !allocator->alloc || !allocator->free || !out_handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid path, allocator, or error pointer", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        void* mem = allocator->alloc(sizeof(nmslib_internal_index_t<float>), allocator->ctx);
        if (!mem) {
            NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index", __FILE__, __LINE__);
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        nmslib_index_handle_t result = nullptr;
        if (dist_type == NMSLIB_DISTTYPE_FLOAT) {
            auto idx = new (mem) nmslib_internal_index_t<float>("", "", data_type, dist_type, allocator);
            auto factory = MethodFactoryRegistry<float>::Instance();
            bool print_progress = false;
            idx->index_ptr.reset(factory.CreateMethod(print_progress, idx->method, idx->space_type, *idx->space, idx->data));
            if (load_data) {
                std::vector<std::string> dummy;
                idx->data.clear();
                idx->space->ReadObjectVectorFromBinData(idx->data, dummy, std::string(path) + ".dat");
            }
            idx->index_ptr->LoadIndex(path);
            idx->index_ptr->ResetQueryTimeParams();
            result = reinterpret_cast<nmslib_index_handle_t>(idx);
        } else {
            auto idx = new (mem) nmslib_internal_index_t<int>("", "", data_type, dist_type, allocator);
            auto factory = MethodFactoryRegistry<int>::Instance();
            bool print_progress = false;
            idx->index_ptr.reset(factory.CreateMethod(print_progress, idx->method, idx->space_type, *idx->space, idx->data));
            if (load_data) {
                std::vector<std::string> dummy;
                idx->data.clear();
                idx->space->ReadObjectVectorFromBinData(idx->data, dummy, std::string(path) + ".dat");
            }
            idx->index_ptr->LoadIndex(path);
            idx->index_ptr->ResetQueryTimeParams();
            result = reinterpret_cast<nmslib_index_handle_t>(idx);
        }
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Index loaded successfully", __FILE__, __LINE__);
        *out_handle = result;
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()), __FILE__, __LINE__);
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_DATA_IO_FAILED, "Failed to load index", __FILE__, __LINE__);
        return NMSLIB_ERROR_DATA_IO_FAILED;
    }
}

nmslib_error_t nmslib_set_query_time_params(
    nmslib_index_handle_t handle,
    nmslib_params_handle_t params
) {
    if (!handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        idx->index_ptr->SetQueryTimeParams(load_params(reinterpret_cast<nmslib_params_wrapper_t*>(params)));
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Query time parameters set successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to set query time parameters", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_set_thread_pool_size(
    nmslib_index_handle_t handle,
    size_t size
) {
    if (!handle || size == 0 || size > 1024) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index or thread pool size", __FILE__, __LINE__);
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        idx->thread_pool_size = size;
        NMSLIBUtil::set_last_error(NMSLIB_SUCCESS, "Thread pool size set successfully", __FILE__, __LINE__);
        return NMSLIB_SUCCESS;
    } catch (...) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_RUNTIME, "Failed to set thread pool size", __FILE__, __LINE__);
        return NMSLIB_ERROR_RUNTIME;
    }
}

size_t nmslib_get_thread_pool_size(nmslib_index_handle_t handle) {
    if (!handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index", __FILE__, __LINE__);
        return std::thread::hardware_concurrency();
    }
    return reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->thread_pool_size;
}

size_t nmslib_data_qty(nmslib_index_handle_t handle) {
    if (!handle) {
        NMSLIBUtil::set_last_error(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index", __FILE__, __LINE__);
        return 0;
    }
    return reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size();
}

} // extern "C"