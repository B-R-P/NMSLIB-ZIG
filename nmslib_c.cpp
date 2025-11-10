#include "nmslib_c.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <type_traits>
#include <functional>
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
#include "init.h"
#include "method/hnsw.h"      // defines similarity::Hnsw

static std::once_flag nmslib_init_flag;

static void nmslib_do_init() {
    // seed = 0, no logging (LIB_LOGNONE). Change if you want logging.
    similarity::initLibrary(0, LIB_LOGNONE, nullptr);
}
using namespace similarity;

// Macro for setting last error (for brevity)
#define SET_LAST_ERROR(code, msg) NMSLIBUtil::set_last_error(code, msg, __FILE__, __LINE__)

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

    inline nmslib_error_t validate_pointer_batch(const void *const *data_ptrs, size_t count, size_t element_count) {
        if (!data_ptrs) return NMSLIB_ERROR_INVALID_ARGUMENT;
        if (count == 0) return NMSLIB_ERROR_INVALID_ARGUMENT;
        for (size_t i = 0; i < count; ++i) {
            if (!data_ptrs[i]) return NMSLIB_ERROR_NULL_POINTER;
            if (element_count > 0 && element_count < 1) return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
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

class BorrowedObject : public Object {
private:
    const void* borrowed_ptr_;
    size_t buf_len_;

public:
    BorrowedObject(int id, const void* ptr, size_t len)
        : Object(static_cast<IdType>(id), static_cast<LabelType>(-1), len, ptr),
          borrowed_ptr_(ptr), buf_len_(len) {}

    void* data() const { return const_cast<void*>(borrowed_ptr_); }
    size_t bufferlength() const { return buf_len_; }
    int datalength() const { return static_cast<int>(buf_len_); }
    IdType id() const { return Object::id(); }
    ~BorrowedObject() {}
};

// Wrapper structure for NMSLIB parameters
struct nmslib_params_wrapper_t {
    std::vector<std::string> params;
    nmslib_allocator_t allocator;
};

template <typename dist_t>
struct nmslib_internal_index_t {
    // NEW header — must be first field to guarantee ABI-safe header reading
    nmslib_index_header_t header;

    // existing fields (kept for compatibility)
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
        : header{dt, dst}, space(), index_ptr(), data(), data_type(dt), dist_type(dst), method(m), space_type(st), allocator(*alloc), thread_pool_size(std::thread::hardware_concurrency())
    {}

    ~nmslib_internal_index_t() {
        for (auto datum : data) {
            delete datum;
        }
        data.clear();
    }
};

static AnyParams load_params(const nmslib_params_wrapper_t* params) {
    if (!params) return AnyParams();
    return AnyParams(params->params);
}

// Dispatch helper - templated to support different return types
template <typename Fn>
auto dispatch_index_by_data_type(nmslib_index_handle_t handle, Fn&& fn) -> std::invoke_result_t<Fn, nmslib_internal_index_t<float>*> {
    using RetT = std::invoke_result_t<Fn, nmslib_internal_index_t<float>*>;
    if (!handle) {
        if constexpr (std::is_same_v<RetT, nmslib_error_t>) {
            SET_LAST_ERROR(NMSLIB_ERROR_NULL_POINTER, "Null index handle");
            return NMSLIB_ERROR_NULL_POINTER;
        } else if constexpr (std::is_same_v<RetT, size_t>) {
            return static_cast<size_t>(0);
        } else if constexpr (std::is_void_v<RetT>) {
            return;
        } else {
            static_assert(false, "Unsupported return type in dispatch");
        }
    }

    auto hdr = reinterpret_cast<nmslib_index_header_t*>(handle);
    switch (hdr->data_type) {
        case NMSLIB_DATATYPE_DENSE_VECTOR:
        case NMSLIB_DATATYPE_SPARSE_VECTOR: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
            return std::forward<Fn>(fn)(idx);
        }
        case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
        case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
            return std::forward<Fn>(fn)(idx);
        }
        default: {
            if constexpr (std::is_same_v<RetT, nmslib_error_t>) {
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Unsupported data type");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            } else if constexpr (std::is_same_v<RetT, size_t>) {
                return static_cast<size_t>(0);
            } else if constexpr (std::is_void_v<RetT>) {
                return;
            } else {
                static_assert(false, "Unsupported return type in dispatch");
            }
        }
    }
}

// Consolidated helper for creating query/data objects
// Consolidated helper for creating query/data objects
template <typename dist_t>
std::unique_ptr<Object> create_object(const Space<dist_t>* space, nmslib_data_type_t data_type, const void* data, size_t elem_count, size_t num_elements, int32_t id) {
    std::unique_ptr<Object> obj;
    switch (data_type) {
        case NMSLIB_DATATYPE_DENSE_VECTOR: {
            const float* vec = static_cast<const float*>(data);
            auto vectSpace = dynamic_cast<const VectorSpaceSimpleStorage<float>*>(space);
            if (!vectSpace) return nullptr;
            std::vector<float> tempVec(vec, vec + elem_count);
            obj.reset(vectSpace->CreateObjFromVect(static_cast<IdType>(id), static_cast<LabelType>(-1), tempVec));
            break;
        }
        case NMSLIB_DATATYPE_SPARSE_VECTOR: {
            const nmslib_sparse_elem_float_t* elems = static_cast<const nmslib_sparse_elem_float_t*>(data);
            if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems, num_elements, true)) {
                SET_LAST_ERROR(err, "Invalid sparse elements");
                return nullptr;
            }
            std::vector<SparseVectElem<float>> tempVec(num_elements);
            for (size_t j = 0; j < num_elements; ++j) {
                tempVec[j].id_ = elems[j].id;
                tempVec[j].val_ = elems[j].value;
            }
            auto sparseSpace = dynamic_cast<const SpaceSparseVectorSimpleStorage<float>*>(space);
            if (!sparseSpace) return nullptr;
            obj.reset(sparseSpace->CreateObjFromVect(static_cast<IdType>(id), static_cast<LabelType>(-1), tempVec));
            break;
        }
        case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR: {
            const uint8_t* u8data = static_cast<const uint8_t*>(data);
            auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(space);
            if (!siftSpace) return nullptr;
            std::vector<uint8_t> u8vec(u8data, u8data + elem_count);
            obj.reset(siftSpace->CreateObjFromUint8Vect(static_cast<IdType>(id), static_cast<LabelType>(-1), u8vec));
            break;
        }
        case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
            const char* str_data = static_cast<const char*>(data);
            std::string str(str_data, elem_count > 0 ? elem_count - 1 : std::strlen(str_data));
            auto temp_obj = space->CreateObjFromStr(static_cast<IdType>(id), static_cast<LabelType>(-1), str, nullptr);
            if (temp_obj) {
                obj = std::move(temp_obj);
            }
            break;
        }
        default:
            return nullptr;
    }
    return obj;
}
// Helper for KNN result extraction
template <typename dist_t>
void extract_knn_results(const KNNQueue<dist_t>* res, nmslib_result_t* result) {
    if (!res) {
        result->size = 0;
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Query result missing");
        return;
    }
    size_t found = res->Size();
    result->size = found;
    if (found == 0) {
        SET_LAST_ERROR(NMSLIB_SUCCESS, "No neighbors found");
        return;
    }
    if (found > result->capacity) {
        SET_LAST_ERROR(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffers too small for " + std::to_string(found));
        result->size = 0;
        return;
    }
    auto cloneQueue = res->Clone();
    std::vector<std::pair<int32_t, float>> tmp; tmp.reserve(found);
    while (!cloneQueue->Empty()) {
        float d = static_cast<float>(cloneQueue->TopDistance());
        const Object* o = cloneQueue->TopObject();
        tmp.emplace_back(static_cast<int32_t>(o->id()), d);
        cloneQueue->Pop();
    }
    delete cloneQueue;
    std::reverse(tmp.begin(), tmp.end());
    for (size_t i = 0; i < found; ++i) {
        result->ids[i] = tmp[i].first;
        result->distances[i] = tmp[i].second;
    }
}

static const AnyParams defaultQueryParams = AnyParams({"efSearch=200"});

extern "C" {
__attribute__((constructor))
static void nmslib_force_space_registry_init() {
    (void) similarity::SpaceFactoryRegistry<float>::Instance();
    (void) similarity::SpaceFactoryRegistry<int>::Instance();
    (void) similarity::SpaceFactoryRegistry<size_t>::Instance();
}

void nmslib_init(void) {
    std::call_once(nmslib_init_flag, nmslib_do_init);
}

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
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }

    std::call_once(nmslib_init_flag, nmslib_do_init);

    try {
        {
            void* mem = allocator->alloc(sizeof(nmslib_internal_index_t<float>), allocator->ctx);
            if (!mem) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }

            auto idx = new (mem) nmslib_internal_index_t<float>(method, space, data_type, dist_type, allocator);
            bool created = false;
            try {
                idx->space.reset(SpaceFactoryRegistry<float>::Instance().CreateSpace(
                    space,
                    load_params(reinterpret_cast<nmslib_params_wrapper_t*>(space_params))
                ));
                if (idx->space) {
                    *out_handle = reinterpret_cast<nmslib_index_handle_t>(idx);
                    SET_LAST_ERROR(NMSLIB_SUCCESS, "Index created using float registry");
                    return NMSLIB_SUCCESS;
                }
            } catch (const std::exception& e) {
            } catch (...) {
            }

            idx->~nmslib_internal_index_t();
            allocator->free(mem, allocator->ctx);
        }

        {
            void* mem = allocator->alloc(sizeof(nmslib_internal_index_t<int>), allocator->ctx);
            if (!mem) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index (fallback)");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }

            auto idx_i = new (mem) nmslib_internal_index_t<int>(method, space, data_type, dist_type, allocator);
            try {
                idx_i->space.reset(SpaceFactoryRegistry<int>::Instance().CreateSpace(
                    space,
                    load_params(reinterpret_cast<nmslib_params_wrapper_t*>(space_params))
                ));
            } catch (const std::exception& e) {
                idx_i->~nmslib_internal_index_t();
                allocator->free(mem, allocator->ctx);
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, std::string("Failed to create space (int fallback): ") + e.what());
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            } catch (...) {
                idx_i->~nmslib_internal_index_t();
                allocator->free(mem, allocator->ctx);
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Failed to create space (int fallback unknown error)");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }

            if (!idx_i->space) {
                idx_i->~nmslib_internal_index_t();
                allocator->free(mem, allocator->ctx);
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "No compatible space found in float or int registry");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }

            *out_handle = reinterpret_cast<nmslib_index_handle_t>(idx_i);
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Index created using int registry fallback");
            return NMSLIB_SUCCESS;
        }
    } catch (const std::bad_alloc& e) {
        SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()));
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create index: " + std::string(e.what()));
        return NMSLIB_ERROR_RUNTIME;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create index (unknown error)");
        return NMSLIB_ERROR_RUNTIME;
    }
}


void nmslib_index_destroy(nmslib_index_handle_t handle) {
    if (!handle) return;

    auto hdr = reinterpret_cast<nmslib_index_header_t*>(handle);
    switch (hdr->data_type) {
        case NMSLIB_DATATYPE_DENSE_VECTOR:
        case NMSLIB_DATATYPE_SPARSE_VECTOR: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
            idx->~nmslib_internal_index_t<float>();
            idx->allocator.free(idx, idx->allocator.ctx);
            break;
        }
        case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
        case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
            idx->~nmslib_internal_index_t<int>();
            idx->allocator.free(idx, idx->allocator.ctx);
            break;
        }
        default: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
            idx->~nmslib_internal_index_t<float>();
            idx->allocator.free(idx, idx->allocator.ctx);
            break;
        }
    }
}


nmslib_error_t nmslib_create_index(
    nmslib_index_handle_t handle,
    nmslib_params_handle_t index_params,
    int print_progress
) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }

    return dispatch_index_by_data_type(handle, [index_params, print_progress](auto* idx) -> nmslib_error_t {
        using dist_t = decltype(idx->space->IndexTimeDistance(static_cast<const Object*>(nullptr), static_cast<const Object*>(nullptr)));
        try {
            auto factory = MethodFactoryRegistry<dist_t>::Instance();
            idx->index_ptr.reset(factory.CreateMethod(print_progress != 0, idx->method, idx->space_type, *idx->space, idx->data));
            idx->index_ptr->CreateIndex(load_params(reinterpret_cast<nmslib_params_wrapper_t*>(index_params)));
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Index created successfully");
            return NMSLIB_SUCCESS;
        } catch (const std::bad_alloc& e) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()));
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        } catch (const std::exception& e) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Failed to create index: " + std::string(e.what()));
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Failed to create index (unknown error)");
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        }
    });
}


nmslib_error_t nmslib_reset_index(nmslib_index_handle_t handle) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [](auto* idx) -> nmslib_error_t {
        try {
            for (auto datum : idx->data) {
                delete datum;
            }
            idx->data.clear();
            idx->index_ptr.reset();
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Index reset successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to reset index");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_params_handle_t nmslib_create_params(const nmslib_allocator_t* allocator) {
    if (!allocator || !allocator->alloc || !allocator->free) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid allocator");
        return nullptr;
    }
    try {
        void* mem = allocator->alloc(sizeof(nmslib_params_wrapper_t), allocator->ctx);
        if (!mem) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for params");
            return nullptr;
        }
        auto params = new (mem) nmslib_params_wrapper_t{};
        params->params.reserve(4);
        params->allocator = *allocator;
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Parameters created successfully");
        return reinterpret_cast<nmslib_params_handle_t>(params);
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to create params");
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
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments");
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
                SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid parameter type");
                return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
        params_wrapper->params.push_back(param);
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Parameter added successfully");
        return NMSLIB_SUCCESS;
    } catch (const std::bad_alloc& e) {
        SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()));
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to add parameter");
        return NMSLIB_ERROR_RUNTIME;
    }
}

void nmslib_free_params(nmslib_params_handle_t params) {
    if (!params || !reinterpret_cast<nmslib_params_wrapper_t*>(params)->allocator.free) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid params or allocator");
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
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [space_type, space_type_len, allocator](auto* idx) -> nmslib_error_t {
        try {
            *space_type_len = idx->space_type.length();
            *space_type = NMSLIBUtil::dup_string(idx->space_type, allocator);
            if (!*space_type) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for space type");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Space type retrieved successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to get space type");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}


nmslib_error_t nmslib_get_method(
    nmslib_index_handle_t handle,
    const char** method,
    size_t* method_len,
    const nmslib_allocator_t* allocator
) {
    if (!handle || !method || !method_len || !allocator || !allocator->alloc || !allocator->free) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid arguments");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [method, method_len, allocator](auto* idx) -> nmslib_error_t {
        try {
            *method_len = idx->method.length();
            *method = NMSLIBUtil::dup_string(idx->method, allocator);
            if (!*method) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for method");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Method retrieved successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to get method");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}


void nmslib_free_string(char* str, const nmslib_allocator_t* allocator) {
    if (!str || !allocator || !allocator->free) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid string or allocator");
        return;
    }
    allocator->free(str, allocator->ctx);
}

nmslib_error_t nmslib_get_last_error_detail(
    nmslib_error_detail_t* detail,
    const nmslib_allocator_t* allocator
) {
    if (!detail || !allocator || !allocator->alloc || !allocator->free) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid detail or allocator pointer");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        detail->code = last_error_detail.code;
        detail->message = NMSLIBUtil::dup_string(last_error_detail.message, allocator);
        if (!detail->message) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for error message");
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        detail->file = NMSLIBUtil::dup_string(last_error_detail.file, allocator);
        if (!detail->file) {
            allocator->free(const_cast<char*>(detail->message), allocator->ctx);
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate memory for error file");
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        }
        detail->line = last_error_detail.line;
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Error detail retrieved successfully");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to get last error detail");
        return NMSLIB_ERROR_RUNTIME;
    }
}

nmslib_error_t nmslib_add_data_point(
    nmslib_index_handle_t handle,
    const void* data,
    size_t element_count,
    int32_t id
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(handle, data, element_count)) {
        SET_LAST_ERROR(err, "Invalid inputs for adding data point");
        return err;
    }
    return dispatch_index_by_data_type(handle, [data, element_count, id](auto* idx) -> nmslib_error_t {
        try {
            auto obj = create_object(idx->space.get(), idx->data_type, data, element_count, idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR ? element_count : 0, id);
            if (!obj) {
                SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create object");
                return NMSLIB_ERROR_RUNTIME;
            }
            idx->data.push_back(obj.release());
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Data point added successfully");
            return NMSLIB_SUCCESS;
        } catch (const std::bad_alloc& e) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()));
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to add data point");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}


nmslib_error_t nmslib_add_data_point_batch(
    nmslib_index_handle_t index,
    const void* data,
    size_t count,
    size_t element_count,
    const int32_t* ids,
    const size_t* num_elements
) {
    if (!index || !data || count == 0 || element_count == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid batch inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [data, count, element_count, ids, num_elements](auto* idx) -> nmslib_error_t {
        try {
            size_t offset = 0;
            for (size_t i = 0; i < count; ++i) {
                int32_t curr_id_int = ids ? ids[i] : static_cast<int32_t>(i);
                size_t curr_num_elements = (num_elements && idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) ? num_elements[i] : element_count;
                size_t stride;
                size_t effective_dim = element_count;
                if (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) {
                    stride = curr_num_elements * sizeof(nmslib_sparse_elem_float_t);
                    effective_dim = 0;
                } else if (idx->data_type == NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
                    stride = element_count * sizeof(uint8_t);
                } else {
                    stride = element_count * sizeof(float);
                }
                const void* curr_data = static_cast<const char*>(data) + offset;
                auto obj = create_object(idx->space.get(), idx->data_type, curr_data, effective_dim, curr_num_elements, curr_id_int);
                if (!obj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create batch object");
                    return NMSLIB_ERROR_RUNTIME;
                }
                idx->data.push_back(obj.release());
                offset += stride;
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Batch added successfully");
            return NMSLIB_SUCCESS;
        } catch (const std::bad_alloc& e) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Batch alloc failed: " + std::string(e.what()));
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to add batch");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_add_data_point_batch_uint8(
    nmslib_index_handle_t handle,
    const unsigned char* data,
    size_t count,
    size_t element_count,
    const int32_t* ids
) {
    if (!handle || !data || count == 0 || element_count == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid uint8 batch inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [data, count, element_count, ids](auto* idx) -> nmslib_error_t {
        try {
            if (idx->data_type != NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not uint8 vector space");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
            if (!siftSpace) {
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not SIFT space");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            for (size_t i = 0; i < count; ++i) {
                int32_t curr_id_int = ids ? ids[i] : static_cast<int32_t>(i);
                const uint8_t* vec_start = data + i * element_count;
                std::vector<uint8_t> tempVec(vec_start, vec_start + element_count);
                Object* raw_obj = siftSpace->CreateObjFromUint8Vect(static_cast<IdType>(curr_id_int), static_cast<LabelType>(-1), tempVec);
                if (!raw_obj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to create uint8 object");
                    return NMSLIB_ERROR_OUT_OF_MEMORY;
                }
                idx->data.push_back(raw_obj);
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "UInt8 batch added successfully");
            return NMSLIB_SUCCESS;
        } catch (const std::bad_alloc& e) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Memory allocation failed: " + std::string(e.what()));
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to add uint8 batch");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_add_data_point_batch_string(
    nmslib_index_handle_t index,
    const char* const* data,
    size_t count,
    const int32_t* ids
) {
    if (!index || !data || count == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid string batch inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }

    auto hdr = reinterpret_cast<nmslib_index_header_t*>(index);
    if (hdr->data_type != NMSLIB_DATATYPE_OBJECT_AS_STRING) {
        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not string space");
        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
    }

    return dispatch_index_by_data_type(index, [data, count, ids](auto* idx) -> nmslib_error_t {
        try {
            for (size_t i = 0; i < count; ++i) {
                if (!data[i]) {
                    SET_LAST_ERROR(NMSLIB_ERROR_NULL_POINTER, "Null string in batch");
                    return NMSLIB_ERROR_NULL_POINTER;
                }
                std::string str(data[i]);
                std::unique_ptr<Object> obj(idx->space->CreateObjFromStr(ids ? ids[i] : static_cast<int32_t>(i), -1, str, nullptr));
                if (!obj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create string object");
                    return NMSLIB_ERROR_RUNTIME;
                }
                idx->data.push_back(obj.release());
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "String batch added successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to add string batch");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}


nmslib_error_t nmslib_knn_query_get_size(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    size_t* out_size,
    size_t num_elements
) {
    if (!index || !query || k == 0 || !out_size) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid knn query inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        *out_size = k;
        SET_LAST_ERROR(NMSLIB_SUCCESS, "KNN size retrieved");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to get knn size");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_knn_query_fill(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* result,
    size_t num_elements
) {
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(index, query, query_size_or_elem_count, result)) {
        SET_LAST_ERROR(err, "Invalid KNN query inputs");
        return err;
    }
    if (!result->ids || !result->distances || result->capacity == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Result buffers invalid");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }

    return dispatch_index_by_data_type(index, [query, query_size_or_elem_count, k, result, num_elements](auto* idx) -> nmslib_error_t {
        using dist_t = decltype(idx->space->IndexTimeDistance(static_cast<const Object*>(nullptr), static_cast<const Object*>(nullptr)));
        if (!idx->index_ptr) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built");
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        }

        try {
            size_t effective_num_elements = (idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) ? num_elements : 0;
            auto qobj = create_object(idx->space.get(), idx->data_type, query, query_size_or_elem_count, effective_num_elements, 0);
            if (!qobj) {
                SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Failed to create query object");
                result->size = 0;
                return NMSLIB_ERROR_INVALID_ARGUMENT;
            }

            similarity::KNNQuery<dist_t> knnQuery(*idx->space, qobj.get(), k);
            idx->index_ptr->SetQueryTimeParams(defaultQueryParams);
            idx->index_ptr->Search(&knnQuery);
            extract_knn_results(knnQuery.Result(), result);
            SET_LAST_ERROR(NMSLIB_SUCCESS, "KNN query filled successfully");
            return NMSLIB_SUCCESS;
        } catch (const std::bad_alloc& e) {
            SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Query alloc failed: " + std::string(e.what()));
            return NMSLIB_ERROR_OUT_OF_MEMORY;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "KNN query failed");
            return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
        }
    });
}

nmslib_error_t nmslib_knn_query_batch(
    nmslib_index_handle_t index,
    const void* queries,
    size_t query_count,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* results,
    const size_t* num_elements,
    size_t thread_pool_size
) {
    if (!index || !queries || query_count == 0 || !results) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid batch knn inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        for (size_t i = 0; i < query_count; ++i) {
            nmslib_error_t err = nmslib_knn_query_fill(index, static_cast<const char*>(queries) + i * query_size_or_elem_count * sizeof(float), query_size_or_elem_count, k, &results[i], num_elements ? num_elements[i] : 0);
            if (err != NMSLIB_SUCCESS) return err;
        }
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Batch knn query executed");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to execute batch knn query");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_range_query_get_size(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    double radius,
    size_t* out_size,
    size_t num_elements
) {
    if (!index || !query || radius < 0 || !out_size) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid range query inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        *out_size = 100;  // Heuristic
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Range size retrieved");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to get range size");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_range_query_fill(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    double radius,
    nmslib_result_t* result,
    size_t num_elements
) {
    if (!index || !query || !result || result->capacity == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid range fill inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        result->size = 0;  // Placeholder
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Range query filled");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to fill range query");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

nmslib_error_t nmslib_get_distance(
    nmslib_index_handle_t index,
    size_t pos1,
    size_t pos2,
    float* distance
) {
    if (!index || pos1 >= nmslib_data_qty(index) || pos2 >= nmslib_data_qty(index) || !distance) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid distance inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [pos1, pos2, distance](auto* idx) -> nmslib_error_t {
        try {
            *distance = idx->space->IndexTimeDistance(idx->data[pos1], idx->data[pos2]);
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Distance computed successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to compute distance");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_get_data_point_size(
    nmslib_index_handle_t index,
    size_t position,
    size_t* size
) {
    if (!index || position >= nmslib_data_qty(index) || !size) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid data point size inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [position, size](auto* idx) -> nmslib_error_t {
        try {
            *size = idx->data[position]->datalength();
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Data point size retrieved");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to get data point size");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_get_data_point_fill(
    nmslib_index_handle_t index,
    size_t position,
    void* data,
    size_t size
) {
    if (!index || !data || size == 0 || position >= nmslib_data_qty(index)) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid data point fill inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [position, data, size](auto* idx) -> nmslib_error_t {
        try {
            const void* src = idx->data[position]->data();
            size_t src_size = idx->data[position]->datalength();
            if (size < src_size) {
                SET_LAST_ERROR(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Buffer too small for data point");
                return NMSLIB_ERROR_BUFFER_TOO_SMALL;
            }
            std::memcpy(data, src, src_size);
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Data point filled");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to fill data point");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_get_data_point_string(
    nmslib_index_handle_t index,
    size_t position,
    const char** data,
    size_t* data_len,
    const nmslib_allocator_t* allocator
) {
    if (!index || !data || !data_len || !allocator || position >= nmslib_data_qty(index)) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid string data point inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [position, data, data_len, allocator](auto* idx) -> nmslib_error_t {
        try {
            if (idx->data_type != NMSLIB_DATATYPE_OBJECT_AS_STRING) {
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Invalid data type for string");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            const char* str = static_cast<const char*>(idx->data[position]->data());
            *data_len = idx->data[position]->datalength() + 1;
            *data = NMSLIBUtil::dup_string(std::string(str, *data_len - 1), allocator);
            if (!*data) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate string");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "String data point retrieved");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to get string data point");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_borrow_data_dense(
    nmslib_index_handle_t index,
    size_t position,
    void** data,
    size_t* size,
    void (**free_fn)(void*)
) {
    if (!index || !data || !size || !free_fn || position >= nmslib_data_qty(index)) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid dense borrow inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [position, data, size, free_fn](auto* idx) -> nmslib_error_t {
        try {
            if (idx->data_type != NMSLIB_DATATYPE_DENSE_VECTOR) {
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not dense vector");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            nmslib_borrowed_data_t* borrowed = static_cast<nmslib_borrowed_data_t*>(
                idx->allocator.alloc(sizeof(nmslib_borrowed_data_t), idx->allocator.ctx));
            if (!borrowed) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            *size = idx->data[position]->bufferlength();
            void* tempData = idx->allocator.alloc(*size, idx->allocator.ctx);
            if (!tempData) {
                idx->allocator.free(borrowed, idx->allocator.ctx);
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate data copy");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            std::memcpy(tempData, idx->data[position]->data(), *size);
            borrowed->data = tempData;
            borrowed->allocator = idx->allocator;
            *data = tempData;
            *free_fn = nmslib_borrowed_data_free;
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Dense data borrowed");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to borrow dense data");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_borrow_data_sparse(
    nmslib_index_handle_t index,
    size_t position,
    void** data,
    size_t* size,
    void (**free_fn)(void*)
) {
    if (!index || !data || !size || !free_fn || position >= nmslib_data_qty(index)) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid sparse borrow inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(index, [position, data, size, free_fn](auto* idx) -> nmslib_error_t {
        try {
            if (idx->data_type != NMSLIB_DATATYPE_SPARSE_VECTOR) {
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse vector");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
            }
            nmslib_borrowed_data_t* borrowed = static_cast<nmslib_borrowed_data_t*>(
                idx->allocator.alloc(sizeof(nmslib_borrowed_data_t), idx->allocator.ctx));
            if (!borrowed) {
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            *size = idx->data[position]->bufferlength() / sizeof(nmslib_sparse_elem_float_t);
            void* tempData = idx->allocator.alloc(*size * sizeof(nmslib_sparse_elem_float_t), idx->allocator.ctx);
            if (!tempData) {
                idx->allocator.free(borrowed, idx->allocator.ctx);
                SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate sparse copy");
                return NMSLIB_ERROR_OUT_OF_MEMORY;
            }
            std::memcpy(tempData, idx->data[position]->data(), *size * sizeof(nmslib_sparse_elem_float_t));
            borrowed->data = tempData;
            borrowed->allocator = idx->allocator;
            *data = tempData;
            *free_fn = nmslib_borrowed_data_free;
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Sparse data borrowed");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to borrow sparse data");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_save_index(
    nmslib_index_handle_t handle,
    const char* path,
    int save_data
) {
    if (!handle || !path) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid save inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [path, save_data](auto* idx) -> nmslib_error_t {
        try {
            if (!idx->index_ptr) {
                SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Index not built");
                return NMSLIB_ERROR_INVALID_ARGUMENT;
            }
            if (save_data) {
                std::vector<std::string> dummy;
                idx->space->WriteObjectVectorBinData(idx->data, dummy, std::string(path) + ".dat");
            }
            idx->index_ptr->SaveIndex(path);
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Index saved successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_DATA_IO_FAILED, "Failed to save index");
            return NMSLIB_ERROR_DATA_IO_FAILED;
        }
    });
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
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid load inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        void* mem;
        nmslib_index_handle_t result = nullptr;
        switch (dist_type) {
            case NMSLIB_DISTTYPE_FLOAT: {
                mem = allocator->alloc(sizeof(nmslib_internal_index_t<float>), allocator->ctx);
                if (!mem) {
                    SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index");
                    return NMSLIB_ERROR_OUT_OF_MEMORY;
                }
                auto idx = new (mem) nmslib_internal_index_t<float>("hnsw", "l2", data_type, dist_type, allocator);
                idx->space.reset(SpaceFactoryRegistry<float>::Instance().CreateSpace("l2", AnyParams()));
                auto factory = MethodFactoryRegistry<float>::Instance();
                bool print_progress = false;
                idx->index_ptr.reset(factory.CreateMethod(print_progress, "hnsw", "l2", *idx->space, idx->data));
                if (load_data) {
                    std::vector<std::string> dummy;
                    idx->space->ReadObjectVectorFromBinData(idx->data, dummy, std::string(path) + ".dat");
                }
                idx->index_ptr->LoadIndex(path);
                idx->index_ptr->ResetQueryTimeParams();
                result = reinterpret_cast<nmslib_index_handle_t>(idx);
                break;
            }
            case NMSLIB_DISTTYPE_INT: {
                mem = allocator->alloc(sizeof(nmslib_internal_index_t<int>), allocator->ctx);
                if (!mem) {
                    SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index");
                    return NMSLIB_ERROR_OUT_OF_MEMORY;
                }
                auto idx = new (mem) nmslib_internal_index_t<int>("hnsw", "l2", data_type, dist_type, allocator);
                idx->space.reset(SpaceFactoryRegistry<int>::Instance().CreateSpace("l2", AnyParams()));
                auto factory = MethodFactoryRegistry<int>::Instance();
                bool print_progress = false;
                idx->index_ptr.reset(factory.CreateMethod(print_progress, "hnsw", "l2", *idx->space, idx->data));
                if (load_data) {
                    std::vector<std::string> dummy;
                    idx->space->ReadObjectVectorFromBinData(idx->data, dummy, std::string(path) + ".dat");
                }
                idx->index_ptr->LoadIndex(path);
                idx->index_ptr->ResetQueryTimeParams();
                result = reinterpret_cast<nmslib_index_handle_t>(idx);
                break;
            }
            default:
                SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid dist type for load");
                return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Index loaded successfully");
        *out_handle = result;
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_DATA_IO_FAILED, "Failed to load index");
        return NMSLIB_ERROR_DATA_IO_FAILED;
    }
}

nmslib_error_t nmslib_set_query_time_params(
    nmslib_index_handle_t handle,
    nmslib_params_handle_t params
) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [params](auto* idx) -> nmslib_error_t {
        try {
            if (!idx->index_ptr) {
                SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built");
                return NMSLIB_ERROR_INDEX_BUILD_FAILED;
            }
            idx->index_ptr->SetQueryTimeParams(load_params(reinterpret_cast<nmslib_params_wrapper_t*>(params)));
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Query time params set");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to set query time params");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

nmslib_error_t nmslib_set_thread_pool_size(
    nmslib_index_handle_t handle,
    size_t size
) {
    if (!handle || size == 0 || size > 1024) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid thread pool size");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    return dispatch_index_by_data_type(handle, [size](auto* idx) -> nmslib_error_t {
        try {
            idx->thread_pool_size = size;
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Thread pool size set");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to set thread pool size");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}

size_t nmslib_get_thread_pool_size(nmslib_index_handle_t handle) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return std::thread::hardware_concurrency();
    }
    return dispatch_index_by_data_type(handle, [](auto* idx) -> size_t {
        return idx->thread_pool_size;
    });
}

size_t nmslib_data_qty(nmslib_index_handle_t handle) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return 0;
    }
    return dispatch_index_by_data_type(handle, [](auto* idx) -> size_t {
        return idx->data.size();
    });
}

size_t nmslib_index_memory_usage(nmslib_index_handle_t handle) {
    if (!handle) return 0;
    return dispatch_index_by_data_type(handle, [](auto* idx) -> size_t {
        try {
            if (!idx->index_ptr) return 0;
            size_t total = 0;
            for (auto obj : idx->data) {
                total += obj->bufferlength();
            }
            size_t dim = 0;
            if (!idx->data.empty()) {
                dim = idx->space->GetElemQty(idx->data[0]);
            }
            total += idx->data.size() * dim * sizeof(float);
            return total;
        } catch (...) {
            return 0;
        }
    });
}

nmslib_error_t nmslib_add_data_point_batch_pointers(
    nmslib_index_handle_t handle,
    nmslib_data_mode_t data_mode,
    const void *const *data_ptrs,
    size_t count,
    size_t element_count,
    const int32_t* ids,
    const size_t* num_elements
) {
    if (!handle || !data_ptrs || count == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid pointer batch inputs");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    if (nmslib_error_t err = NMSLIBUtil::validate_pointer_batch(data_ptrs, count, element_count)) {
        return err;
    }

    return dispatch_index_by_data_type(handle, [data_mode, data_ptrs, count, element_count, ids, num_elements](auto* idx) -> nmslib_error_t {
        try {
            for (size_t i = 0; i < count; ++i) {
                const int32_t curr_id = ids ? ids[i] : static_cast<int32_t>(i);
                std::unique_ptr<Object> obj;
                size_t curr_num = (data_mode == NMSLIB_DATA_MODE_SPARSE && num_elements) ? num_elements[i] : element_count;
                size_t effective_dim = element_count;
                if (data_mode == NMSLIB_DATA_MODE_SPARSE) {
                    effective_dim = 0;
                    if (curr_num == 0) {
                        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "No elements for sparse");
                        return NMSLIB_ERROR_INVALID_ARGUMENT;
                    }
                    if (idx->data_type != NMSLIB_DATATYPE_SPARSE_VECTOR) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    const nmslib_sparse_elem_float_t* elems = static_cast<const nmslib_sparse_elem_float_t*>(data_ptrs[i]);
                    if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems, curr_num, true)) {
                        return err;
                    }
                } else if (data_mode == NMSLIB_DATA_MODE_DENSE_FLOAT) {
                    if (idx->data_type != NMSLIB_DATATYPE_DENSE_VECTOR) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not dense float space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                } else if (data_mode == NMSLIB_DATA_MODE_UINT8) {
                    if (idx->data_type != NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not uint8 space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    const uint8_t* u8vec_ptr = static_cast<const uint8_t*>(data_ptrs[i]);
                    std::vector<uint8_t> u8data(u8vec_ptr, u8vec_ptr + element_count);
                    auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
                    if (!siftSpace) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not SIFT uint8 space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    Object* raw_obj = siftSpace->CreateObjFromUint8Vect(curr_id, static_cast<LabelType>(-1), u8data);
                    obj.reset(raw_obj);
                } else {
                    SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Unsupported data mode");
                    return NMSLIB_ERROR_INVALID_ARGUMENT;
                }
                if (!obj && data_mode != NMSLIB_DATA_MODE_UINT8) {
                    obj = create_object(idx->space.get(), idx->data_type, data_ptrs[i], effective_dim, curr_num, curr_id);
                }
                if (!obj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create object from data pointer");
                    return NMSLIB_ERROR_RUNTIME;
                }
                idx->data.push_back(obj.release());
            }
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Pointer batch added successfully");
            return NMSLIB_SUCCESS;
        } catch (...) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to add pointer batch");
            return NMSLIB_ERROR_RUNTIME;
        }
    });
}


void nmslib_free_result(nmslib_result_t* result, const nmslib_allocator_t* allocator) {
    if (!result || !allocator || !allocator->free) return;
    if (result->ids) allocator->free(result->ids, allocator->ctx);
    if (result->distances) allocator->free(result->distances, allocator->ctx);
    result->ids = nullptr;
    result->distances = nullptr;
    result->size = 0;
    result->capacity = 0;
}

void nmslib_initialize_pool(nmslib_index_handle_t handle) {
    if (!handle) return;

    try {
        dispatch_index_by_data_type(handle, [](auto* idx) -> void {
            using dist_t = decltype(idx->space->IndexTimeDistance(static_cast<const Object*>(nullptr), static_cast<const Object*>(nullptr)));
            if (idx && idx->index_ptr) {
                Index<dist_t>* base_ptr = idx->index_ptr.get();
                auto* hnsw = dynamic_cast<Hnsw<dist_t>*>(base_ptr);
                if (hnsw) {
                    AnyParams dummy;
                    hnsw->CreateIndex(dummy);
                }
            }
        });
    } catch (const std::exception& e) {
        fprintf(stderr, "NMSLIB pool initialization failed: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "NMSLIB pool initialization failed: unknown error\n");
    }
}

} // extern "C"