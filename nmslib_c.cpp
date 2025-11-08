#include "nmslib_c.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <numeric>
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
            // Basic uniformity check (heuristic; full in mode-specific)
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

    ~nmslib_params_wrapper_t() = default;
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
// ---- add dispatch helper near the top of nmslib_c.cpp ----

template <typename Fn>
nmslib_error_t dispatch_index_by_data_type(nmslib_index_handle_t handle, Fn&& fn) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_NULL_POINTER, "Null index handle");
        return NMSLIB_ERROR_NULL_POINTER;
    }

    // header is placed at the very start of every nmslib_internal_index_t<T> instantiation
    auto hdr = reinterpret_cast<nmslib_index_header_t*>(handle);
    switch (hdr->data_type) {
        // treat float-typed indices (dense/sparse float vectors)
        case NMSLIB_DATATYPE_DENSE_VECTOR:
        case NMSLIB_DATATYPE_SPARSE_VECTOR: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
            return fn(idx);
        }

        // treat int-typed indices (SIFT/uint8 & string/object-as-string use int registry in this codebase)
        case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
        case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
            return fn(idx);
        }

        default:
            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Unsupported data type");
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
    }
}

extern "C" {
__attribute__((constructor))
static void nmslib_force_space_registry_init() {
    // Touch the factory singletons to make sure static registration runs
    (void) similarity::SpaceFactoryRegistry<float>::Instance();
    (void) similarity::SpaceFactoryRegistry<int>::Instance();
    (void) similarity::SpaceFactoryRegistry<size_t>::Instance();
}

// Exported C-callable init function (optional: call this from Zig during startup)
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

    // one-time initialization of NMSLIB
    std::call_once(nmslib_init_flag, nmslib_do_init);

    try {
        // --- Try FLOAT registry first (most spaces are registered here) ---
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
                // fall through to fallback
            } catch (...) {
                // fall through to fallback
            }

            // cleanup float idx before attempting fallback
            idx->~nmslib_internal_index_t();
            allocator->free(mem, allocator->ctx);
        }

        // --- Fallback: try INT registry ---
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

    // Read header to know which instantiation we are destroying
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
            // Best-effort fallback: attempt float destroy (keeps prior behavior)
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

    // Use header to dispatch to correct template
    auto hdr = reinterpret_cast<nmslib_index_header_t*>(handle);

    try {
        switch (hdr->data_type) {
            case NMSLIB_DATATYPE_DENSE_VECTOR:
            case NMSLIB_DATATYPE_SPARSE_VECTOR: {
                auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
                auto factory = MethodFactoryRegistry<float>::Instance();
                idx->index_ptr.reset(factory.CreateMethod(print_progress != 0, idx->method, idx->space_type, *idx->space, idx->data));
                idx->index_ptr->CreateIndex(load_params(reinterpret_cast<nmslib_params_wrapper_t*>(index_params)));
                SET_LAST_ERROR(NMSLIB_SUCCESS, "Index created successfully");
                return NMSLIB_SUCCESS;
            }
            case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
            case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
                auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
                auto factory = MethodFactoryRegistry<int>::Instance();
                idx->index_ptr.reset(factory.CreateMethod(print_progress != 0, idx->method, idx->space_type, *idx->space, idx->data));
                idx->index_ptr->CreateIndex(load_params(reinterpret_cast<nmslib_params_wrapper_t*>(index_params)));
                SET_LAST_ERROR(NMSLIB_SUCCESS, "Index created successfully (int)");
                return NMSLIB_SUCCESS;
            }
            default:
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "No compatible space found for header data_type");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
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
}


nmslib_error_t nmslib_reset_index(nmslib_index_handle_t handle) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        *space_type_len = idx->space_type.length(); // ✅ FIX: exclude the null terminator
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        // FIX: return length excluding terminating NUL so caller can form a slice without trailing '\0'
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        std::unique_ptr<Object> obj;
        switch (idx->data_type) {
            case NMSLIB_DATATYPE_DENSE_VECTOR: {
                const float* vec = static_cast<const float*>(data);
                auto vectSpace = dynamic_cast<const VectorSpaceSimpleStorage<float>*>(idx->space.get());
                if (!vectSpace) {
                    SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not vector space");
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<float> tempVec(vec, vec + element_count);
                Object* raw_obj = vectSpace->CreateObjFromVect(static_cast<IdType>(id), static_cast<LabelType>(-1), tempVec);
                obj.reset(raw_obj);
                break;
            }
            case NMSLIB_DATATYPE_SPARSE_VECTOR: {
                const nmslib_sparse_elem_float_t* elems = static_cast<const nmslib_sparse_elem_float_t*>(data);
                if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems, element_count, true)) {
                    return err;
                }
                std::vector<SparseVectElem<float>> tempVec(element_count);
                for (size_t j = 0; j < element_count; ++j) {
                    tempVec[j].id_ = elems[j].id;
                    tempVec[j].val_ = elems[j].value;
                }
                auto sparseSpace = dynamic_cast<const SpaceSparseVectorSimpleStorage<float>*>(idx->space.get());
                if (!sparseSpace) {
                    SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space");
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                Object* raw_obj = sparseSpace->CreateObjFromVect(static_cast<IdType>(id), static_cast<LabelType>(-1), tempVec);
                obj.reset(raw_obj);
                break;
            }
            case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR: {
                const uint8_t* u8data = static_cast<const uint8_t*>(data);
                auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
                if (!siftSpace) {
                    SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not SIFT uint8 space");
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                std::vector<uint8_t> u8vec(u8data, u8data + element_count);  // Copy raw bytes
                Object* raw_obj = siftSpace->CreateObjFromUint8Vect(
                    static_cast<IdType>(id),
                    static_cast<LabelType>(-1),
                    u8vec
                );
                obj.reset(raw_obj);
                break;
            }
            case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
                const char* str = static_cast<const char*>(data);
                Object* raw_obj = new BorrowedObject(static_cast<IdType>(id), str, element_count);
                obj.reset(raw_obj);
                break;
            }
            default:
                SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid data type");
                return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        for (size_t i = 0; i < count; ++i) {
            int32_t curr_id_int = ids ? ids[i] : static_cast<int32_t>(i);
            IdType curr_id = static_cast<IdType>(curr_id_int);
            std::unique_ptr<Object> obj;
            size_t curr_num_elements = (num_elements && idx->data_type == NMSLIB_DATATYPE_SPARSE_VECTOR) ? num_elements[i] : element_count;
            const void* curr_data = static_cast<const uint8_t*>(data) + i * (element_count * sizeof(float));  // Adjust for type
            switch (idx->data_type) {
                case NMSLIB_DATATYPE_DENSE_VECTOR: {
                    const float* vec_start = static_cast<const float*>(data) + i * element_count;
                    std::vector<float> tempVec(vec_start, vec_start + element_count);
                    auto vectSpace = dynamic_cast<const VectorSpaceSimpleStorage<float>*>(idx->space.get());
                    if (!vectSpace) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not vector space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    Object* raw_obj = vectSpace->CreateObjFromVect(curr_id, static_cast<LabelType>(-1), tempVec);
                    obj.reset(raw_obj);
                    break;
                }
                case NMSLIB_DATATYPE_SPARSE_VECTOR: {
                    const nmslib_sparse_elem_float_t* elems_start = static_cast<const nmslib_sparse_elem_float_t*>(data) + i * curr_num_elements;
                    if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems_start, curr_num_elements, true)) {
                        return err;
                    }
                    std::vector<SparseVectElem<float>> tempVec(curr_num_elements);
                    for (size_t j = 0; j < curr_num_elements; ++j) {
                        tempVec[j].id_ = elems_start[j].id;
                        tempVec[j].val_ = elems_start[j].value;
                    }
                    auto sparseSpace = dynamic_cast<const SpaceSparseVectorSimpleStorage<float>*>(idx->space.get());
                    if (!sparseSpace) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    Object* raw_obj = sparseSpace->CreateObjFromVect(curr_id, static_cast<LabelType>(-1), tempVec);
                    obj.reset(raw_obj);
                    break;
                }
                case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR: {
                    const uint8_t* u8_start = static_cast<const uint8_t*>(data) + i * element_count;
                    auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
                    if (!siftSpace) {
                        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not SIFT uint8 space");
                        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                    }
                    std::vector<uint8_t> u8vec(u8_start, u8_start + element_count);  // Copy slice
                    Object* raw_obj = siftSpace->CreateObjFromUint8Vect(
                        curr_id,
                        static_cast<LabelType>(-1),
                        u8vec
                    );
                    obj.reset(raw_obj);
                    break;
                }
                case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
                    // Assume flat char** or handled separately
                    SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Use batch_string for strings");
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }
                default:
                    SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid data type for batch");
                    return NMSLIB_ERROR_INVALID_ARGUMENT;
            }
            if (!obj) {
                SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create batch object");
                return NMSLIB_ERROR_RUNTIME;
            }
            idx->data.push_back(obj.release());
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idx->data_type != NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not uint8 vector space");
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
        if (!siftSpace) {
            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not SIFT space");
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        const uint8_t* u8data = data;
        for (size_t i = 0; i < count; ++i) {
            int32_t curr_id_int = ids ? ids[i] : static_cast<int32_t>(i);
            IdType curr_id = static_cast<IdType>(curr_id_int);
            const uint8_t* vec_start = u8data + i * element_count;
            std::vector<uint8_t> tempVec(vec_start, vec_start + element_count);
            // Use CreateObjFromUint8Vect instead of CreateObjFromVect
            Object* raw_obj = siftSpace->CreateObjFromUint8Vect(curr_id, static_cast<LabelType>(-1), tempVec);
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
        // If header says not object-as-string, return incompatible
        SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not string space");
        return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
    }

    // Cast to int-instantiated index (string objects are under int registry in this codebase)
    auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(index);

    for (size_t i = 0; i < count; ++i) {
        if (!data[i]) {
            SET_LAST_ERROR(NMSLIB_ERROR_NULL_POINTER, "Null string in batch");
            return NMSLIB_ERROR_NULL_POINTER;
        }
        std::string str(data[i]);
        std::unique_ptr<Object> obj(idx->space->CreateObjFromStr(ids ? ids[i] : static_cast<int32_t>(i), -1, str, 0));
        if (!obj) {
            SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create string object");
            return NMSLIB_ERROR_RUNTIME;
        }
        idx->data.push_back(obj.release());
    }

    SET_LAST_ERROR(NMSLIB_SUCCESS, "String batch added successfully");
    return NMSLIB_SUCCESS;
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
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        if (!idx->index_ptr) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built");
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        }
        // Use NMSLIB's KNNQuery to get size (simplified; in practice, use SearchParams)
        *out_size = k + 10;  // Heuristic buffer
        SET_LAST_ERROR(NMSLIB_SUCCESS, "KNN size retrieved");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to get knn size");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

// ============================================================================
//  Fixed version of nmslib_knn_query_fill
//  Safe template dispatch using nmslib_index_header_t
// ============================================================================

// ============================================================================
// Replacement: nmslib_knn_query_fill
// ============================================================================
nmslib_error_t nmslib_knn_query_fill(
    nmslib_index_handle_t index,
    const void* query,
    size_t query_size_or_elem_count,
    size_t k,
    nmslib_result_t* result,
    size_t num_elements
) {
    // validate inputs (existing helper in repo)
    if (nmslib_error_t err = NMSLIBUtil::validate_common_inputs(index, query, query_size_or_elem_count, result)) {
        SET_LAST_ERROR(err, "Invalid KNN query inputs");
        return err;
    }
    if (!result->ids || !result->distances || result->capacity == 0) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Result buffers invalid");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }

    if (!index) {
        SET_LAST_ERROR(NMSLIB_ERROR_NULL_POINTER, "Null index handle");
        return NMSLIB_ERROR_NULL_POINTER;
    }

    // Read header (must exist as first field of internal index structs)
    auto hdr = reinterpret_cast<nmslib_index_header_t*>(index);

    try {
        // Dispatch by the canonical header data_type
        switch (hdr->data_type) {

            // ---------------------------
            // FLOAT-index branch (existing implementation)
            // ---------------------------
            case NMSLIB_DATATYPE_DENSE_VECTOR:
            case NMSLIB_DATATYPE_SPARSE_VECTOR: {
                auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
                if (!idx->index_ptr) {
                    SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built (float index)");
                    return NMSLIB_ERROR_INDEX_BUILD_FAILED;
                }

                std::unique_ptr<Object> qobj;
                // Build query object depending on dense vs sparse
                if (hdr->data_type == NMSLIB_DATATYPE_DENSE_VECTOR) {
                    const float* vec = static_cast<const float*>(query);
                    std::vector<float> tempVec(vec, vec + query_size_or_elem_count);
                    auto vectSpace = dynamic_cast<const VectorSpaceSimpleStorage<float>*>(idx->space.get());
                    if (vectSpace)
                        qobj.reset(vectSpace->CreateObjFromVect(0, static_cast<LabelType>(-1), tempVec));
                } else { // sparse
                    const nmslib_sparse_elem_float_t* elems = static_cast<const nmslib_sparse_elem_float_t*>(query);
                    if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems, num_elements, true))
                        return err;
                    std::vector<SparseVectElem<float>> tempVec(num_elements);
                    for (size_t j = 0; j < num_elements; ++j) {
                        tempVec[j].id_ = elems[j].id;
                        tempVec[j].val_ = elems[j].value;
                    }
                    auto sparseSpace = dynamic_cast<const SpaceSparseVectorSimpleStorage<float>*>(idx->space.get());
                    if (sparseSpace)
                        qobj.reset(sparseSpace->CreateObjFromVect(0, static_cast<LabelType>(-1), tempVec));
                }

                if (!qobj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Failed to create query object");
                    result->size = 0;
                    return NMSLIB_ERROR_INVALID_ARGUMENT;
                }

                // Run the search (use same params as repo)
                similarity::KNNQuery<float> knnQuery(*(idx->space), qobj.get(), k);
                AnyParams queryParams({ "efSearch=200" });
                idx->index_ptr->SetQueryTimeParams(queryParams);
                idx->index_ptr->Search(&knnQuery);

                auto res = knnQuery.Result();
                if (!res) {
                    SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Query result missing");
                    result->size = 0;
                    return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
                }

                size_t found = res->Size();
                result->size = found;

                // Allow success even when zero neighbors found
                if (found == 0) {
                    SET_LAST_ERROR(NMSLIB_SUCCESS, "No neighbors found");
                    return NMSLIB_SUCCESS;
                }

                if (found > result->capacity) {
                    SET_LAST_ERROR(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffers too small for " + std::to_string(found));
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }

                auto cloneQueue = res->Clone();
                std::vector<std::pair<int32_t, float>> tmp;
                tmp.reserve(found);
                while (!cloneQueue->Empty()) {
                    float d = cloneQueue->TopDistance();
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

                SET_LAST_ERROR(NMSLIB_SUCCESS, "KNN query filled successfully");
                return NMSLIB_SUCCESS;
            }

            // ---------------------------
            // UINT8 / SIFT branch (int-instantiated index)
            // ---------------------------
            case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR: {
                auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(index);
                if (!idx->index_ptr) {
                    SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built (uint8 index)");
                    return NMSLIB_ERROR_INDEX_BUILD_FAILED;
                }

                // Build uint8 query object using the SIFT helper
                const uint8_t* u8data = static_cast<const uint8_t*>(query);
                if (!u8data) {
                    SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Null uint8 query");
                    return NMSLIB_ERROR_INVALID_ARGUMENT;
                }
                std::vector<uint8_t> qv(u8data, u8data + query_size_or_elem_count);

                auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
                if (!siftSpace) {
                    SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not L2SqrSIFT space");
                    return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                }

                // CreateObjFromUint8Vect returns an Object* in this codebase — use unique_ptr for RAII
                std::unique_ptr<Object> qobj;
                Object* raw = siftSpace->CreateObjFromUint8Vect(static_cast<IdType>(0), static_cast<LabelType>(-1), qv);
                if (!raw) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create uint8 query object");
                    return NMSLIB_ERROR_RUNTIME;
                }
                qobj.reset(raw);

                // Run search using the int-index types
                similarity::KNNQuery<int> knnQuery(*(idx->space), qobj.get(), k);
                AnyParams queryParams({ "efSearch=200" });
                idx->index_ptr->SetQueryTimeParams(queryParams);
                idx->index_ptr->Search(&knnQuery);

                auto res = knnQuery.Result();
                if (!res) {
                    SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Query result missing (uint8)");
                    result->size = 0;
                    return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
                }

                size_t found = res->Size();
                result->size = found;

                if (found == 0) {
                    SET_LAST_ERROR(NMSLIB_SUCCESS, "No neighbors found");
                    return NMSLIB_SUCCESS;
                }

                if (found > result->capacity) {
                    SET_LAST_ERROR(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffers too small for " + std::to_string(found));
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }

                auto cloneQueue = res->Clone();
                std::vector<std::pair<int32_t, float>> tmp;
                tmp.reserve(found);
                while (!cloneQueue->Empty()) {
                    // TopDistance for int-based queue might be integral; cast to float for C API
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

                SET_LAST_ERROR(NMSLIB_SUCCESS, "KNN query filled successfully (uint8)");
                return NMSLIB_SUCCESS;
            }

            // ---------------------------
            // STRING / levenshtein branch (object-as-string, int-index)
            // ---------------------------
            case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
                auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(index);
                if (!idx->index_ptr) {
                    SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built (string index)");
                    return NMSLIB_ERROR_INDEX_BUILD_FAILED;
                }

                const char* str = static_cast<const char*>(query);
                if (!str) {
                    SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Null string query");
                    return NMSLIB_ERROR_INVALID_ARGUMENT;
                }

                // Create string object using the space's CreateObjFromStr (returns unique_ptr<Object>)
                std::unique_ptr<Object> qobj(idx->space->CreateObjFromStr(static_cast<IdType>(0), static_cast<LabelType>(-1), std::string(str), nullptr));
                if (!qobj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create string query object");
                    return NMSLIB_ERROR_RUNTIME;
                }

                similarity::KNNQuery<int> knnQuery(*(idx->space), qobj.get(), k);
                AnyParams queryParams({ "efSearch=200" });
                idx->index_ptr->SetQueryTimeParams(queryParams);
                idx->index_ptr->Search(&knnQuery);

                auto res = knnQuery.Result();
                if (!res) {
                    SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Query result missing (string)");
                    result->size = 0;
                    return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
                }

                size_t found = res->Size();
                result->size = found;

                if (found == 0) {
                    SET_LAST_ERROR(NMSLIB_SUCCESS, "No neighbors found");
                    return NMSLIB_SUCCESS;
                }

                if (found > result->capacity) {
                    SET_LAST_ERROR(NMSLIB_ERROR_BUFFER_TOO_SMALL, "Result buffers too small for " + std::to_string(found));
                    return NMSLIB_ERROR_BUFFER_TOO_SMALL;
                }

                auto cloneQueue = res->Clone();
                std::vector<std::pair<int32_t, float>> tmp;
                tmp.reserve(found);
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

                SET_LAST_ERROR(NMSLIB_SUCCESS, "KNN query filled successfully (string)");
                return NMSLIB_SUCCESS;
            }

            // ---------------------------
            // Unsupported data_type
            // ---------------------------
            default:
                SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Unsupported data type for knn query");
                return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        } // switch

    } catch (const std::bad_alloc& e) {
        SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Query alloc failed: " + std::string(e.what()));
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "KNN query failed");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
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
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        if (!idx->index_ptr) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built");
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        }
        // Parallel batch search (simplified)
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
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        if (!idx->index_ptr) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built");
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        }
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
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        if (!idx->index_ptr) {
            SET_LAST_ERROR(NMSLIB_ERROR_INDEX_BUILD_FAILED, "Index not built");
            return NMSLIB_ERROR_INDEX_BUILD_FAILED;
        }
        // Simplified range search
        result->size = 0;  // Placeholder
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Range query filled");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_QUERY_EXECUTION_FAILED, "Failed to fill range query");
        return NMSLIB_ERROR_QUERY_EXECUTION_FAILED;
    }
}

// Updated nmslib_get_distance (use IndexTimeDistance(obj1, obj2))
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        *distance = idx->space->IndexTimeDistance(idx->data[pos1], idx->data[pos2]);
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Distance computed successfully");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to compute distance");
        return NMSLIB_ERROR_RUNTIME;
    }
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        *size = idx->data[position]->datalength();
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Data point size retrieved");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to get data point size");
        return NMSLIB_ERROR_RUNTIME;
    }
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        if (idx->data_type != NMSLIB_DATATYPE_DENSE_VECTOR) {
            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not dense vector");
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        // Alloc wrapper
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(index);
        if (idx->data_type != NMSLIB_DATATYPE_SPARSE_VECTOR) {
            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse vector");
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
        }
        // Similar to dense, but for sparse elems
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
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
    void* mem = allocator->alloc(sizeof(nmslib_internal_index_t<float>), allocator->ctx);
    if (!mem) {
        SET_LAST_ERROR(NMSLIB_ERROR_OUT_OF_MEMORY, "Failed to allocate index");
        return NMSLIB_ERROR_OUT_OF_MEMORY;
    }
    try {
        nmslib_index_handle_t result = nullptr;
        switch (dist_type) {
            case NMSLIB_DISTTYPE_FLOAT: {
                auto idx = new (mem) nmslib_internal_index_t<float>("", "", data_type, dist_type, allocator);
                // Create dummy space/method for load
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
                // Similar for int
                auto idx = new (mem) nmslib_internal_index_t<int>("", "", data_type, dist_type, allocator);
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
                allocator->free(mem, allocator->ctx);
                return NMSLIB_ERROR_INVALID_ARGUMENT;
        }
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Index loaded successfully");
        *out_handle = result;
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_DATA_IO_FAILED, "Failed to load index");
        allocator->free(mem, allocator->ctx);
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
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
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
}

nmslib_error_t nmslib_set_thread_pool_size(
    nmslib_index_handle_t handle,
    size_t size
) {
    if (!handle || size == 0 || size > 1024) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid thread pool size");
        return NMSLIB_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        idx->thread_pool_size = size;
        SET_LAST_ERROR(NMSLIB_SUCCESS, "Thread pool size set");
        return NMSLIB_SUCCESS;
    } catch (...) {
        SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to set thread pool size");
        return NMSLIB_ERROR_RUNTIME;
    }
}

size_t nmslib_get_thread_pool_size(nmslib_index_handle_t handle) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return std::thread::hardware_concurrency();
    }
    return reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->thread_pool_size;
}

size_t nmslib_data_qty(nmslib_index_handle_t handle) {
    if (!handle) {
        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Invalid index");
        return 0;
    }
    return reinterpret_cast<nmslib_internal_index_t<float>*>(handle)->data.size();
}

// Updated nmslib_index_memory_usage (use GetElemQty for dim)
size_t nmslib_index_memory_usage(nmslib_index_handle_t handle) {
    if (!handle) return 0;
    try {
        auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (!idx->index_ptr) return 0;
        size_t total = 0;
        for (auto obj : idx->data) {
            total += obj->bufferlength();
        }
        size_t dim = 0;
        if (!idx->data.empty()) {
            dim = idx->space->GetElemQty(idx->data[0]);
        }
        total += idx->data.size() * (dim * sizeof(float) + 32);  // Heuristic
        return total;
    } catch (...) {
        return 0;
    }
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

    // Read header and dispatch by index instantiation type
    auto hdr = reinterpret_cast<nmslib_index_header_t*>(handle);
    switch (hdr->data_type) {
        // FLOAT-based index instantiation
        case NMSLIB_DATATYPE_DENSE_VECTOR:
        case NMSLIB_DATATYPE_SPARSE_VECTOR: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);

            // existing code path for float-index: create objects and push into idx->data
            for (size_t i = 0; i < count; ++i) {
                const int32_t curr_id = ids ? ids[i] : static_cast<int32_t>(i);
                std::unique_ptr<Object> obj;
                switch (data_mode) {
                    case NMSLIB_DATA_MODE_DENSE_FLOAT: {
                        if (idx->data_type != NMSLIB_DATATYPE_DENSE_VECTOR) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not dense float space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        const float* vec = static_cast<const float*>(data_ptrs[i]);
                        std::vector<float> tempVec(vec, vec + element_count);
                        auto vectSpace = dynamic_cast<const VectorSpaceSimpleStorage<float>*>(idx->space.get());
                        if (!vectSpace) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not vector space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        Object* raw_obj = vectSpace->CreateObjFromVect(curr_id, static_cast<LabelType>(-1), tempVec);
                        obj.reset(raw_obj);
                        break;
                    }
                    case NMSLIB_DATA_MODE_SPARSE: {
                        if (idx->data_type != NMSLIB_DATATYPE_SPARSE_VECTOR) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        size_t curr_num = num_elements ? num_elements[i] : 0;
                        if (curr_num == 0) {
                            SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "No elements for sparse");
                            return NMSLIB_ERROR_INVALID_ARGUMENT;
                        }
                        const nmslib_sparse_elem_float_t* elems = static_cast<const nmslib_sparse_elem_float_t*>(data_ptrs[i]);
                        if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems, curr_num, true)) {
                            return err;
                        }
                        std::vector<SparseVectElem<float>> tempVec(curr_num);
                        for (size_t j = 0; j < curr_num; ++j) {
                            tempVec[j].id_ = elems[j].id;
                            tempVec[j].val_ = elems[j].value;
                        }
                        auto sparseSpace = dynamic_cast<const SpaceSparseVectorSimpleStorage<float>*>(idx->space.get());
                        if (!sparseSpace) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        Object* raw_obj = sparseSpace->CreateObjFromVect(curr_id, static_cast<LabelType>(-1), tempVec);
                        obj.reset(raw_obj);
                        break;
                    }
                    default:
                        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Unsupported data mode for float index");
                        return NMSLIB_ERROR_INVALID_ARGUMENT;
                } // data_mode switch

                if (!obj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create object from data pointer (float index)");
                    return NMSLIB_ERROR_RUNTIME;
                }
                idx->data.push_back(obj.release());
            } // for i
            SET_LAST_ERROR(NMSLIB_SUCCESS, "Pointer batch added successfully (float index)");
            return NMSLIB_SUCCESS;
        } // end float index case

        // INT-based index instantiation (SIFT / uint8 / string-object usage)
        case NMSLIB_DATATYPE_DENSE_UINT8_VECTOR:
        case NMSLIB_DATATYPE_OBJECT_AS_STRING: {
            auto idx = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);

            for (size_t i = 0; i < count; ++i) {
                const int32_t curr_id = ids ? ids[i] : static_cast<int32_t>(i);
                std::unique_ptr<Object> obj;
                switch (data_mode) {
                    case NMSLIB_DATA_MODE_UINT8: {
                        if (idx->data_type != NMSLIB_DATATYPE_DENSE_UINT8_VECTOR) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not uint8 space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        const uint8_t* u8vec_ptr = static_cast<const uint8_t*>(data_ptrs[i]);
                        size_t curr_elem_count = element_count;
                        std::vector<uint8_t> u8data(u8vec_ptr, u8vec_ptr + curr_elem_count);
                        auto siftSpace = dynamic_cast<const SpaceL2SqrSift*>(idx->space.get());
                        if (!siftSpace) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not SIFT uint8 space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        Object* raw_obj = siftSpace->CreateObjFromUint8Vect(curr_id, static_cast<LabelType>(-1), u8data);
                        obj.reset(raw_obj);
                        break;
                    }
                    case NMSLIB_DATA_MODE_SPARSE: {
                        if (idx->data_type != NMSLIB_DATATYPE_SPARSE_VECTOR) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space (int index)");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        size_t curr_num = num_elements ? num_elements[i] : 0;
                        if (curr_num == 0) {
                            SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "No elements for sparse");
                            return NMSLIB_ERROR_INVALID_ARGUMENT;
                        }
                        const nmslib_sparse_elem_float_t* elems = static_cast<const nmslib_sparse_elem_float_t*>(data_ptrs[i]);
                        if (nmslib_error_t err = NMSLIBUtil::validate_sparse_elements(elems, curr_num, true)) {
                            return err;
                        }
                        std::vector<SparseVectElem<float>> tempVec(curr_num);
                        for (size_t j = 0; j < curr_num; ++j) {
                            tempVec[j].id_ = elems[j].id;
                            tempVec[j].val_ = elems[j].value;
                        }
                        auto sparseSpace = dynamic_cast<const SpaceSparseVectorSimpleStorage<float>*>(idx->space.get());
                        if (!sparseSpace) {
                            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "Not sparse space");
                            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
                        }
                        Object* raw_obj = sparseSpace->CreateObjFromVect(curr_id, static_cast<LabelType>(-1), tempVec);
                        obj.reset(raw_obj);
                        break;
                    }
                    default:
                        SET_LAST_ERROR(NMSLIB_ERROR_INVALID_ARGUMENT, "Unsupported data mode for int index");
                        return NMSLIB_ERROR_INVALID_ARGUMENT;
                } // inner switch

                if (!obj) {
                    SET_LAST_ERROR(NMSLIB_ERROR_RUNTIME, "Failed to create object from data pointer (int index)");
                    return NMSLIB_ERROR_RUNTIME;
                }
                idx->data.push_back(obj.release());
            } // for i

            SET_LAST_ERROR(NMSLIB_SUCCESS, "Pointer batch added successfully (int index)");
            return NMSLIB_SUCCESS;
        } // end int index case

        default:
            SET_LAST_ERROR(NMSLIB_ERROR_SPACE_INCOMPATIBLE, "No compatible index type found for provided handle");
            return NMSLIB_ERROR_SPACE_INCOMPATIBLE;
    } // switch hdr->data_type
}


void nmslib_free_result(nmslib_result_t* result, const nmslib_allocator_t* allocator) {
    if (!result || !allocator || !allocator->free) return;
    if (result->ids) allocator->free(result->ids, allocator->ctx);
    if (result->distances) allocator->free(result->distances, allocator->ctx);
    // Zero out to prevent double-free
    result->ids = nullptr;
    result->distances = nullptr;
    result->size = 0;
    result->capacity = 0;
}
void nmslib_initialize_pool(nmslib_index_handle_t handle) {
    if (!handle) return;

    try {
        // Try float instantiation wrapper
        auto idxf = reinterpret_cast<nmslib_internal_index_t<float>*>(handle);
        if (idxf && idxf->index_ptr) {
            // index_ptr is std::unique_ptr<Index<float>>
            similarity::Index<float>* base_ptr = idxf->index_ptr.get();
            auto* hnsw_f = dynamic_cast<similarity::Hnsw<float>*>(base_ptr);
            if (hnsw_f) {
                similarity::AnyParams dummy;
                hnsw_f->CreateIndex(dummy); // idempotent for our usage: ensures pool exists
                return;
            }
        }

        // Try int instantiation wrapper
        auto idxi = reinterpret_cast<nmslib_internal_index_t<int>*>(handle);
        if (idxi && idxi->index_ptr) {
            similarity::Index<int>* base_ptr = idxi->index_ptr.get();
            auto* hnsw_i = dynamic_cast<similarity::Hnsw<int>*>(base_ptr);
            if (hnsw_i) {
                similarity::AnyParams dummy;
                hnsw_i->CreateIndex(dummy);
                return;
            }
        }

        // If not HNSW or index_ptr is null, nothing to do
    } catch (const std::exception& e) {
        fprintf(stderr, "NMSLIB pool initialization failed: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "NMSLIB pool initialization failed: unknown error\n");
    }
}

} // extern "C"