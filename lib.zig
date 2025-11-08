const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

// Import the C interface
const c = @cImport({
    @cInclude("nmslib_c.h");
});
extern fn nmslib_free_result(result: *c.nmslib_result_t, allocator: *const c.nmslib_allocator_t) void;

// Error set mapping to C wrapper errors
pub const Error = Allocator.Error || error{
    NullPointer,
    InvalidArgument,
    OutOfMemory,
    BufferTooSmall,
    SpaceIncompatible,
    QueryTooLarge,
    InvalidSparseElement,
    IndexBuildFailed,
    QueryExecutionFailed,
    DataIOFailed,
    PluginRegistrationFailed,
    Internal,
    Runtime,
    IndexNotBuilt,
    IndexAlreadyBuilt,
};

fn mapError(err: c.nmslib_error_t) Error!void {
    if (err != c.NMSLIB_SUCCESS)
        std.debug.print("NMSLIB call failed with code {}\n", .{err});
    return switch (err) {
        c.NMSLIB_SUCCESS => {},
        c.NMSLIB_ERROR_NULL_POINTER => error.NullPointer,
        c.NMSLIB_ERROR_INVALID_ARGUMENT => error.InvalidArgument,
        c.NMSLIB_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        c.NMSLIB_ERROR_BUFFER_TOO_SMALL => error.BufferTooSmall,
        c.NMSLIB_ERROR_SPACE_INCOMPATIBLE => error.SpaceIncompatible,
        c.NMSLIB_ERROR_QUERY_TOO_LARGE => error.QueryTooLarge,
        c.NMSLIB_ERROR_INVALID_SPARSE_ELEMENT => error.InvalidSparseElement,
        c.NMSLIB_ERROR_INDEX_BUILD_FAILED => error.IndexBuildFailed,
        c.NMSLIB_ERROR_QUERY_EXECUTION_FAILED => error.QueryExecutionFailed,
        c.NMSLIB_ERROR_DATA_IO_FAILED => error.DataIOFailed,
        c.NMSLIB_ERROR_PLUGIN_REGISTRATION_FAILED => error.PluginRegistrationFailed,
        c.NMSLIB_ERROR_INTERNAL => error.Internal,
        c.NMSLIB_ERROR_RUNTIME => error.Runtime,
        c.NMSLIB_ERROR_INDEX_NOT_BUILT => error.IndexNotBuilt,
        else => {
            std.debug.print("Unknown error code: {}\n", .{err});
            return error.Runtime;
        },
    };
}

pub fn nmslibInitFromZig() void {
    // call the C-exported init function (this is safe to call multiple times)
    c.nmslib_init();
}

// Data type enum
pub const DataType = enum {
    DenseVector,
    SparseVector,
    DenseUInt8Vector,
    ObjectAsString,

    fn toC(self: DataType) c.nmslib_data_type_t {
        return switch (self) {
            .DenseVector => c.NMSLIB_DATATYPE_DENSE_VECTOR,
            .SparseVector => c.NMSLIB_DATATYPE_SPARSE_VECTOR,
            .DenseUInt8Vector => c.NMSLIB_DATATYPE_DENSE_UINT8_VECTOR,
            .ObjectAsString => c.NMSLIB_DATATYPE_OBJECT_AS_STRING,
        };
    }

    fn toDataMode(self: DataType) c.nmslib_data_mode_t {
        return switch (self) {
            .DenseVector => c.NMSLIB_DATA_MODE_DENSE_FLOAT,
            .SparseVector => c.NMSLIB_DATA_MODE_SPARSE,
            .DenseUInt8Vector => c.NMSLIB_DATA_MODE_UINT8,
            .ObjectAsString => undefined, // Use legacy batch_string
        };
    }
};

// Distance type enum
pub const DistType = enum {
    Float,
    Int,

    fn toC(self: DistType) c.nmslib_dist_type_t {
        return switch (self) {
            .Float => c.NMSLIB_DISTTYPE_FLOAT,
            .Int => c.NMSLIB_DISTTYPE_INT,
        };
    }
};

// Parameter value union
pub const ParamValue = union(enum) {
    String: []const u8,
    Int: i32,
    Double: f64,
};

// Unified descriptor (with getId method)
pub const Descriptor = union(DataType) {
    DenseVector: struct {
        id: i32,
        data_ptr: [*]const f32,
        length: usize,
        data: []const f32, // slice mirror
    },
    SparseVector: struct {
        id: i32,
        data_ptr: [*]const SparseElem,
        num_elements: usize,
        data: []const SparseElem,
    },
    DenseUInt8Vector: struct {
        id: i32,
        data_ptr: [*]const u8,
        length: usize,
        data: []const u8,
    },
    ObjectAsString: struct {
        id: i32,
        data_ptr: [*]const u8,
        length: usize,
        data: []const u8,
    },

    pub fn getId(self: Descriptor) i32 {
        return switch (self) {
            .DenseVector => |d| d.id,
            .SparseVector => |s| s.id,
            .DenseUInt8Vector => |u| u.id,
            .ObjectAsString => |o| o.id,
        };
    }
};




// Sparse elem
pub const SparseElem = extern struct { id: u32, value: f32 };
const array_list = std.array_list;

// Data storage (unified)
pub const DataStorage = struct {
    arena: std.heap.ArenaAllocator,
    descriptors: array_list.Managed(Descriptor),

    pub fn deinit(self: *DataStorage, _: Allocator) void {
        self.arena.deinit();
        self.descriptors.deinit();
    }

pub fn init(allocator: Allocator, data_type: DataType) !DataStorage {
    _ = data_type; // Unused for unified

    const arena = std.heap.ArenaAllocator.init(allocator);
    const descriptors = array_list.Managed(Descriptor).init(allocator);

    return .{
        .arena = arena,
        .descriptors = descriptors,
    };
}







};

// Alloc context (unchanged)
const AllocContext = struct {
    map: std.HashMap(*anyopaque, usize, Context, std.hash_map.default_max_load_percentage),
    child_allocator: Allocator,
    const Context = struct {
        pub fn hash(_: @This(), key: *anyopaque) u64 {
            return @intFromPtr(key);
        }
        pub fn eql(_: @This(), a: *anyopaque, b: *anyopaque) bool {
            return a == b;
        }
    };

    fn init(parent: Allocator) !@This() {
        const map = std.HashMap(*anyopaque, usize, Context, std.hash_map.default_max_load_percentage).init(parent);

        return .{
            .map = map,
            .child_allocator = parent,
        };




    }

    fn deinit(self: *AllocContext) void {
        var iter = self.map.iterator();
        while (iter.next()) |entry| {
            const len = entry.value_ptr.*;
            const slice_ptr = @as([*]u8, @ptrCast(entry.key_ptr.*));
            const slice = slice_ptr[0..len];
            self.child_allocator.free(slice);
        }
        self.map.deinit();
    }
};

fn c_alloc_fn(size: usize, ctx: ?*anyopaque) callconv(.c) ?*anyopaque {
    if (ctx == null) return null;

    // Safely recover the original AllocContext pointer
    const tracker = @as(*AllocContext, @ptrCast(@alignCast(ctx.?)));

    // Allocate memory via the tracked allocator
    const mem = tracker.child_allocator.alloc(u8, size) catch return null;
    const ptr = @as(*anyopaque, @ptrCast(mem.ptr));

    // Track allocation for correct freeing later
    tracker.map.put(ptr, mem.len) catch {
        tracker.child_allocator.free(mem);
        return null;
    };

    return ptr;
}

fn c_free_fn(ptr: ?*anyopaque, ctx: ?*anyopaque) callconv(.c) void {
    if (ptr == null or ctx == null) return;

    // Recover the same AllocContext pointer
    const tracker = @as(*AllocContext, @ptrCast(@alignCast(ctx.?)));

    // Free memory if previously tracked
    if (tracker.map.get(ptr.?)) |len| {
        const slice_ptr = @as([*]u8, @ptrCast(ptr.?));
        const slice = slice_ptr[0..len];
        tracker.child_allocator.free(slice);
        _ = tracker.map.remove(ptr.?);
    }
}





// Add `keys` field to Params struct (replace the existing Params definition's field list accordingly)
pub const Params = struct {
    c_params: c.nmslib_params_handle_t,
    allocator: Allocator,
    alloc_context: *AllocContext,

    // track duplicated keys so we can answer has() queries from Zig side
    keys: array_list.AlignedManaged([]u8, null),


    pub fn init(allocator: Allocator) !Params {
        const tracker = try allocator.create(AllocContext);
        errdefer allocator.destroy(tracker);
        tracker.* = try AllocContext.init(allocator);

        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(tracker),
        };
        const c_params = c.nmslib_create_params(&c_alloc) orelse {
            tracker.deinit();
            allocator.destroy(tracker);
            return error.OutOfMemory;
        };

        // Use internal ArrayList managed variant compatible with your stdlib.
        const keys = array_list.AlignedManaged([]u8, null).init(allocator);

        return .{
            .c_params = c_params,
            .allocator = allocator,
            .alloc_context = tracker,
            .keys = keys,
        };
    }


    pub fn deinit(self: *Params) void {
        c.nmslib_free_params(self.c_params);

        for (self.keys.items) |k|
            self.allocator.free(k);


        self.keys.deinit();
        self.alloc_context.map.deinit();
        self.allocator.destroy(self.alloc_context);
    }




    pub fn add(self: *Params, key: []const u8, value: ParamValue) !void {
        const key_z = try self.allocator.dupeZ(u8, key);
        const key_full = key_z[0 .. key_z.len + 1]; // include trailing NUL

        switch (value) {
            .String => |s| {
                const value_z = try self.allocator.dupeZ(u8, s);
                defer self.allocator.free(value_z[0 .. value_z.len + 1]);
                try mapError(c.nmslib_add_param(
                    self.c_params,
                    key_full.ptr,        // ✅ pass pointer, not slice
                    2,
                    @ptrCast(value_z.ptr) // ✅ pointer to NUL-terminated string
                ));
            },
            .Int => |i| try mapError(c.nmslib_add_param(
                self.c_params,
                key_full.ptr,
                0,
                &i,
            )),
            .Double => |d| try mapError(c.nmslib_add_param(
                self.c_params,
                key_full.ptr,
                1,
                &d,
            )),
        }

        try self.keys.append(key_full);
    }



    pub fn has(self: *const Params, key: []const u8) bool {
        for (self.keys.items) |k| {
            if (k.len == key.len and std.mem.eql(u8, k, key)) return true;
        }
        return false;
    }


    pub fn fromSlice(allocator: Allocator, pairs: []const struct { key: []const u8, value: ParamValue }) !Params {
        var params = try Params.init(allocator);
        errdefer params.deinit();
        for (pairs) |pair| try params.add(pair.key, pair.value);
        return params;
    }
};

// --- New helper: validateCreateInputs ---
fn validateCreateInputs(
    space_type: []const u8,
    data_type: DataType,
    dist_type: DistType,
    index_params: ?*Params,
) !void {
    // simple span for string comparisons
    const s = space_type;

    // Levenshtein ("leven") requires integer distance type
    if (std.mem.eql(u8, s, "leven") and dist_type != DistType.Int) {
        return error.InvalidArgument;
    }

    // Vector spaces (L2/cosine variants) require a dim parameter
    if (std.mem.startsWith(u8, s, "l2") or std.mem.startsWith(u8, s, "l2sqr") or std.mem.startsWith(u8, s, "cosine")) {
        if (index_params == null) return error.InvalidArgument;
        if (!index_params.?.has("dim")) return error.InvalidArgument;
    }

    // Dense uint8 vectors expect integer distance
    if (data_type == .DenseUInt8Vector and dist_type != DistType.Int) {
        return error.InvalidArgument;
    }

    // Otherwise accept
    return;
}

pub const QueryResult = struct {
    ids: []i32,
    distances: []f32,
    full_ids: []i32,
    full_distances: []f32,
    used: usize,
    allocator: Allocator,

    pub fn deinit(self: @This()) void {
        self.allocator.free(self.full_ids);
        self.allocator.free(self.full_distances);
    }
};



pub const BatchResult = struct {
    results: []QueryResult,
    allocator: Allocator,

    pub fn deinit(self: @This()) void {
        for (self.results) |result| result.deinit();
        self.allocator.free(self.results);
    }
};

// Points (QueryPoint/DataPoint unchanged, but DataPoint uses union without id)
pub const QueryPoint = union(DataType) {
    DenseVector: []const f32,
    SparseVector: []const SparseElem,
    DenseUInt8Vector: []const u8,
    ObjectAsString: []const u8,
};

pub const DataPoint = union(DataType) {
    DenseVector: []const f32,
    SparseVector: []const SparseElem,
    DenseUInt8Vector: []const u8,
    ObjectAsString: []const u8,
};

// Index
pub const Index = struct {
    const Self = @This();
    pub const track_cpp_memory: bool = true;

    handle: c.nmslib_index_handle_t,
    allocator: Allocator,
    alloc_context: *AllocContext,
    data_type: DataType,
    dist_type: DistType,
    data_storage: DataStorage,
    built: bool,
    cpp_index_phantom: ?[]u8,

    pub fn init(
        allocator: Allocator,
        space_type: []const u8,
        index_params: ?Params,
        method: []const u8,
        data_type: DataType,
        dist_type: DistType,
    ) !Self {
        c.nmslib_init();

        // ✅ FIXED: tracker is now owned by the returned Index
        var tracker = try allocator.create(AllocContext);
        errdefer allocator.destroy(tracker);   // only destroy on error
        tracker.* = try AllocContext.init(allocator);
        errdefer tracker.deinit();             // deinit on error

        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(tracker),
        };

        // Duplicate strings for C API
        const space_z = try allocator.dupeZ(u8, space_type);
        defer allocator.free(space_z);
        const method_z = try allocator.dupeZ(u8, method);
        defer allocator.free(method_z);

        const c_params: c.nmslib_params_handle_t = if (index_params) |p| p.c_params else null;
        var handle: c.nmslib_index_handle_t = undefined;

        const result = c.nmslib_index_create(
            space_z,
            c_params orelse null,
            method_z,
            data_type.toC(),
            dist_type.toC(),
            &c_alloc,
            &handle,
        );

        if (result != c.NMSLIB_SUCCESS) {
            var detail: c.nmslib_error_detail_t = undefined;
            const temp_alloc = c.nmslib_allocator_t{
                .alloc = c_alloc_fn,
                .free = c_free_fn,
                .ctx = @ptrCast(tracker),
            };
            _ = c.nmslib_get_last_error_detail(&detail, &temp_alloc);
            std.debug.print(
                "NMSLIB Error: {s} at {s}:{}\n",
                .{ std.mem.span(detail.message), std.mem.span(detail.file), detail.line },
            );
            c.nmslib_free_string(@constCast(detail.message), &temp_alloc);
            c.nmslib_free_string(@constCast(detail.file), &temp_alloc);
        }
        try mapError(result);

        if (handle == null)
            return error.Runtime;

        const storage = try DataStorage.init(allocator, data_type);

        return .{
            .handle = handle,
            .allocator = allocator,
            .data_type = data_type,
            .dist_type = dist_type,
            .data_storage = storage,
            .built = false,
            .alloc_context = @ptrCast(tracker),
            .cpp_index_phantom = null,
        };
    }



    pub fn deinit(self: *Self) void {
        if (self.built) {
            c.nmslib_index_destroy(self.handle);
            if (self.cpp_index_phantom) |phantom| self.allocator.free(phantom);
        }
        self.alloc_context.deinit();
        self.allocator.destroy(self.alloc_context);
        self.data_storage.deinit(self.allocator);
    }

    pub fn reset(self: *Self) !void {
        try self.clearIndexCache();
        self.data_storage.deinit(self.allocator);
        self.data_storage = try DataStorage.init(self.allocator, self.data_type);
    }

    pub fn buildIndex(self: *Self, index_params: ?Params, print_progress: bool) !void {
        if (self.built) return error.IndexAlreadyBuilt;
        const c_params: ?c.nmslib_params_handle_t = if (index_params) |params| params.c_params else null;
        const progress: i32 = if (print_progress) 1 else 0;
        try mapError(c.nmslib_create_index(self.handle, c_params orelse null, progress));

        self.built = true;
        // Batch add via pointers (unified)
        const len = self.data_storage.descriptors.items.len;
        if (len > 0) {
            const c_ptrs = try self.allocator.alloc(?*const anyopaque, len);
            defer self.allocator.free(c_ptrs);
            const c_ids = try self.allocator.alloc(i32, len);
            defer self.allocator.free(c_ids);
            for (0..len) |i| {
                const desc = self.data_storage.descriptors.items[i];
                c_ptrs[i] = switch (self.data_type) {
                    .DenseVector => @as(?*const anyopaque, @ptrCast(desc.DenseVector.data.ptr)),
                    .SparseVector => @as(?*const anyopaque, @ptrCast(desc.SparseVector.data.ptr)),
                    .DenseUInt8Vector => @as(?*const anyopaque, @ptrCast(desc.DenseUInt8Vector.data.ptr)),
                    .ObjectAsString => @as(?*const anyopaque, @ptrCast(desc.ObjectAsString.data.ptr)),
                };

                const desc_id: i32 = switch (self.data_type) {
                    .DenseVector => desc.DenseVector.id,
                    .SparseVector => desc.SparseVector.id,
                    .DenseUInt8Vector => desc.DenseUInt8Vector.id,
                    .ObjectAsString => desc.ObjectAsString.id,
                };

                c_ids[i] = if (desc_id != 0) desc_id else @as(i32, @intCast(i));
            }




            var num_elems_slice: ?[]usize = null;
            defer if (num_elems_slice) |slice| self.allocator.free(slice);
            var num_elems_ptr: ?[*]const usize = null;
            if (self.data_type == .SparseVector) {
                num_elems_slice = try self.allocator.alloc(usize, len);
                for (0..len) |i| {
                    num_elems_slice.?[i] = self.data_storage.descriptors.items[i].SparseVector.num_elements;
                }
                num_elems_ptr = num_elems_slice.?.ptr;
            }

            // Handle string data separately — don't call the pointer-batch function for string objects.
            if (self.data_type == .ObjectAsString) {
                // The C API for string batches expects const char* const* data and ids.
                // c_ptrs currently contains pointers to the string data (`ObjectAsString.data_ptr`).
                // Cast the generic pointer array to the expected C pointer type when calling.
                try mapError(c.nmslib_add_data_point_batch_string(
                    self.handle,
                    @as([*] [*]const u8, @ptrCast(c_ptrs.ptr)), // cast to const char* const*
                    len,
                    c_ids.ptr
                ));
            } else {
                const mode = self.data_type.toDataMode();
                try mapError(c.nmslib_add_data_point_batch_pointers(
                    self.handle,
                    mode,
                    c_ptrs.ptr,
                    len,
                    self.getDim(),
                    c_ids.ptr,
                    num_elems_ptr
                ));
            }
        }
    }


    pub fn clearIndexCache(self: *Self) !void {
        if (!self.built) return;
        c.nmslib_index_destroy(self.handle);
        if (self.cpp_index_phantom) |phantom| {
            self.allocator.free(phantom);
            self.cpp_index_phantom = null;
        }
        self.built = false;
    }

    fn getDim(self: *Self) usize {
        if (self.data_storage.descriptors.items.len == 0) return 0;
        return switch (self.data_storage.descriptors.items[0]) {
            .DenseVector => |d| d.data.len,
            .DenseUInt8Vector => |d| d.data.len,
            .SparseVector, .ObjectAsString => 0,
        };
    }


    pub fn addDenseBatch(self: *Self, data: []const []const f32, ids: ?[]const i32) !void {
        if (self.built) return error.IndexAlreadyBuilt;
        if (data.len == 0) return error.InvalidArgument;

        const dim = data[0].len;
        for (data, 0..) |vec, i| {
            if (vec.len != dim) return error.InvalidArgument;

            const slice = try self.data_storage.arena.allocator().alloc(f32, dim);
            std.mem.copyForwards(f32, slice, vec);

            try self.data_storage.descriptors.append(.{
                .DenseVector = .{
                    .id = if (ids) |id_slice| id_slice[i] else @intCast(i),
                    .data_ptr = slice.ptr,
                    .length = slice.len,
                    .data = slice,
                },
            });
        }
    }





    pub fn addSparseBatch(self: *Self, data: []const []const SparseElem, ids: ?[]const i32) !void {
        if (self.built) return error.IndexAlreadyBuilt;
        if (data.len == 0) return error.InvalidArgument;

        // --- Defensive validation: ensure non-empty rows, strictly increasing & positive element ids ---
        for (data) |row| {
            if (row.len == 0) return error.InvalidArgument;
            var last_id: u32 = 0;
            for (row) |elem| {
                // Require strictly increasing and positive IDs (NMSLIB requires 1-based indices)
                if (elem.id == 0 or elem.id <= last_id) return error.InvalidArgument;
                last_id = elem.id;
            }
        }
        // --- end validation ---

        // Allocate and register sparse data
        for (data, 0..) |row, i| {
            const ptr = try self.data_storage.arena.allocator().alloc(SparseElem, row.len);
            std.mem.copyForwards(SparseElem, ptr, row);

            try self.data_storage.descriptors.append(.{
                .SparseVector = .{
                    // ✅ Use 1-based default IDs (NMSLIB convention)
                    .id = if (ids) |id_slice| id_slice[i] else @intCast(i + 1),
                    .data_ptr = ptr.ptr,
                    .num_elements = ptr.len,
                    .data = ptr,
                },
            });
        }
    }





    pub fn addUInt8Batch(self: *Self, data: []const []const u8, ids: ?[]const i32) !void {
        if (self.built) return error.IndexAlreadyBuilt;
        if (data.len == 0 or data[0].len == 0) return error.InvalidArgument;

        const dim = data[0].len;
        for (data, 0..) |vec, i| {
            if (vec.len != dim) return error.InvalidArgument;

            const ptr = try self.data_storage.arena.allocator().alloc(u8, dim);
            std.mem.copyForwards(u8, ptr, vec);

            try self.data_storage.descriptors.append(.{
                .DenseUInt8Vector = .{
                    .id = if (ids) |id_slice| id_slice[i] else @intCast(i),
                    .data_ptr = ptr.ptr,
                    .length = ptr.len,
                    .data = ptr,
                },
            });
        }
    }



    pub fn addStringBatch(self: *Self, data: []const []const u8, ids: ?[]const i32) !void {
        if (self.built) return error.IndexAlreadyBuilt;
        if (data.len == 0) return error.InvalidArgument;
        var total_size: usize = 0;
        for (data) |str| total_size += str.len + 1;
        try self.data_storage.descriptors.ensureTotalCapacityPrecise(total_size);

        for (data, 0..) |str, i| {
            const ptr = try self.data_storage.arena.allocator().dupe(u8, str);
            try self.data_storage.descriptors.append(.{
                .ObjectAsString = .{
                    .id = if (ids) |id_slice| id_slice[i] else @intCast(i),
                    .data_ptr = ptr.ptr,
                    .length = ptr.len,
                    .data = ptr,
                },
            });
        }
    }

    pub fn knnQuery(self: *Self, query_in: QueryPoint, k: usize) !QueryResult {
        if (!self.built) try self.buildIndex(null, false);

        c.nmslib_initialize_pool(self.handle);

        var c_query_ptr: ?*const anyopaque = null;
        var c_query_len: usize = 0;
        var c_num_elements: usize = 0;

        switch (query_in) {
            .DenseVector => |q| {
                c_query_ptr = @ptrCast(q.ptr);
                c_query_len = q.len;        // number of floats
                c_num_elements = 0;
            },
            .SparseVector => |q| {
                c_query_ptr = @ptrCast(q.ptr);
                c_query_len = q.len;        // number of sparse elems
                c_num_elements = q.len;
            },
            .DenseUInt8Vector => |q| {
                c_query_ptr = @ptrCast(q.ptr);
                c_query_len = q.len;        // number of bytes
                c_num_elements = 0;
            },
            .ObjectAsString => |q| {
                c_query_ptr = @ptrCast(q.ptr);
                c_query_len = q.len;        // length of string bytes
                c_num_elements = 0;
            },
        }

        // Ask C how many slots are needed
        var capacity: usize = 0;
        try mapError(c.nmslib_knn_query_get_size(
            self.handle,
            c_query_ptr,
            c_query_len,
            k,
            &capacity,
            c_num_elements,
        ));
        if (capacity == 0) capacity = k;

        // Allocate result buffers (owned by the returned QueryResult on success)
        const ids = try self.allocator.alloc(i32, capacity);
        const distances = try self.allocator.alloc(f32, capacity);

        // Ensure we free on error; disable before returning
        var free_on_err: bool = true;
        defer if (free_on_err) {
            self.allocator.free(ids);
            self.allocator.free(distances);
        };

        var c_result = c.nmslib_result_t{
            .ids = ids.ptr,
            .distances = distances.ptr,
            .size = 0,
            .capacity = capacity,
        };

        try mapError(c.nmslib_knn_query_fill(
            self.handle,
            c_query_ptr,
            c_query_len,
            k,
            &c_result,
            c_num_elements,
        ));

        if (c_result.size > c_result.capacity) return error.Runtime;
        const n = c_result.size;

        // Prevent freeing of the buffers — QueryResult will own them
        free_on_err = false;

        const ids_trimmed = ids[0..n];
        const dist_trimmed = distances[0..n];

        return QueryResult{
            .ids = ids_trimmed,
            .distances = dist_trimmed,
            .full_ids = ids,                // full allocated buffers: will be freed by QueryResult.deinit
            .full_distances = distances,
            .used = n,
            .allocator = self.allocator,
        };
    }




    pub fn knnQueryBatch(self: *Self, queries: []const []const f32, k: usize, thread_pool_size: ?usize) !BatchResult {
        if (!self.built) try self.buildIndex(null, false);

        c.nmslib_initialize_pool(self.handle);

        _ = thread_pool_size; // intentionally unused for now
        var results = try self.allocator.alloc(QueryResult, queries.len);
        errdefer {
            for (0..queries.len) |j| {
                if (results[j].ids.len > 0) {
                    results[j].deinit();
                }
            }
            self.allocator.free(results);
        }

        for (0..queries.len) |i| {
            var capacity: usize = undefined;
            try mapError(c.nmslib_knn_query_get_size(self.handle, queries[i].ptr, queries[i].len, k, &capacity, 0));

            const ids = try self.allocator.alloc(i32, capacity);
            var free_ids_on_err: bool = true;
            defer if (free_ids_on_err) self.allocator.free(ids);

            const distances = try self.allocator.alloc(f32, capacity);
            var free_dist_on_err: bool = true;
            defer if (free_dist_on_err) self.allocator.free(distances);

            var c_result = c.nmslib_result_t{ .ids = ids.ptr, .distances = distances.ptr, .size = 0, .capacity = capacity };

            try mapError(c.nmslib_knn_query_fill(self.handle, queries[i].ptr, queries[i].len, k, &c_result, 0));

            if (c_result.size > c_result.capacity) return error.Runtime;
            const n = c_result.size;

            // transfer ownership for this result
            results[i] = .{ .ids = ids[0..n], .distances = distances[0..n], .allocator = self.allocator };
            free_ids_on_err = false;
            free_dist_on_err = false;
        }

        return .{ .results = results, .allocator = self.allocator };
    }




    pub fn rangeQuery(self: *Self, query: []const f32, radius: f64) !QueryResult {
        if (!self.built) try self.buildIndex(null, false);
        if (self.data_type != .DenseVector) return error.SpaceIncompatible;

        var size: usize = undefined;
        try mapError(c.nmslib_range_query_get_size(self.handle, query.ptr, query.len, radius, &size, 0));

        const ids = try self.allocator.alloc(i32, size);
        errdefer self.allocator.free(ids);
        const distances = try self.allocator.alloc(f32, size);
        errdefer self.allocator.free(distances);

        var result = c.nmslib_result_t{
            .ids = ids.ptr,
            .distances = distances.ptr,
            .size = 0,
            .capacity = size,
        };

        try mapError(c.nmslib_range_query_fill(self.handle, query.ptr, query.len, radius, &result, 0));

        if (result.size > result.capacity) return error.Runtime;
        const n = result.size;

        // --- 🩹 FIX START ---
        std.debug.assert(n <= size);

        return .{
            .ids = ids[0..n],
            .distances = distances[0..n],
            .used = n,
            .allocator = self.allocator,
        };
        // --- 🩹 FIX END ---
    }




    pub fn getDistance(self: *Self, pos1: usize, pos2: usize) !f32 {
        if (!self.built) try self.buildIndex(null, false);
        var distance: f32 = std.math.nan(f32);
        try mapError(c.nmslib_get_distance(self.handle, pos1, pos2, &distance));
        if (std.math.isNan(distance)) return error.QueryExecutionFailed;
        return distance;
    }


    pub fn getDataPoint(self: *Self, pos: usize) !DataPoint {
        if (self.data_storage.descriptors.items.len == 0 and self.built)
            return error.DataNotLoaded;
        const qty = self.dataQty();
        if (pos >= qty) return error.InvalidArgument;
        const desc = self.data_storage.descriptors.items[pos];
        return switch (desc) {
            .DenseVector => |d| .{ .DenseVector = d.data },
            .SparseVector => |s| .{ .SparseVector = s.data },
            .DenseUInt8Vector => |u| .{ .DenseUInt8Vector = u.data },
            .ObjectAsString => |o| .{ .ObjectAsString = o.data },
        };
    }


    /// Returns a borrowed slice to the internal string data.
    /// The memory is owned by the index’s arena; do NOT free it.
    pub fn borrowDataPointString(self: *Self, pos: usize) ![]const u8 {
        const qty = self.dataQty();
        if (pos >= qty) return error.InvalidArgument;
        const o = self.data_storage.descriptors.items[pos].ObjectAsString;
        return o.data_ptr[0..o.length];
    }


    pub fn borrowDataDense(self: *Self, pos: usize) !struct { data: []const f32, free_fn: *const fn (?*anyopaque) void } {
        if (self.data_type != .DenseVector) return error.SpaceIncompatible;
        const qty = self.dataQty();
        if (pos >= qty) return error.InvalidArgument;
        const d = self.data_storage.descriptors.items[pos].DenseVector;
        return .{ .data = d.data_ptr[0..d.length], .free_fn = struct { fn noop(_: ?*anyopaque) void {} }.noop };  // No-op free
    }

    pub fn borrowDataSparse(self: *Self, pos: usize) !struct { data: []const SparseElem, free_fn: *const fn (?*anyopaque) void } {
        if (self.data_type != .SparseVector) return error.SpaceIncompatible;
        const qty = self.dataQty();
        if (pos >= qty) return error.InvalidArgument;
        const s = self.data_storage.descriptors.items[pos].SparseVector;
        return .{ .data = s.data_ptr[0..s.num_elements], .free_fn = struct { fn noop(_: ?*anyopaque) void {} }.noop };
    }

    pub fn save(self: *Self, path: []const u8, save_data: bool) !void {
        if (!self.built) try self.buildIndex(null, false);
        const path_z = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(path_z);
        try mapError(c.nmslib_save_index(self.handle, path_z, @intFromBool(save_data)));
    }
    fn loadDescriptors(self: *Self) !void {
        const qty = c.nmslib_data_qty(self.handle);
        const alloc = self.data_storage.arena.allocator();

        for (0..qty) |pos| {
            var size_bytes: usize = 0;
            try mapError(c.nmslib_get_data_point_size(self.handle, pos, &size_bytes));

            switch (self.data_type) {
                .DenseVector => {
                    // Ensure byte-size aligns with f32 elements
                    if (size_bytes % @sizeOf(f32) != 0) return error.Runtime;
                    const elem_count = size_bytes / @sizeOf(f32);

                    var tmp_bytes = try alloc.alloc(u8, size_bytes);
                    defer alloc.free(tmp_bytes);

                    try mapError(
                        c.nmslib_get_data_point_fill(
                            self.handle,
                            pos,
                            @ptrCast(tmp_bytes.ptr),
                            size_bytes,
                        ),
                    );

                    // Interpret the bytes as f32 values (const view).
                    const buf = @as([*]const f32, @alignCast(@ptrCast(tmp_bytes.ptr)))[0..elem_count];
                    const copied = try alloc.dupe(f32, buf);

                    const desc = Descriptor{
                        .DenseVector = .{
                            .id = @intCast(pos),
                            .data_ptr = copied.ptr,
                            .length = copied.len,
                            .data = copied,
                        },
                    };
                    try self.data_storage.descriptors.append(desc);

                },

                .SparseVector => {
                    if (size_bytes % @sizeOf(SparseElem) != 0) return error.Runtime;
                    const elem_count = size_bytes / @sizeOf(SparseElem);

                    var tmp_bytes = try alloc.alloc(u8, size_bytes);
                    defer alloc.free(tmp_bytes);

                    try mapError(
                        c.nmslib_get_data_point_fill(
                            self.handle,
                            pos,
                            @ptrCast(tmp_bytes.ptr),
                            size_bytes,
                        ),
                    );

                    const buf = @as([*]const SparseElem, @alignCast(@ptrCast(tmp_bytes.ptr)))[0..elem_count];
                    const copied = try alloc.dupe(SparseElem, buf);

                    const desc = Descriptor{
                        .SparseVector = .{
                            .id = @intCast(pos),
                            .data_ptr = copied.ptr,
                            .num_elements = copied.len,
                            .data = copied,
                        },
                    };
                    try self.data_storage.descriptors.append(desc);
                },

                .DenseUInt8Vector => {
                    if (size_bytes == 0) return error.Runtime;

                    var buf = try alloc.alloc(u8, size_bytes);
                    defer alloc.free(buf);

                    try mapError(
                        c.nmslib_get_data_point_fill(
                            self.handle,
                            pos,
                            @ptrCast(buf.ptr),
                            size_bytes,
                        ),
                    );

                    const copied = try alloc.dupe(u8, buf);

                    const desc = Descriptor{
                        .DenseUInt8Vector = .{
                            .id = @intCast(pos),
                            .data_ptr = copied.ptr,
                            .length = copied.len,
                            .data = copied,
                        },
                    };
                    try self.data_storage.descriptors.append(desc);
                },

                .ObjectAsString => {
                    var tmp_bytes = try alloc.alloc(u8, size_bytes);
                    defer alloc.free(tmp_bytes);

                    try mapError(
                        c.nmslib_get_data_point_fill(
                            self.handle,
                            pos,
                            @ptrCast(tmp_bytes.ptr),
                            size_bytes,
                        ),
                    );

                    const slice = tmp_bytes[0..size_bytes];
                    const copied = try alloc.dupe(u8, slice);

                    const desc = Descriptor{
                        .ObjectAsString = .{
                            .id = @intCast(pos),
                            .data_ptr = copied.ptr,
                            .length = copied.len,
                            .data = copied,
                        },
                    };
                    try self.data_storage.descriptors.append(desc);
                },
            }
        }
    }








    pub fn load(allocator: Allocator, path: []const u8, data_type: DataType, dist_type: DistType, load_data: bool) !Self {
        const tracker = try allocator.create(AllocContext);
        errdefer allocator.destroy(tracker);
        tracker.* = try AllocContext.init(allocator);
        errdefer tracker.deinit();

        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(tracker),
        };
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        var handle: c.nmslib_index_handle_t = undefined;
        try mapError(c.nmslib_load_index(path_z, data_type.toC(), dist_type.toC(), &c_alloc, @intFromBool(load_data), &handle));
        if (handle == null) return error.Runtime;

        const storage = try DataStorage.init(allocator, data_type);
        var phantom: ?[]u8 = null;
        if (track_cpp_memory) {
            const mem_size = c.nmslib_index_memory_usage(handle);
            phantom = try allocator.alloc(u8, mem_size);
        }

        var index = Self{
            .handle = handle,
            .allocator = allocator,
            .alloc_context = tracker,
            .data_type = data_type,
            .dist_type = dist_type,
            .data_storage = storage,
            .built = true,
            .cpp_index_phantom = phantom,
        };

        // 🔧 Populate Zig-side descriptors from NMSLIB if data was saved
        if (load_data) try index.loadDescriptors();

        return index;
    }



    pub fn setQueryTimeParams(self: *Self, params: Params) !void {
        if (!self.built) try self.buildIndex(null, false);
        try mapError(c.nmslib_set_query_time_params(self.handle, params.c_params));
    }

    pub fn setThreadPoolSize(self: *Self, size: usize) !void {
        if (!self.built) try self.buildIndex(null, false);
        try mapError(c.nmslib_set_thread_pool_size(self.handle, size));
    }

    pub fn getThreadPoolSize(self: *Self) usize {
        if (!self.built) return 0;
        return c.nmslib_get_thread_pool_size(self.handle);
    }

    pub fn dataQty(self: *Self) usize {
        if (self.built) return c.nmslib_data_qty(self.handle);
        return self.data_storage.descriptors.items.len;
    }

    pub fn getSpaceType(self: *Self) ![]const u8 {
        var space_type: [*c]const u8 = undefined;
        var space_type_len: usize = undefined;
        const c_alloc = c.nmslib_allocator_t{ .alloc = c_alloc_fn, .free = c_free_fn, .ctx = @ptrCast(self.alloc_context) };
        try mapError(c.nmslib_get_space_type(self.handle, &space_type, &space_type_len, &c_alloc));
        defer c.nmslib_free_string(@constCast(space_type), &c_alloc);
        return try self.allocator.dupe(u8, space_type[0..space_type_len]);
    }

    pub fn getMethod(self: *Self) ![]const u8 {
        var method: [*c]const u8 = undefined;
        var method_len: usize = undefined;
        const c_alloc = c.nmslib_allocator_t{ .alloc = c_alloc_fn, .free = c_free_fn, .ctx = @ptrCast(self.alloc_context) };
        try mapError(c.nmslib_get_method(self.handle, &method, &method_len, &c_alloc));
        defer c.nmslib_free_string(@constCast(method), &c_alloc);
        return try self.allocator.dupe(u8, method[0..method_len]);
    }

    pub fn getLastErrorDetail(self: *Self) !struct { code: Error, message: []const u8, file: []const u8, line: i32 } {
        var detail: c.nmslib_error_detail_t = undefined;
        const c_alloc = c.nmslib_allocator_t{ .alloc = c_alloc_fn, .free = c_free_fn, .ctx = @ptrCast(self.alloc_context) };
        try mapError(c.nmslib_get_last_error_detail(&detail, &c_alloc));
        const msg = try self.allocator.dupe(u8, std.mem.span(detail.message));
        errdefer self.allocator.free(msg);
        const file = try self.allocator.dupe(u8, std.mem.span(detail.file));
        errdefer self.allocator.free(file);
        return .{
            .code = mapError(detail.code) catch error.Runtime,
            .message = msg,
            .file = file,
            .line = detail.line,
        };
    }

    pub fn getDataType(self: *Self) DataType {
        return self.data_type;
    }
};

// Tests (fixed: use addStringBatch; fixed query for string; added deinit for strings)
test "Index dense vector workflow" {
    const allocator = testing.allocator;
    var params = try Params.init(allocator);
    defer params.deinit();
    try params.add("dim", .{ .Int = 4 });
    var index = try Index.init(allocator, "l2", params, "hnsw", .DenseVector, .Float);
    defer index.deinit();
    const data = [_][]const f32{ &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, &[_]f32{ 0.0, 1.0, 0.0, 0.0 }, &[_]f32{ 0.0, 0.0, 1.0, 0.0 } };
    const ids = [_]i32{ 10, 20, 30 };
    try index.addDenseBatch(&data, &ids);
    try index.buildIndex(null, false);
    try testing.expectEqual(3, index.dataQty());
    const space_type = try index.getSpaceType();
    defer allocator.free(space_type);
    try testing.expectEqualStrings("l2", space_type);
    const method = try index.getMethod();
    defer allocator.free(method);
    try testing.expectEqualStrings("hnsw", method);
    const query = QueryPoint{ .DenseVector = &[_]f32{ 1.0, 0.0, 0.0, 0.0 } };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();
    try testing.expectEqual(2, result.ids.len);
    try testing.expectApproxEqAbs(0.0, result.distances[0], 0.0001);
    try testing.expectEqual(10, result.ids[0]);
    const dist = try index.getDistance(0, 1);
    try testing.expectApproxEqAbs(@sqrt(2.0), dist, 0.0001);  // Fixed sqrt
    const data_point = try index.getDataPoint(0);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, data_point.DenseVector);
    const borrowed = try index.borrowDataDense(0);
    defer borrowed.free_fn(null);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, borrowed.data);
    try index.save("test_index", true);
    try index.reset();
    try testing.expectEqual(0, index.dataQty());
    var loaded_index = try Index.load(allocator, "test_index", .DenseVector, .Float, true);
    defer loaded_index.deinit();
    try testing.expectEqual(3, loaded_index.dataQty());
    const loaded_point = try loaded_index.getDataPoint(0);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, loaded_point.DenseVector);
}

test "Index sparse vector workflow" {
    const allocator = testing.allocator;
    var index = try Index.init(allocator, "cosinesimil_sparse", null, "hnsw", .SparseVector, .Float);
    defer index.deinit();

    // ✅ Sparse IDs are 1-based and strictly increasing (required by NMSLIB)
    const data = [_][]const SparseElem{
        &[_]SparseElem{ .{ .id = 1, .value = 1.0 }, .{ .id = 2, .value = 2.0 } },
        &[_]SparseElem{ .{ .id = 1, .value = 1.0 }, .{ .id = 3, .value = 3.0 } },
    };
    const ids = [_]i32{ 100, 200 };

    try index.addSparseBatch(&data, &ids);
    try index.buildIndex(null, false);

    const query = QueryPoint{ .SparseVector = &[_]SparseElem{ .{ .id = 1, .value = 1.0 } } };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();

    try testing.expectEqual(2, result.ids.len);

    const borrowed = try index.borrowDataSparse(0);
    defer borrowed.free_fn(null);

    // ✅ Expect 1-based ids — fix test to align with NMSLIB output
    try testing.expectEqualSlices(
        SparseElem,
        &[_]SparseElem{
            .{ .id = 1, .value = 1.0 },
            .{ .id = 2, .value = 2.0 },
        },
        borrowed.data,
    );
}

test "Index uint8 vector workflow" {
    const allocator = testing.allocator;
    // "l2sqr_sift" requires integer distance and 128-D vectors
    var index = try Index.init(allocator, "l2sqr_sift", null, "hnsw", .DenseUInt8Vector, .Int);
    defer index.deinit();

    // Build two valid 128-byte SIFT-style descriptors
    var desc0: [128]u8 = undefined;
    var desc1: [128]u8 = undefined;

    // Fill descriptors deterministically
    for (desc0[0..], 0..) |_, i| desc0[i] = @as(u8, @intCast(i % 256));
    for (desc1[0..], 0..) |_, i| desc1[i] = @as(u8, @intCast((i + 7) % 256));

    // Batch input array
    const data = [_][]const u8{ desc0[0..], desc1[0..] };

    // Optional sanity check
    try testing.expectEqual(@as(usize, 128), data[0].len);

    try index.addUInt8Batch(&data, null);
    try index.buildIndex(null, false);

    // Query using one of the descriptors
    const query = QueryPoint{ .DenseUInt8Vector = desc0[0..] };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();

    try testing.expectEqual(2, result.ids.len);
}



test "Index string data workflow" {
    const allocator = testing.allocator;
    // ✅ Use Int distance for this build — Levenshtein space registered only for integer distance
    var index = try Index.init(allocator, "leven", null, "hnsw", .ObjectAsString, .Int);
    defer index.deinit();

    const data = [_][]const u8{ "hello", "world" };
    try index.addStringBatch(&data, null);
    try index.buildIndex(null, false);

    const query = QueryPoint{ .ObjectAsString = "hello" };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();

    try testing.expectEqual(2, result.ids.len);
    const str = try index.borrowDataPointString(0);
    try testing.expectEqualStrings("hello", str);
}
