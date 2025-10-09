const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;

// Import the C interface
const c = @cImport({
    @cInclude("nmslib_c.h");
});

// Error set mapping to C wrapper errors
pub const Error = Allocator.Error || error{
    NullPointer,
    InvalidArgument,
    OutOfMemory,
    SpaceIncompatible,
    QueryTooLarge,
    InvalidSparseElement,
    IndexBuildFailed,
    QueryExecutionFailed,
    DataIOFailed,
    PluginRegistrationFailed,
    Internal,
    Runtime,
};

// Maps C error codes to Zig errors
fn mapError(err: c.nmslib_error_t) Error!void {
    return switch (err) {
        c.NMSLIB_SUCCESS => {},
        c.NMSLIB_ERROR_NULL_POINTER => error.NullPointer,
        c.NMSLIB_ERROR_INVALID_ARGUMENT => error.InvalidArgument,
        c.NMSLIB_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        c.NMSLIB_ERROR_SPACE_INCOMPATIBLE => error.SpaceIncompatible,
        c.NMSLIB_ERROR_QUERY_TOO_LARGE => error.QueryTooLarge,
        c.NMSLIB_ERROR_INVALID_SPARSE_ELEMENT => error.InvalidSparseElement,
        c.NMSLIB_ERROR_INDEX_BUILD_FAILED => error.IndexBuildFailed,
        c.NMSLIB_ERROR_QUERY_EXECUTION_FAILED => error.QueryExecutionFailed,
        c.NMSLIB_ERROR_DATA_IO_FAILED => error.DataIOFailed,
        c.NMSLIB_ERROR_PLUGIN_REGISTRATION_FAILED => error.PluginRegistrationFailed,
        c.NMSLIB_ERROR_INTERNAL => error.Internal,
        c.NMSLIB_ERROR_RUNTIME => error.Runtime,
        else => error.Runtime, // Fallback for unknown errors
    };
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

// Parameter management struct
pub const Params = struct {
    c_params: c.nmslib_params_handle_t,
    allocator: Allocator,

    /// Initializes a new parameter set with the given allocator.
    pub fn init(allocator: Allocator) !Params {
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(@constCast(&allocator)),
        };
        const c_params = c.nmslib_create_params(&c_alloc) orelse return error.OutOfMemory;
        return .{ .c_params = c_params, .allocator = allocator };
    }

    /// Frees the parameter set.
    pub fn deinit(self: *Params) void {
        c.nmslib_free_params(self.c_params);
        self.* = undefined;
    }

    /// Adds a parameter with the given key and value.
    pub fn add(self: *Params, key: []const u8, value: ParamValue) !void {
        const key_z = try self.allocator.dupeZ(u8, key);
        defer self.allocator.free(key_z);
        switch (value) {
            .String => |s| {
                const value_z = try self.allocator.dupeZ(u8, s);
                defer self.allocator.free(value_z);
                try mapError(c.nmslib_add_param(self.c_params, key_z, 2, @ptrCast(&value_z)));
            },
            .Int => |i| try mapError(c.nmslib_add_param(self.c_params, key_z, 0, &i)),
            .Double => |d| try mapError(c.nmslib_add_param(self.c_params, key_z, 1, &d)),
        }
    }

    /// Creates a parameter set from a slice of key-value pairs.
    pub fn fromSlice(allocator: Allocator, pairs: []const struct { key: []const u8, value: ParamValue }) !Params {
        var params = try Params.init(allocator);
        errdefer params.deinit();
        for (pairs) |pair| {
            try params.add(pair.key, pair.value);
        }
        return params;
    }
};

// Sparse element struct
pub const SparseElem = struct {
    id: u32,
    value: f32,
};

// Query result struct
pub const QueryResult = struct {
    ids: []i32,
    distances: []f32,
    allocator: Allocator,

    /// Frees the memory associated with the query result.
    pub fn deinit(self: @This()) void {
        self.allocator.free(self.ids);
        self.allocator.free(self.distances);
    }
};

// Batch query result struct
pub const BatchResult = struct {
    results: []QueryResult,
    allocator: Allocator,

    /// Frees the memory associated with the batch query results.
    pub fn deinit(self: @This()) void {
        for (self.results) |result| result.deinit();
        self.allocator.free(self.results);
    }
};

// Data point retrieval struct
pub const DataPoint = union(DataType) {
    DenseVector: []f32,
    SparseVector: []SparseElem,
    DenseUInt8Vector: []u8,
    ObjectAsString: []u8,

    /// Frees the memory associated with the data point.
    pub fn deinit(self: @This(), allocator: Allocator) void {
        switch (self) {
            .DenseVector => |slice| allocator.free(slice),
            .SparseVector => |slice| allocator.free(slice),
            .DenseUInt8Vector => |slice| allocator.free(slice),
            .ObjectAsString => |slice| allocator.free(slice),
        }
    }
};

pub const QueryPoint = union(DataType) {
    DenseVector: []const f32,
    SparseVector: []const SparseElem,
    DenseUInt8Vector: []const u8,
    ObjectAsString: []const u8,
};
// Main index struct
pub const Index = struct {
    handle: c.nmslib_index_handle_t,
    allocator: Allocator,

    // Fixed Index.init to properly handle optional params
    pub fn init(allocator: Allocator, space_type: []const u8, params: ?Params, method: []const u8, data_type: DataType, dist_type: DistType) !Index {
        var handle: c.nmslib_index_handle_t = undefined;
        const space_z = try allocator.dupeZ(u8, space_type);
        defer allocator.free(space_z);
        const method_z = try allocator.dupeZ(u8, method);
        defer allocator.free(method_z);
        const c_params: c.nmslib_params_handle_t = if (params) |p| p.c_params else null;
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(@constCast(&allocator)),
        };
        try mapError(c.nmslib_index_create(space_z, c_params, method_z, data_type.toC(), dist_type.toC(), &c_alloc, &handle));
        return Index{
            .allocator = allocator,
            .handle = handle,
        };
    }

    /// Frees the index.
    pub fn deinit(self: *Index) void {
        c.nmslib_index_destroy(self.handle);
        self.* = undefined;
    }

    /// Resets the index, clearing all data.
    pub fn reset(self: *Index) !void {
        try mapError(c.nmslib_reset_index(self.handle));
    }

    /// Creates the index with the specified parameters.
    pub fn create(self: *Index, params: ?Params, print_progress: bool) !void {
        const c_params: c.nmslib_params_handle_t = if (params) |p| p.c_params else null;
        try mapError(c.nmslib_create_index(self.handle, c_params, @intFromBool(print_progress)));
    }

    /// Adds a batch of dense vectors to the index.
    pub fn addDenseBatch(self: *Index, data: []const []const f32, ids: ?[]const i32) !void {
        if (data.len == 0 or data[0].len == 0) return error.InvalidArgument;
        const c_ids: ?[*]const i32 = if (ids) |i| i.ptr else null;
        try mapError(c.nmslib_add_data_point_batch(self.handle, @ptrCast(data.ptr), data.len, data[0].len, c_ids, null));
    }

    /// Adds a batch of sparse vectors to the index.
    pub fn addSparseBatch(self: *Index, data: []const []const SparseElem, ids: ?[]const i32) !void {
        if (data.len == 0) return error.InvalidArgument;
        const c_ids: ?[*]const i32 = if (ids) |i| i.ptr else null;
        var num_elements = try self.allocator.alloc(usize, data.len);
        defer self.allocator.free(num_elements);
        for (data, 0..) |row, i| num_elements[i] = row.len;
        try mapError(c.nmslib_add_data_point_batch(self.handle, @ptrCast(data.ptr), data.len, 0, c_ids, num_elements.ptr));
    }

    /// Adds a batch of uint8 vectors to the index.
    pub fn addUInt8Batch(self: *Index, data: []const []const u8, ids: ?[]const i32) !void {
        if (data.len == 0 or data[0].len == 0) return error.InvalidArgument;
        const c_ids: ?[*]const i32 = if (ids) |i| i.ptr else null;
        try mapError(c.nmslib_add_data_point_batch_uint8(self.handle, @ptrCast(data.ptr), data.len, data[0].len, c_ids));
    }

    /// Adds a batch of string data points to the index.
    pub fn addStringBatch(self: *Index, data: []const []const u8, ids: ?[]const i32) !void {
        if (data.len == 0) return error.InvalidArgument;
        const c_data = try self.allocator.alloc([*c]const u8, data.len);
        defer self.allocator.free(c_data);
        for (data, 0..) |str, i| {
            c_data[i] = str.ptr;
        }
        const c_ids: ?[*]const i32 = if (ids) |i| i.ptr else null;
        try mapError(c.nmslib_add_data_point_batch_string(self.handle, c_data.ptr, data.len, c_ids));
    }

    /// Performs a k-NN query for a single query vector.
    pub fn knnQuery(self: *Index, query_in: QueryPoint, k: usize) !QueryResult {
        const data_type = try self.getDataType();
        if (std.meta.activeTag(query_in) != data_type) return error.SpaceIncompatible;

        var size: usize = undefined;
        var c_query_ptr: *const anyopaque = undefined;
        var c_query_len: usize = undefined;
        var c_num_elements: usize = 0;

        switch (query_in) {
            .DenseVector => |q| {
                c_query_ptr = q.ptr;
                c_query_len = q.len;
                c_num_elements = 0;
            },
            .SparseVector => |q| {
                c_query_ptr = q.ptr;
                c_query_len = 0;
                c_num_elements = q.len;
            },
            .DenseUInt8Vector => |q| {
                c_query_ptr = q.ptr;
                c_query_len = q.len;
                c_num_elements = 0;
            },
            .ObjectAsString => |q| {
                c_query_ptr = q.ptr;
                c_query_len = q.len;
                c_num_elements = 0;
            },
        }

        try mapError(c.nmslib_knn_query_get_size(self.handle, c_query_ptr, c_query_len, k, &size, c_num_elements));

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
        try mapError(c.nmslib_knn_query_fill(self.handle, c_query_ptr, c_query_len, k, &result, c_num_elements));

        return QueryResult{
            .ids = ids[0..result.size],
            .distances = distances[0..result.size],
            .allocator = self.allocator,
        };
    }

    /// Performs a batch k-NN query.
    pub fn knnQueryBatch(self: *Index, queries: []const []const f32, k: usize, thread_pool_size: ?usize) !BatchResult {
        if (queries.len == 0 or queries[0].len == 0) return error.InvalidArgument;

        const c_thread_pool_size = thread_pool_size orelse c.nmslib_get_thread_pool_size(self.handle);
        var c_results = try self.allocator.alloc(c.nmslib_result_t, queries.len);
        defer self.allocator.free(c_results);
        var results = try self.allocator.alloc(QueryResult, queries.len);
        errdefer {
            for (results) |r| r.deinit();
            self.allocator.free(results);
        }

        for (0..queries.len) |i| {
            var size: usize = undefined;
            try mapError(c.nmslib_knn_query_get_size(self.handle, queries[i].ptr, queries[i].len, k, &size, 0));
            const ids = try self.allocator.alloc(i32, size);
            errdefer self.allocator.free(ids);
            const distances = try self.allocator.alloc(f32, size);
            errdefer self.allocator.free(distances);

            c_results[i] = .{
                .ids = ids.ptr,
                .distances = distances.ptr,
                .size = 0,
                .capacity = size,
            };
            results[i] = .{
                .ids = ids[0..size],
                .distances = distances[0..size],
                .allocator = self.allocator,
            };
        }

        try mapError(c.nmslib_knn_query_batch(self.handle, @ptrCast(queries.ptr), queries.len, queries[0].len, k, c_results.ptr, null, c_thread_pool_size));

        for (0..queries.len) |i| {
            results[i].ids = results[i].ids[0..c_results[i].size];
            results[i].distances = results[i].distances[0..c_results[i].size];
        }

        return BatchResult{ .results = results, .allocator = self.allocator };
    }

    /// Performs a range query for a single query vector.
    pub fn rangeQuery(self: *Index, query: []const f32, radius: f64) !QueryResult {
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

        return QueryResult{
            .ids = ids[0..result.size],
            .distances = distances[0..result.size],
            .allocator = self.allocator,
        };
    }

    /// Computes the distance between two data points.
    pub fn getDistance(self: *Index, pos1: usize, pos2: usize) !f32 {
        var distance: f32 = undefined;
        try mapError(c.nmslib_get_distance(self.handle, pos1, pos2, &distance));
        return distance;
    }

    /// Retrieves a data point at the specified position.
    pub fn getDataPoint(self: *Index, pos: usize) !DataPoint {
        var size: usize = undefined;
        try mapError(c.nmslib_get_data_point_size(self.handle, pos, &size));

        const buffer = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(buffer);

        try mapError(c.nmslib_get_data_point_fill(self.handle, pos, buffer.ptr, size));

        const data_type = try self.getDataType();
        return switch (data_type) {
            .DenseVector => DataPoint{ 
                .DenseVector = @as([*]f32, @alignCast(@ptrCast(buffer.ptr)))[0..size / @sizeOf(f32)] 
            },
            .SparseVector => DataPoint{ 
                .SparseVector = @as([*]SparseElem, @alignCast(@ptrCast(buffer.ptr)))[0..size / @sizeOf(SparseElem)] 
            },
            .DenseUInt8Vector => DataPoint{ .DenseUInt8Vector = buffer },
            .ObjectAsString => DataPoint{ .ObjectAsString = buffer },
        };
    }

    /// Retrieves a string data point at the specified position.
    pub fn getDataPointString(self: *Index, pos: usize) ![]u8 {
        var data: [*c]const u8 = undefined;
        var data_len: usize = undefined;
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(&self.allocator),
        };
        try mapError(c.nmslib_get_data_point_string(self.handle, pos, &data, &data_len, &c_alloc));
        defer c.nmslib_free_string(@constCast(data), &c_alloc);

        return try self.allocator.dupe(u8, data[0..data_len]);
    }

    /// Borrows a dense data point at the specified position.
    pub fn borrowDataDense(self: *Index, pos: usize) !struct { data: []const f32, free_fn: *const fn (*anyopaque) void } {
        var data: ?*anyopaque = null;
        var size: usize = undefined;
        var free_fn: ?*const fn (*anyopaque) void = null;
        try mapError(c.nmslib_borrow_data_dense(self.handle, pos, &data, &size, &free_fn));
        if (data == null) return error.Runtime;
        return .{
            .data = @as([*]const f32, @alignCast(@ptrCast(data.?)))[0..size],
            .free_fn = free_fn orelse return error.Runtime,
        };
    }

    /// Borrows a sparse data point at the specified position.
    pub fn borrowDataSparse(self: *Index, pos: usize) !struct { data: []const SparseElem, free_fn: *const fn (*anyopaque) void } {
        var data: ?*anyopaque = null;
        var size: usize = undefined;
        var free_fn: ?*const fn (*anyopaque) void = null;
        try mapError(c.nmslib_borrow_data_sparse(self.handle, pos, &data, &size, &free_fn));
        if (data == null) return error.Runtime;
        return .{
            .data = @as([*]const SparseElem, @alignCast(@ptrCast(data.?)))[0..size],
            .free_fn = free_fn orelse return error.Runtime,
        };
    }

    /// Saves the index to a file.
    pub fn save(self: *Index, path: []const u8, save_data: bool) !void {
        const path_z = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(path_z);
        try mapError(c.nmslib_save_index(self.handle, path_z, @intFromBool(save_data)));
    }

    /// Loads an index from a file.
    pub fn load(allocator: Allocator, path: []const u8, data_type: DataType, dist_type: DistType, load_data: bool) !Index {
        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(@constCast(&allocator)),
        };
        var handle: c.nmslib_index_handle_t = undefined;
        try mapError(c.nmslib_load_index(path_z, data_type.toC(), dist_type.toC(), &c_alloc, @intFromBool(load_data), &handle));
        if (handle == null) return error.Runtime;
        return .{ .handle = handle, .allocator = allocator };
    }

    /// Sets query-time parameters.
    pub fn setQueryTimeParams(self: *Index, params: Params) !void {
        try mapError(c.nmslib_set_query_time_params(self.handle, params.c_params));
    }

    /// Sets the thread pool size.
    pub fn setThreadPoolSize(self: *Index, size: usize) !void {
        try mapError(c.nmslib_set_thread_pool_size(self.handle, size));
    }

    /// Gets the thread pool size.
    pub fn getThreadPoolSize(self: *Index) usize {
        return c.nmslib_get_thread_pool_size(self.handle);
    }

    /// Gets the number of data points in the index.
    pub fn dataQty(self: *Index) usize {
        return c.nmslib_data_qty(self.handle);
    }

    /// Gets the space type of the index.
    pub fn getSpaceType(self: *Index) ![]const u8 {
        var space_type: [*c]const u8 = undefined;
        var space_type_len: usize = undefined;
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(@constCast(&self.allocator)),
        };
        try mapError(c.nmslib_get_space_type(self.handle, &space_type, &space_type_len, &c_alloc));
        defer c.nmslib_free_string(@constCast(space_type), &c_alloc);
        return try self.allocator.dupe(u8, space_type[0..space_type_len]);
    }

    /// Gets the method of the index.
    pub fn getMethod(self: *Index) ![]const u8 {
        var method: [*c]const u8 = undefined;
        var method_len: usize = undefined;
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(@constCast(&self.allocator)),
        };
        try mapError(c.nmslib_get_method(self.handle, &method, &method_len, &c_alloc));
        defer c.nmslib_free_string(@constCast(method), &c_alloc);
        return try self.allocator.dupe(u8, method[0..method_len]);
    }

    /// Gets the last error detail.
    pub fn getLastErrorDetail(self: *Index) !struct { code: Error, message: []const u8, file: []const u8, line: i32 } {
        var detail: c.nmslib_error_detail_t = undefined;
        const c_alloc = c.nmslib_allocator_t{
            .alloc = c_alloc_fn,
            .free = c_free_fn,
            .ctx = @ptrCast(@constCast(&self.allocator)),
        };
        try mapError(c.nmslib_get_last_error_detail(&detail, &c_alloc));
        defer c.nmslib_free_string(@constCast(detail.message), &c_alloc);
        defer c.nmslib_free_string(@constCast(detail.file), &c_alloc);
        return .{
            .code = mapError(detail.code) catch |e| e,
            .message = std.mem.span(detail.message),
            .file = std.mem.span(detail.file),
            .line = detail.line,
        };
    }

    /// Helper to get the data type of the index (inferred from space and method).
    fn getDataType(self: *Index) !DataType {
        const space_type = try self.getSpaceType();
        defer self.allocator.free(space_type);
        // Simplified heuristic; in practice, map specific space types to data types
        if (std.mem.eql(u8, space_type, "l2") or std.mem.eql(u8, space_type, "cosinesimil")) {
            return .DenseVector;
        } else if (std.mem.eql(u8, space_type, "cosinesimil_sparse")) {
            return .SparseVector;
        } else if (std.mem.eql(u8, space_type, "l2_sift")) {
            return .DenseUInt8Vector;
        } else if (std.mem.eql(u8, space_type, "leven")) {
            return .ObjectAsString;
        }
        return error.SpaceIncompatible;
    }
};
threadlocal var alloc_tracker: ?*AllocTracker = null;

const AllocTracker = struct {
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
        return .{ .map = map, .child_allocator = parent };
    }

    fn deinit(self: *AllocTracker) void {
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
    const alloc_ptr: *Allocator = @alignCast(@ptrCast(ctx.?));
    const allocator = alloc_ptr.*;
    if (alloc_tracker == null) {
        const tracker = allocator.create(AllocTracker) catch return null;
        tracker.* = AllocTracker.init(allocator) catch {
            allocator.destroy(tracker);
            return null;
        };
        alloc_tracker = tracker;
    }
    const mem_slice = allocator.alloc(u8, size) catch return null;
    const ptr = @as(*anyopaque, @ptrCast(mem_slice.ptr));
    alloc_tracker.?.map.put(ptr, size) catch {
        allocator.free(mem_slice);
        return null;
    };
    return ptr;
}

fn c_free_fn(ptr: ?*anyopaque, ctx: ?*anyopaque) callconv(.c) void {
    if (ptr == null or ctx == null) return;
    const alloc_ptr: *Allocator = @alignCast(@ptrCast(ctx.?));
    const allocator = alloc_ptr.*;
    if (alloc_tracker) |tracker| {
        if (tracker.map.get(ptr.?)) |len| {
            const slice_ptr = @as([*]u8, @ptrCast(ptr.?));
            const slice = slice_ptr[0..len];
            allocator.free(slice);
            _ = tracker.map.remove(ptr.?);
        }
    }
}

// Tests
test "Index dense vector workflow" {
    const allocator = testing.allocator;

    var params = try Params.init(allocator);
    defer params.deinit();
    try params.add("dim", .{ .Int = 4 });

    var index = try Index.init(allocator, "l2", params, "hnsw", .DenseVector, .Float);
    defer index.deinit();

    const data = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 0.0, 1.0, 0.0 },
    };
    const ids = [_]i32{ 10, 20, 30 };
    try index.addDenseBatch(&data, &ids);
    try index.create(null, false);

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
    try testing.expectApproxEqAbs(std.math.sqrt(2.0), dist, 0.0001);

    const data_point = try index.getDataPoint(0);
    defer data_point.deinit(allocator);
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, data_point.DenseVector);

    const borrowed = try index.borrowDataDense(0);
    defer borrowed.free_fn(@ptrCast(@constCast(borrowed.data.ptr)));
    try testing.expectEqualSlices(f32, &[_]f32{ 1.0, 0.0, 0.0, 0.0 }, borrowed.data);

    try index.save("test_index", true);
    try index.reset();
    try testing.expectEqual(0, index.dataQty());
    var loaded_index = try Index.load(allocator, "test_index", .DenseVector, .Float, true);
    defer loaded_index.deinit();
    try testing.expectEqual(3, loaded_index.dataQty());
}

test "Index sparse vector workflow" {
    const allocator = testing.allocator;

    var index = try Index.init(allocator, "cosinesimil_sparse", null, "hnsw", .SparseVector, .Float);
    defer index.deinit();

    const data = [_][]const SparseElem{
        &[_]SparseElem{ .{ .id = 0, .value = 1.0 }, .{ .id = 1, .value = 2.0 } },
        &[_]SparseElem{ .{ .id = 1, .value = 1.0 }, .{ .id = 2, .value = 3.0 } },
    };
    const ids = [_]i32{ 100, 200 };
    try index.addSparseBatch(&data, &ids);
    try index.create(null, false);

    const query = QueryPoint{ .SparseVector = &[_]SparseElem{ .{ .id = 0, .value = 1.0 } } };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();
    try testing.expectEqual(2, result.ids.len);

    const borrowed = try index.borrowDataSparse(0);
    defer borrowed.free_fn(@ptrCast(@constCast(borrowed.data.ptr)));
    try testing.expectEqualSlices(SparseElem, &[_]SparseElem{ .{ .id = 0, .value = 1.0 }, .{ .id = 1, .value = 2.0 } }, borrowed.data);
}

test "Index uint8 vector workflow" {
    const allocator = testing.allocator;

    var index = try Index.init(allocator, "l2", null, "hnsw", .DenseUInt8Vector, .Float);
    defer index.deinit();

    const data = [_][]const u8{ &[_]u8{ 255, 0, 0 }, &[_]u8{ 0, 255, 0 } };
    try index.addUInt8Batch(&data, null);
    try index.create(null, false);

    const query = QueryPoint{ .DenseUInt8Vector = &[_]u8{ 255, 0, 0 } };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();
    try testing.expectEqual(2, result.ids.len);

    const data_point = try index.getDataPoint(0);
    defer data_point.deinit(allocator);
    try testing.expectEqualSlices(u8, &[_]u8{ 255, 0, 0 }, data_point.DenseUInt8Vector);
}

test "Index string data workflow" {
    const allocator = testing.allocator;

    var index = try Index.init(allocator, "leven", null, "hnsw", .ObjectAsString, .Float);
    defer index.deinit();

    const data = [_][]const u8{ "hello", "world" };
    try index.addStringBatch(&data, null);
    try index.create(null, false);

    const query = QueryPoint{ .ObjectAsString = "hello" };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();
    try testing.expectEqual(2, result.ids.len);

    const str = try index.getDataPointString(0);
    defer allocator.free(str);
    try testing.expectEqualStrings("hello", str);
}