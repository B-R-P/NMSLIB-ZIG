# NMSLIB-ZIG

A Zig binding for [NMSLIB](https://github.com/nmslib/nmslib) (Non-Metric Space Library), providing efficient similarity search and k-NN graph construction for high-dimensional data.

## Overview

NMSLIB-ZIG is a Zig wrapper around NMSLIB, a popular library for approximate nearest neighbor search. This binding exposes NMSLIB's functionality through a clean Zig API with proper memory management and error handling. It uses a custom C wrapper (`nmslib_c.cpp`) to bridge between Zig and the C++ NMSLIB library.

### Key Features

- **Multiple Data Types**: Support for dense vectors, sparse vectors, uint8 vectors, and string data
- **Various Distance Metrics**: L2, cosine similarity, sparse metrics, and more
- **Multiple Indexing Methods**: HNSW, VP-Tree, Sequential Search, and others
- **Efficient Memory Management**: Custom allocator integration with Zig's allocator system
- **Batch Operations**: Efficient batch insertion and querying
- **Save/Load Indexes**: Persist indexes to disk for reuse
- **Thread Pool Support**: Configurable thread pool for parallel queries
- **Comprehensive Error Handling**: Proper error mapping from C to Zig errors

## Architecture

The project consists of three main components:

1. **lib.zig**: The main Zig API that provides idiomatic Zig bindings
2. **nmslib_c.cpp**: A C wrapper around the C++ NMSLIB library
3. **nmslib_c.h**: C header defining the interface between Zig and C++

The architecture uses:
- `@cImport` to import the C interface into Zig
- Custom allocator callbacks for memory management across language boundaries
- Thread-local error tracking for detailed error reporting

## Installation

### Prerequisites

- Zig (master or recent version)
- C++ compiler with C++17 support
- OpenMP support (for parallel operations)

### Building

```bash
zig build
```

### Running Tests

```bash
zig build test
```

## Usage

### Basic Example: Dense Vector Search

```zig
const std = @import("std");
const nmslib = @import("lib.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create space parameters (specify dimensionality)
    var params = try nmslib.Params.init(allocator);
    defer params.deinit();
    try params.add("dim", .{ .Int = 128 });

    // Create index with L2 distance and HNSW method
    var index = try nmslib.Index.init(
        allocator,
        "l2",           // Space type
        params,         // Space parameters
        "hnsw",         // Indexing method
        .DenseVector,   // Data type
        .Float          // Distance type
    );
    defer index.deinit();

    // Add data points
    const data = [_][]const f32{
        &[_]f32{ 1.0, 0.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 1.0, 0.0, 0.0 },
        &[_]f32{ 0.0, 0.0, 1.0, 0.0 },
    };
    const ids = [_]i32{ 1, 2, 3 };
    try index.addDenseBatch(&data, &ids);

    // Build the index
    try index.create(null, false);

    // Perform k-NN query
    const query = nmslib.QueryPoint{ .DenseVector = &[_]f32{ 1.0, 0.1, 0.0, 0.0 } };
    const result = try index.knnQuery(query, 2);
    defer result.deinit();

    std.debug.print("Found {} neighbors\n", .{result.ids.len});
    for (result.ids, result.distances) |id, dist| {
        std.debug.print("ID: {}, Distance: {d:.4}\n", .{ id, dist });
    }
}
```

### Sparse Vector Example

```zig
const nmslib = @import("lib.zig");

var index = try nmslib.Index.init(
    allocator,
    "cosinesimil_sparse",
    null,
    "hnsw",
    .SparseVector,
    .Float
);
defer index.deinit();

// Sparse vectors as arrays of {id, value} pairs
const data = [_][]const nmslib.SparseElem{
    &[_]nmslib.SparseElem{
        .{ .id = 0, .value = 1.0 },
        .{ .id = 5, .value = 2.0 },
    },
    &[_]nmslib.SparseElem{
        .{ .id = 1, .value = 1.0 },
        .{ .id = 10, .value = 3.0 },
    },
};

try index.addSparseBatch(&data, null);
try index.create(null, false);

const query = nmslib.QueryPoint{
    .SparseVector = &[_]nmslib.SparseElem{
        .{ .id = 0, .value = 1.0 }
    }
};
const result = try index.knnQuery(query, 5);
defer result.deinit();
```

### String Data Example

```zig
var index = try nmslib.Index.init(
    allocator,
    "leven",        // Levenshtein distance
    null,
    "hnsw",
    .ObjectAsString,
    .Float
);
defer index.deinit();

const data = [_][]const u8{ "hello", "world", "test" };
try index.addStringBatch(&data, null);
try index.create(null, false);

const query = nmslib.QueryPoint{ .ObjectAsString = "hello" };
const result = try index.knnQuery(query, 2);
defer result.deinit();
```

### Saving and Loading Indexes

```zig
// Save index
try index.save("my_index.bin", true); // true = save data

// Load index
var loaded_index = try nmslib.Index.load(
    allocator,
    "my_index.bin",
    .DenseVector,
    .Float,
    true  // true = load data
);
defer loaded_index.deinit();
```

### Batch Queries

```zig
const queries = [_][]const f32{
    &[_]f32{ 1.0, 0.0, 0.0 },
    &[_]f32{ 0.0, 1.0, 0.0 },
};

const batch_result = try index.knnQueryBatch(&queries, 10, null);
defer batch_result.deinit();

for (batch_result.results, 0..) |result, i| {
    std.debug.print("Query {}: Found {} neighbors\n", .{ i, result.ids.len });
}
```

### Range Queries

```zig
// Find all points within a distance radius
const query = &[_]f32{ 1.0, 0.0, 0.0 };
const result = try index.rangeQuery(query, 0.5); // radius = 0.5
defer result.deinit();
```

## API Reference

### Core Types

#### `Index`
The main index structure for similarity search.

**Methods:**
- `init()` - Create a new index
- `deinit()` - Free the index
- `addDenseBatch()` - Add dense vector batch
- `addSparseBatch()` - Add sparse vector batch
- `addUInt8Batch()` - Add uint8 vector batch
- `addStringBatch()` - Add string data batch
- `create()` - Build the index structure
- `knnQuery()` - Perform k-nearest neighbor query
- `knnQueryBatch()` - Batch k-NN queries
- `rangeQuery()` - Find all neighbors within radius
- `save()` - Save index to disk
- `load()` - Load index from disk
- `getDistance()` - Get distance between two points
- `getDataPoint()` - Retrieve a data point
- `dataQty()` - Get number of data points

#### `Params`
Parameter management for spaces and indexes.

**Methods:**
- `init()` - Create new parameter set
- `deinit()` - Free parameter set
- `add()` - Add a parameter
- `fromSlice()` - Create from key-value pairs

#### `DataType`
Enum of supported data types:
- `DenseVector` - Dense floating-point vectors
- `SparseVector` - Sparse vectors (id, value pairs)
- `DenseUInt8Vector` - Dense uint8 vectors
- `ObjectAsString` - String objects

#### `DistType`
Enum of distance computation types:
- `Float` - Floating-point distances
- `Int` - Integer distances

#### `QueryResult`
Result of a k-NN or range query containing:
- `ids` - Array of result IDs
- `distances` - Array of distances
- `deinit()` - Free the result

### Supported Space Types

- `l2` - Euclidean (L2) distance
- `cosinesimil` - Cosine similarity
- `cosinesimil_sparse` - Cosine similarity for sparse vectors
- `l2_sift` - L2 distance optimized for SIFT descriptors
- `leven` - Levenshtein distance (edit distance)
- And many more (see NMSLIB documentation)

### Supported Index Methods

- `hnsw` - Hierarchical Navigable Small World graphs (recommended)
- `sw-graph` - Small World graph
- `vptree` - Vantage Point Tree
- `seq_search` - Sequential search (brute force)

## Error Handling

The library provides comprehensive error handling through Zig's error system:

```zig
pub const Error = error{
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
```

All operations return `!T` (error union types), allowing proper error propagation with `try`.

## Thread Safety

- Index creation and data insertion should be done from a single thread
- Queries can be performed from multiple threads after index creation
- Use `setThreadPoolSize()` to configure parallel query execution

## Performance Considerations

1. **Index Method Selection**: HNSW is recommended for most use cases (good balance of speed and accuracy)
2. **Parameters**: Tune space and index parameters for your specific use case
3. **Batch Operations**: Use batch methods for better performance when inserting multiple data points
4. **Memory**: Dense vectors use more memory than sparse vectors
5. **Build vs Query Time**: More complex index methods (like HNSW) take longer to build but provide faster queries

## Benchmarking

Run the included tests to verify functionality:

```bash
zig build test
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows Zig style conventions
- All tests pass
- New features include tests
- Documentation is updated

## License

This project wraps NMSLIB, which is licensed under the Apache License 2.0. Please refer to the NMSLIB project for its license terms.

## Credits

- [NMSLIB](https://github.com/nmslib/nmslib) - The underlying similarity search library
- Built with [Zig](https://ziglang.org/)

## References

- [NMSLIB GitHub Repository](https://github.com/nmslib/nmslib)
- [NMSLIB Documentation](https://github.com/nmslib/nmslib/blob/master/manual/README.md)
- [Zig Documentation](https://ziglang.org/documentation/master/)
