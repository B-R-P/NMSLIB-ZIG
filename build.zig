const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    // Force GNU ABI on Windows (uses bundled MinGW-w64, no SDK needed)
    const final_target = if (target.result.os.tag == .windows)
        b.resolveTargetQuery(std.Target.Query.parse(.{
            .arch_os_abi = "x86_64-windows-gnu",
        }) catch unreachable)
    else
        target;

    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    // NMSLIB source files (from setup.py, excluding space_sqfd.cc, dummy_app.cc, main.cc)
    const nmslib_sources = &[_][]const u8{
        "src/distcomp_bregman.cc",
        "src/distcomp_diverg.cc",
        "src/distcomp_edist.cc",
        "src/distcomp_js.cc",
        "src/distcomp_l2sqr_sift.cc",
        "src/distcomp_lp.cc",
        "src/distcomp_overlap.cc",
        "src/distcomp_rankcorr.cc",
        "src/distcomp_scalar.cc",
        "src/distcomp_sparse_scalar_fast.cc",
        "src/experimentconf.cc",
        "src/global.cc",
        "src/init.cc",
        "src/knnquery.cc",
        "src/logging.cc",
        "src/memory.cc",
        "src/params.cc",
        "src/params_cmdline.cc",
        "src/query.cc",
        "src/rangequery.cc",
        "src/searchoracle.cc",
        "src/space.cc",
        "src/utils.cc",
        "src/method/dummy.cc",
        "src/method/hnsw.cc",
        "src/method/hnsw_distfunc_opt.cc",
        "src/method/pivot_neighb_invindx.cc",
        "src/method/seqsearch.cc",
        "src/method/simple_inverted_index.cc",
        "src/method/small_world_rand.cc",
        "src/method/vptree.cc",
        "src/space/space_ab_diverg.cc",
        "src/space/space_bregman.cc",
        "src/space/space_dummy.cc",
        "src/space/space_js.cc",
        "src/space/space_l2sqr_sift.cc",
        "src/space/space_lp.cc",
        "src/space/space_renyi_diverg.cc",
        "src/space/space_scalar.cc",
        "src/space/space_sparse_dense_fusion.cc",
        "src/space/space_sparse_jaccard.cc",
        "src/space/space_sparse_lp.cc",
        "src/space/space_sparse_scalar_bin_fast.cc",
        "src/space/space_sparse_scalar_fast.cc",
        "src/space/space_sparse_vector.cc",
        "src/space/space_sparse_vector_inter.cc",
        "src/space/space_string.cc",
        "src/space/space_vector.cc",
        "src/space/space_word_embed.cc",
    };

    // Flags for library (performance-oriented)
    const cpp_flags_lib = &[_][]const u8{
        "-std=c++17",
        "-O3",
        "-flto",
        "-fno-semantic-interposition",
        "-fno-plt",
        "-fexceptions",
        "-fPIC",
        "-march=native",
        "-fopenmp",
        "-pthread",
        "-Wl,-z,now",
    };

    // Flags for tests (debug-oriented, no LTO to ensure static init)
    const cpp_flags_test = &[_][]const u8{
        "-std=c++17",
        "-O0",
        "-g",
        // prefer explicit DWARF (strong hint to Clang/LLVM when using MinGW)
        "-gdwarf-4",
        // keep frame pointers so backtraces are reliable
        "-fno-omit-frame-pointer",
        "-fexceptions",
        "-fPIC",
        "-march=native",
        "-fopenmp",
        "-pthread",
        "-fno-function-sections",
    };

    // Create the root module for the library
    const root_module = b.createModule(.{
        .root_source_file = b.path("lib.zig"),
        .target = final_target,
        .optimize = optimize,
    });
    b.modules.put("nmslib", root_module) catch unreachable;

    // Create a static library for NMSLIB wrapper + sources
    const lib = b.addLibrary(.{
        .name = "nmslib",
        .linkage = .static,
        .root_module = root_module,
    });

    // Add NMSLIB sources + wrapper
    lib.addCSourceFiles(.{
        .files = nmslib_sources ++ &[_][]const u8{"nmslib_c.cpp"},
        .flags = cpp_flags_lib,
    });

    // Add include directories (from setup.py and existing build.zig)
    lib.addIncludePath(b.path("."));
    lib.addIncludePath(b.path("include"));
    lib.addIncludePath(b.path("include/factory"));
    lib.addIncludePath(b.path("include/method"));
    lib.addIncludePath(b.path("include/space"));

    lib.linkLibCpp();
    // Install the header file
    lib.installHeader(b.path("nmslib_c.h"), "nmslib_c.h");

    // Install the library
    b.installArtifact(lib);


    // Add a test step
    const test_module = b.createModule(.{
        .root_source_file = b.path("lib.zig"),
        .target = final_target,
        .optimize = .Debug,
    });
    const tests = b.addTest(.{
        .root_module = test_module,
    });

    // Configure test includes and sources (same as lib, but with test flags)
    tests.addCSourceFiles(.{
        .files = nmslib_sources ++ &[_][]const u8{"nmslib_c.cpp"},
        .flags = cpp_flags_test,
    });
    tests.addIncludePath(b.path("."));
    tests.addIncludePath(b.path("include"));
    tests.addIncludePath(b.path("include/factory"));
    tests.addIncludePath(b.path("include/method"));
    tests.addIncludePath(b.path("include/space"));
    tests.linkLibCpp();

    // Create a run step for tests
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_tests.step);
}
