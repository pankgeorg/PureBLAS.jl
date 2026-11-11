# julia --project=wasm wasm/build.jl [OUT.wasm]: the prebuilt module of wasm/linalg.jl's operations (see WASM.md),
# optimized by wasm-opt -O3 (WasmCodegen's default; it takes most of the build's time). ENV: OPS=lu,qr,... and
# TYPES=f64,c64 build a subset; ROUTE=fill leaves to WasmCodegen's runtime the BLAS/LAPACK functions it has;
# RELAXED_SIMD=1 compiles vector muladd to relaxed SIMD's relaxed_madd (fused where the hardware has FMA).
using WasmCodegen, PureBLAS, WasmCompat, SIMD, LinearAlgebra
include(joinpath(@__DIR__, "linalg.jl"))
out = get(ARGS, 1, joinpath(@__DIR__, "pureblas.wasm"))
ops = haskey(ENV, "OPS") ? Symbol.(split(ENV["OPS"], ',')) : first.(PureBLASWasm.OPS)
types = haskey(ENV, "TYPES") ? String.(split(ENV["TYPES"], ',')) : ["f64", "c64"]
config = WasmCodegen.Config(; trim = :safe, on_unsupported = :error, progress = true,
                            relaxed_simd = get(ENV, "RELAXED_SIMD", "0") == "1",
                            foreign_functions = PureBLAS.wasm_foreign_functions(; mode = Symbol(get(ENV, "ROUTE", "all"))))
config.optimize === nothing && error("modules are built optimized (wasm-opt -O3) only")
t = time()
c = WasmCodegen.compile(PureBLASWasm.entries(; ops, types); config)
write(out, c.bytes)
println("wrote ", out, ": ", length(c.bytes), " bytes in ", round(Int, time() - t), " s")
