# PureBLAS in WebAssembly

PureBLAS compiles to WebAssembly (Wasm GC) with [WasmCodegen.jl](https://github.com/gbaraldi/WasmCodegen.jl), the
compiler of Julia programs to Wasm. Unmodified LinearAlgebra code (`A * B`, `A \ b`, `lu`, `qr`, `cholesky`, `svd`,
`eigen`, `exp`, ...) then runs in a browser or in Node, in Float64 and ComplexF64, with PureBLAS answering its BLAS
and LAPACK calls. The results match native Julia to rounding.

Two ways to use it:
- **From JavaScript:** load the prebuilt module of the release and call its exports (*From JavaScript*).
- **From Julia:** compile a program of your own with WasmCodegen and this package's extension (*From Julia*).

## From JavaScript: the prebuilt module

The release `wasm-v<date>` has `pureblas.wasm` (optimized with `wasm-opt -O3`), `julia.mjs` (WasmCodegen's
JavaScript host), `check.mjs` and `reference.json`. The module exports these operations, each in Float64 (`_f64`)
and ComplexF64 (`_c64`): `gemm`, `solve`, `lu`, `inv`, `cholesky`, `qr`, `svd`, `svdvals`, `eigen`,
`eigen_hermitian` and `exp` (layouts: the table in [`wasm/linalg.jl`](wasm/linalg.jl)).

A host passes numbers only:
1. it fills the input buffer (`input_resize`, `input_set`);
2. it calls an operation;
3. it reads the output buffer (`output_length`, `output_get`).

Matrices are column-major, and a complex number is two Float64s, real then imaginary. Integers are 64-bit, so they
are `BigInt`s in JavaScript.

```js
// node example.mjs  (Node 24; with pureblas.wasm and julia.mjs from the release next to it)
import fs from 'node:fs';
import { instantiate } from './julia.mjs';
const { instance } = await instantiate(fs.readFileSync('pureblas.wasm'));
const e = instance.exports;

// A = [4 1; 2 3] (column-major), b = [1, 2]: x = A \ b
const input = [4, 2, 1, 3, 1, 2];
e.input_resize(BigInt(input.length));
input.forEach((v, i) => e.input_set(BigInt(i + 1), v));
const status = e.solve_f64(2n, 1n);          // 0: done; > 0: LAPACK's info (singular); -1: wrong input length
const x = Array.from({ length: Number(e.output_length()) }, (_, i) => e.output_get(BigInt(i + 1)));
console.log(status, x);                       // 0n [ 0.09999999999999998, 0.6000000000000001 ]
```

`node check.mjs` runs the release's reference cases (inputs and results from native Julia,
[`wasm/reference.jl`](wasm/reference.jl)) and compares.

An error other than a status (an SVD that does not converge, say) is thrown as a WebAssembly exception.

## From Julia: compile your own program

Your program calls LinearAlgebra as usual. The build script loads WasmCodegen, PureBLAS and WasmCompat, and routes
LinearAlgebra's BLAS/LAPACK calls to PureBLAS with `PureBLAS.wasm_foreign_functions()`:

```julia
# build.jl: julia --project=. build.jl, in an environment with WasmCodegen, WasmCompat and this PureBLAS
using WasmCodegen, PureBLAS, WasmCompat, SIMD, LinearAlgebra

function solve(n::Int)                      # your program: plain LinearAlgebra
    A = [1.0 / (i + j - 1) + (i == j) for i in 1:n, j in 1:n]
    x = A \ ones(n)
    return sum(x)
end

config = WasmCodegen.Config(trim = :safe, foreign_functions = PureBLAS.wasm_foreign_functions())
c = WasmCodegen.compile([(solve, (Int,))]; config)   # optimized with wasm-opt -O3 (WasmCodegen's default)
write("solve.wasm", c.bytes)
cp(WasmCodegen.JS_HOST, "julia.mjs"; force = true)
```

```js
const { instance } = await instantiate(fs.readFileSync('solve.wasm'));
console.log(instance.exports.solve(10n));
```

This example, as written, builds in about 10 minutes into a 4.1 MB module, and prints 4.799868917380728, the same
number as native Julia.

What each piece does:
- **`PureBLASWasmCodegenExt`**, this package's extension, loaded with WasmCodegen
  ([`ext/PureBLASWasmCodegenExt.jl`](ext/PureBLASWasmCodegenExt.jl)):
  - it runs PureBLAS on one thread, without prefetch or `madvise` hints, and without the register-sized
    `_trsv!` ladder;
  - it uses double-shift Hessenberg QR at every size, which is 1.25–1.45× faster than multishift in Wasm at
    n = 256–512;
  - it defines `PureBLAS.wasm_foreign_functions(; mode = :all)`.
- **`WasmCompat`'s `WasmCompatSIMDExt`** runs SIMD.jl's vector intrinsics lane by lane, and **`WasmCompatBaseExt`**
  makes `Base.generating_output()` false.
- **`wasm_foreign_functions(mode = :fill)`** routes to PureBLAS only the functions WasmCodegen's own runtime BLAS
  lacks. That runtime has real `gemm`, `getrf`, `potrf`, `geqrf`, `gesdd` and `syevr`. A program that uses only
  those then stays small (an `lu` is about 12 KB) and pays for PureBLAS only for the rest: complex, `eigen`,
  `exp`, `schur`, ...

[`wasm/build.jl`](wasm/build.jl) builds the release module the same way, and [`wasm/Project.toml`](wasm/Project.toml)
pins the versions it was built with.

## Limits

Measured on an aarch64 host, Node 24 (V8), one thread, `-O3` modules:

| | Native OpenBLAS | Native PureBLAS | Wasm PureBLAS |
|---|---:|---:|---:|
| gemm f64, n = 256 (GF/s) | 18.2 | 11.7 | 0.10 |
| gemm f32, n = 256 (GF/s) | 38.7 | 23.8 | 0.28 |
| gemm c64, n = 256 (GF/s) | 16.0 | 14.5 | 0.13 |

Through the release module's `gemm_f64`/`gemm_c64`, including the copy of the operands from the input buffer: 0.09 and
0.11 GF/s at n = 256.

- **Speed:** about 100× below native. WebAssembly here has no SIMD (SIMD.jl's vectors run lane by lane), and
  WasmCodegen reads and writes memory through a pointer without a static origin one byte at a time. With typed
  pointer loads (a WasmCodegen change, not yet merged) gemm runs 5× faster (0.58 GF/s at n = 256).
- **Algorithms still win where the runtime's are simpler:** against WasmCodegen's own runtime BLAS, PureBLAS is
  1.5× faster on `svd` and up to 13× faster on symmetric `eigen` (it reduces to tridiagonal and bidiagonal form;
  the runtime uses Jacobi rotations). The runtime is 2–4× faster on `lu`, `cholesky` and `qr`.
- **Size:** the release module is 16.8 MB (all 22 operations; 39.4 MB before `wasm-opt`). A program of your own that
  reaches PureBLAS for one operation is 2–10 MB after `-O3`.
- **Build time:** `wasm-opt -O3` dominates: 17–55 minutes for a PureBLAS module on two threads (up to 9 GB of
  memory), on top of 3–9 minutes of compilation. The release module took 9 minutes to compile and 30 minutes of
  `wasm-opt` (8.8 GB peak).
- **Not covered:** real `schur` (WasmCodegen does not compile `Schur(gees!(...)...)`, a splat of a union of
  tuples; complex `schur` works), threads, and Float32 `svd` with vectors (as natively in PureBLAS).

## What is still missing

| Missing | Where | Effect today |
|---|---|---|
| Typed 4/8-byte loads through pointers without a static origin | WasmCodegen | 5× slower, about a third larger modules |
| SIMD128 lowering of SIMD.jl's 16-byte vectors | WasmCodegen | lanes run one by one (`WasmCompatSIMDExt`) |
| Union-split of a splat (`Schur(t...)` with `t` a union of tuples) | WasmCodegen | real `schur` does not compile |
| Reuse of locals, smaller input to wasm-opt | WasmCodegen | 17–55 min of `wasm-opt` per module |
| `jl_generating_output` → 0 | WasmCodegen (or `WasmCompatBaseExt` today) | — |
| Arena pointers by pointer arithmetic | PureBLAS (this fork, `src/arena.jl`) | without it, every arena scratch access traps in Wasm |
