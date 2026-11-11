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

This example, as written, builds in about 9 minutes into a 2.8 MB module, and prints 4.799868917380728, the same
number as native Julia.

What each piece does:
- **`PureBLASWasmCodegenExt`**, this package's extension, loaded with WasmCodegen
  ([`ext/PureBLASWasmCodegenExt.jl`](ext/PureBLASWasmCodegenExt.jl)):
  - it runs PureBLAS on one thread, without prefetch or `madvise` hints, and without the register-sized
    `_trsv!` ladder;
  - it uses double-shift Hessenberg QR at every size, which is 1.19–1.23× faster than multishift in Wasm at
    n = 256–512;
  - it defines `PureBLAS.wasm_foreign_functions(; mode = :all)`.
- **`WasmCompat`'s `WasmCompatSIMDExt`** compiles SIMD.jl's 16-byte float vectors (`Vec{2,Float64}`,
  `Vec{4,Float32}`) to WebAssembly SIMD128 and runs other vectors lane by lane. **`WasmCompatBaseExt`** makes
  `Base.generating_output()` false.
- **`wasm_foreign_functions(mode = :fill)`** routes to PureBLAS only the functions WasmCodegen's own runtime BLAS
  lacks. That runtime has real `gemm`, `getrf`, `potrf`, `geqrf`, `gesdd` and `syevr`. A program that uses only
  those then stays small (an `lu` is about 12 KB) and pays for PureBLAS only for the rest: complex, `eigen`,
  `exp`, `schur`, ...

[`wasm/build.jl`](wasm/build.jl) builds the release module the same way, and [`wasm/Project.toml`](wasm/Project.toml)
pins the versions it was built with: WasmCodegen 98cad1f (main), WasmBuilder 0.1.1, IRStructurizer 0c96648 and
WasmCompat c57da6c (main).

## Speed, size and limits

Measured on an aarch64 host (Neoverse-N1) with one thread. The modules are built with `-O3`, the times are ms per
call (the median over 3 rounds of 3 interleaved trials), and the inputs are the same well-conditioned matrices for
every column. "Old" is the previous release, `wasm-v2026-11-07`, whose WasmCodegen had neither typed pointer loads
nor SIMD128.

| n = 256 | Wasm, old (Node 24) | Wasm, this release (Node 24) | speedup | native PureBLAS | native OpenBLAS | Wasm / native PureBLAS |
|---|---:|---:|---:|---:|---:|---:|
| `gemm_f64` | 374 | 29.6 | 12.7× | 3.18 | 2.75 | 9× |
| `solve_f64` | 185 | 15.2 | 12.2× | 1.40 | 1.07 | 11× |
| `lu_f64` | 180 | 15.3 | 11.7× | 1.47 | 1.42 | 10× |
| `inv_f64` | 372 | 53.8 | 6.9× | 3.90 | 3.58 | 14× |
| `cholesky_f64` | 58 | 14.3 | 4.0× | 0.74 | 0.92 | 19× |
| `qr_f64` | 917 | 68.4 | 13.4× | 8.80 | 4.86 | 8× |
| `svd_f64` | 2,054 | 378 | 5.4× | 27.8 | 22.7 | 14× |
| `svdvals_f64` | 466 | 241 | 1.9× | 7.21 | 11.2 | 33× |
| `eigen_f64` | 7,676 | 1,391 | 5.5× | 71.2 | 46.7 | 20× |
| `eigen_hermitian_f64` | 1,236 | 133 | 9.3× | 12.4 | 11.6 | 11× |
| `exp_f64` | 2,035 | 183 | 11.1× | 17.6 | 12.0 | 10× |
| `gemm_c64` | 1,177 | 89.4 | 13.2× | 9.40 | 8.66 | 10× |
| `solve_c64` | 586 | 50.7 | 11.6× | 4.29 | 3.62 | 12× |
| `lu_c64` | 563 | 49.5 | 11.4× | 4.32 | 3.74 | 11× |
| `inv_c64` | 1,690 | 142 | 11.9× | 12.0 | 10.5 | 12× |
| `cholesky_c64` | 257 | 56.8 | 4.5× | 2.17 | 2.05 | 26× |
| `qr_c64` | 4,106 | 312 | 13.2× | 21.9 | 19.1 | 14× |
| `svd_c64` | 7,218 | 648 | 11.1× | 57.8 | 61.5 | 11× |
| `svdvals_c64` | 3,319 | 223 | 14.9× | 17.8 | 32.2 | 12× |
| `eigen_c64` | 22,973 | 2,865 | 8.0× | 434 | 157 | 7× |
| `eigen_hermitian_c64` | 3,353 | 271 | 12.4× | 28.9 | 34.5 | 9× |
| `exp_c64` | 7,019 | 538 | 13.1× | 58.7 | 50.6 | 9× |

Across all 46 cases (n = 64 and 256 for every operation, and n = 512 for gemm), this release is 11.7× faster than the
previous one (the median; the range is 1.9–16×), and 11× slower than native PureBLAS (the range is 6–33×). Bun 1.4
gives the same picture (11.2× over the previous release). gemm reaches 1.18 GF/s (f64) and 1.54 GF/s (c64) at
n = 512 on Node. Natively it reaches 10.5 and 13.4 GF/s with PureBLAS, and 14.2 and 15.5 GF/s with OpenBLAS.

- **Where the remaining gap comes from:** the vector width is not the cause. This host's NEON is 128-bit, the same
  width as SIMD128. Without relaxed SIMD, a vector `muladd` is a multiply and an add, where native code uses one
  fused instruction. Every load and store also goes through a typed view of linear memory, and the engine
  allocates registers. The operations that gained least spend much of their time in scalar recurrences that
  SIMD128 does not reach, such as the bidiagonal QR in `svdvals` and the Hessenberg QR sweeps in real `eigen`.
- **`relaxed_simd`:** `RELAXED_SIMD=1` builds the module with relaxed SIMD's `relaxed_madd`, which fuses where the
  engine and hardware allow. It was measured and is not the release build. It was 2% faster on Node and 3.5%
  slower on Bun. Its results depend on whether the engine fuses: Hermitian eigenvectors with clustered eigenvalues
  differed from the plain build by up to 3e-4 on Node and 7e-5 on Bun, still within `check.mjs`. The plain build
  gives the previous release's results bit for bit, on every engine.
- **Size:** the release module is 13.9 MB with all 22 operations (33.3 MB before `wasm-opt`). The previous release
  was 16.8 MB. No function is close to V8's limits: the largest body is 2.1 MB and the most locals 616.
- **Build time:** `wasm-opt -O3` dominates. The release module takes 7.7 minutes to compile and 36.5 minutes of
  `wasm-opt` on two threads (6.5 GB peak).
- **Not covered:** real `schur` (WasmCodegen does not compile `Schur(gees!(...)...)`, a splat of a union of
  tuples; complex `schur` works), threads, and Float32 `svd` with vectors (as natively in PureBLAS).

## What the extension still changes, and why

Each of these was rechecked against WasmCodegen 98cad1f:

| Overlay | Why it stays |
|---|---|
| `_trsv_reg_n!` → `_trsv_simd!` | Without it, the register ladder inlines `_trsv!` bodies of up to 29.7 MB and 382,000 locals. `wasm-opt -O3` cannot finish them (more than 25 minutes and 8.8 GB for `cholesky` + `solve` alone), and V8 refuses such bodies (7.65 MB / 50,000 locals) |
| `_dlaqr0!` → `_dlahqr!` | Speed: multishift QR is 1.23× slower at n = 256 and 1.19× slower at n = 512 (real `eigen`, Node), its module is 25% larger, and it takes 50% longer to build |
| `wasm_foreign_functions` | WasmCodegen's own runtime BLAS has no `zgetrf`, `zgesdd`, `zheevr`, `dgeevx`, `zlacpy`, ...: without the route, complex `lu`, `svd` and `eigen` and real `eigen` do not compile |
| one-thread gates, `_gemm_threaded!` | There is no thread pool in the module |
| `_prefetch`, `madvise` | WasmCodegen has no `llvm.prefetch`, and `madvise` is in the `mmap` family it refuses |

## What is still missing

| Missing | Where | Effect today |
|---|---|---|
| Union-split of a splat (`Schur(t...)` with `t` a union of tuples) | WasmCodegen | real `schur` does not compile |
| Reuse of locals, smaller input to wasm-opt | WasmCodegen | 36 min of `wasm-opt` for the release module |
| `jl_generating_output` → 0 | WasmCodegen (or `WasmCompatBaseExt` today) | — |
| Arena pointers by pointer arithmetic | PureBLAS (this fork, `src/arena.jl`) | without it, every arena scratch access traps in Wasm |
