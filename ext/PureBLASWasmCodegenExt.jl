"""
    PureBLASWasmCodegenExt

PureBLAS in WebAssembly, compiled by [WasmCodegen](https://github.com/gbaraldi/WasmCodegen.jl).
Loaded with WasmCodegen; see `WASM.md`. Its methods are overlays in WasmCodegen's method table
(`WasmCodegen.Runtime.WASM_METHODS`): they replace PureBLAS's own only when WasmCodegen compiles a
program, so native PureBLAS is unchanged.

- **Hints a module has no use for:** `_prefetch` (an `llvm.prefetch` `llvmcall`) and `madvise`
  (`_advise_huge!`) do nothing.
- **One thread:** every `_*_workers` gate returns 1, so the threaded branches fold away and the
  pool (`Threads.@spawn`, `Threads.Event`, a try/catch) is never compiled. `_gemm_threaded!`
  throws, for a gate that inference cannot bound.
- **`_trsv!` without the register ladder:** `_TRSV_REG_MAX` is the build machine's scalar FP
  register count (16 on x86-64, 32 on AArch64), and the ladder inlines a fully unrolled kernel for
  every n up to it, in eight shapes. A WebAssembly engine has no such register file, and on an
  AArch64 build host the ladder makes `_trsv!` too large to optimize or load. In Wasm,
  `_trsv_reg_n!` takes `_trsv_simd!`, the path the ladder falls through to above its bound.
- **Double-shift Hessenberg QR at every size (a speed choice, not a workaround):** `_dlaqr0!`
  takes `_dlahqr!`, as PureBLAS does natively for n ≤ 200. In Wasm, multishift QR (`_dlaqr5!`)
  is 1.23× slower at n = 256 and 1.19× slower at n = 512 (real `eigen`, Node 24, WasmCodegen
  with SIMD128), and its module is 25% larger: its advantage is level-3 BLAS, which is the slow
  part in Wasm.

[`PureBLAS.wasm_foreign_functions`](@ref) routes unmodified LinearAlgebra's BLAS and LAPACK calls
to PureBLAS, through `WasmCodegen.Config(foreign_functions)`.

The SIMD.jl vector intrinsics PureBLAS's kernels use are not here: they are generic, and
`WasmCompatSIMDExt` (WasmCompat.jl) compiles 16-byte float vectors to WebAssembly SIMD128
(WasmCodegen's `SIMD128`) and runs the others lane by lane.
"""
module PureBLASWasmCodegenExt

using PureBLAS, WasmCodegen
using PureBLAS: PtrMatrix, PtrVector
import LinearAlgebra
import LinearAlgebra.libblastrampoline_jll: libblastrampoline
import Base.Experimental: @overlay
const MT = WasmCodegen.Runtime.WASM_METHODS

## Hints
@overlay MT @inline PureBLAS._prefetch(p::Ptr{T}) where {T} = nothing
@overlay MT @inline PureBLAS._advise_huge!(A::AbstractMatrix{T}) where {T} = nothing
@overlay MT @inline PureBLAS._advise_huge_arr!(A::Array{T}) where {T} = nothing

## One thread
@overlay MT @inline PureBLAS.get_num_threads() = 1
@overlay MT @inline PureBLAS._l1_workers(bytes::Int, n::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._elem_workers(m::Int, n::Int, nw::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._gemm_workers(m::Int, n::Int, k::Int) = 1
@overlay MT @inline PureBLAS._syrk_workers(n::Int, k::Int) = 1
@overlay MT @inline PureBLAS._ger_workers(m::Int, n::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._gemvn_workers(m::Int, n::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._gemvt_workers(m::Int, n::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._symv_workers(n::Int, nblk::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._gbmvn_workers(m::Int, n::Int, kl::Int, ku::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._trsm_workers(k::Int, n::Int) = 1
@overlay MT @inline PureBLAS._trsm_workers_r(k::Int, m::Int, ::Type{T}) where {T} = 1
@overlay MT @inline PureBLAS._trmm_workers_l(k::Int, n::Int) = 1
@overlay MT @inline PureBLAS._trmm_workers_r(k::Int, m::Int, ::Type{T}) where {T} = 1
@noinline _no_threads() = error("PureBLAS: no thread pool in WebAssembly")
@overlay MT PureBLAS._gemm_threaded!(C::PtrMatrix{T}, args...) where {T} = _no_threads()

## `_trsv!` without the register ladder
@overlay MT @inline PureBLAS._trsv_reg_n!(up::Bool, tr::Bool, unit::Bool, n::Int, A, x) =
    PureBLAS._trsv_simd!(up, tr, unit, n, A, x)

## Double-shift Hessenberg QR at every size (a speed choice; see the module's documentation)
@overlay MT function PureBLAS._dlaqr0!(wantt::Bool, wantz::Bool, H::AbstractMatrix{R}, ilo::Int,
                                       ihi::Int, wr::AbstractVector{R}, wi::AbstractVector{R},
                                       iloz::Int, ihiz::Int, Z) where {R <: Real}
    size(H, 1) == 0 && return 0
    return PureBLAS._dlahqr!(wantt, wantz, H, ilo, ihi, wr, wi, iloz, ihiz, Z)
end

## LinearAlgebra's BLAS/LAPACK calls → PureBLAS's C-ABI methods (`Config(foreign_functions)`)

# LinearAlgebra's argument types are not always the ones PureBLAS's methods declare (`zheevr`
# passes `vl`, `vu`, `abstol` as `Ref{ComplexF64}` where Fortran reads a real): each pointer is
# cast to the declared type, which keeps its owner in Wasm.
@inline _adapt(::Type{T}, x::T) where {T} = x
@inline _adapt(::Type{Ptr{T}}, p::Ptr{T}) where {T} = p
@inline _adapt(::Type{Ptr{T}}, p::Ptr) where {T} = Ptr{T}(p)
@inline _adapt(::Type{T}, x) where {T} = convert(T, x)

# `?laic1` (used by `\` through `QRPivoted`): PureBLAS's `_laic1` kernel, which has no C-ABI entry
function _laic1(job::Ptr{Int64}, j::Ptr{Int64}, x::Ptr{T}, sest::Ptr{R}, w::Ptr{T},
                gamma::Ptr{T}, sestpr::Ptr{R}, s::Ptr{T}, c::Ptr{T}) where {T, R}
    J = Int(unsafe_load(j))
    sp, ss, cc = PureBLAS._laic1(Int(unsafe_load(job)), PtrVector(x, J), unsafe_load(sest),
                                 PtrVector(w, J), unsafe_load(gamma))
    unsafe_store!(sestpr, R(sp)); unsafe_store!(s, T(ss)); unsafe_store!(c, T(cc))
    return nothing
end

# (name => adapter) for each Fortran/CBLAS symbol of PureBLAS's C-ABI layer
const ROUTES = Pair{Symbol,Any}[]
for name in names(PureBLAS; all = true)
    s = string(name)
    (endswith(s, "_64_") || startswith(s, "cblas_")) && isdefined(PureBLAS, name) || continue
    f = getfield(PureBLAS, name)
    (f isa Function && length(methods(f)) == 1) || continue
    m = only(methods(f)); sig = m.sig.parameters[2:end]; as = [Symbol(:a, i) for i in 1:m.nargs-1]
    w = Symbol("_route_", name)
    @eval @inline $w($(as...)) = $f($((:(_adapt($(sig[i]), $(as[i]))) for i in eachindex(as))...))
    push!(ROUTES, name => getfield(@__MODULE__, w))
end
for name in (:dlaic1_64_, :slaic1_64_, :zlaic1_64_, :claic1_64_)
    isdefined(PureBLAS, name) || push!(ROUTES, name => _laic1)
end

# the BLAS/LAPACK functions WasmCodegen's own runtime answers (for `mode = :fill`)
_runtime_has(name::Symbol) =
    isdefined(WasmCodegen, :RUNTIME_CCALLS) && haskey(WasmCodegen.RUNTIME_CCALLS, name)

function PureBLAS.wasm_foreign_functions(; mode::Symbol = :all)
    mode in (:all, :fill) || throw(ArgumentError("mode must be :all or :fill, got :$mode"))
    ff = Pair{Tuple{Symbol,String},Any}[]
    for (name, f) in ROUTES
        mode === :fill && _runtime_has(name) && continue
        push!(ff, (name, libblastrampoline) => f)
    end
    return ff
end

end # module
