# The entry points of the prebuilt module (wasm/build.jl): LinearAlgebra's common operations in Float64 and
# ComplexF64, through PureBLAS. A host passes numbers only: it fills the input buffer, calls an operation, and reads
# the output buffer. Matrices are column-major; a complex number is two Float64s, real then imaginary.
#
#   input_resize(len); input_set(i, x) for i in 1:len     the operands, one after the other
#   <op>_<f64|c64>(dims...) -> status                       0: done; > 0: LAPACK's info (singular, not definite);
#                                                          -1: the input's length does not match dims
#   (any other error, e.g. an SVD that does not converge, is thrown: a WebAssembly exception in JavaScript)
#   output_length(); output_get(i)                          the results, one after the other
#
# | Operation            | Input            | Output                                                     |
# |----------------------|------------------|------------------------------------------------------------|
# | gemm(m, k, n)        | A m×k, B k×n     | A*B, m×n                                                   |
# | solve(n, nrhs)       | A n×n, B n×nrhs  | A\B, n×nrhs                                                |
# | lu(n)                | A n×n            | the factors L\U, n×n; the pivots, n (as Float64)           |
# | inv(n)               | A n×n            | inv(A)                                                     |
# | cholesky(n)          | A n×n Hermitian  | U (upper; the strict lower part zero), from A's upper part |
# | qr(m, n)             | A m×n            | Q m×k (thin), R k×n; k = min(m, n)                         |
# | svd(m, n)            | A m×n            | S k (real), U m×k, Vt k×n                                  |
# | svdvals(m, n)        | A m×n            | S k (real)                                                 |
# | eigen(n)             | A n×n            | eigenvalues n, eigenvectors n×n: always complex            |
# | eigen_hermitian(n)   | A n×n Hermitian  | eigenvalues n (real, ascending), eigenvectors n×n          |
# | exp(n)               | A n×n            | exp(A)                                                     |
module PureBLASWasm
using LinearAlgebra

const IN = Float64[]
const OUT = Float64[]
input_resize(len::Int) = (resize!(IN, len); fill!(IN, 0.0); length(IN))
input_set(i::Int, x::Float64) = (@inbounds IN[i] = x; 0)
output_length() = length(OUT)
output_get(i::Int) = @inbounds OUT[i]

nreal(::Type{Float64}) = 1
nreal(::Type{ComplexF64}) = 2
# the matrix at offset `off` (in elements of T) of the input
function operand(::Type{T}, off::Int, m::Int, n::Int) where {T}
    A = Matrix{T}(undef, m, n)
    k = nreal(T) * off
    @inbounds for j in 1:n, i in 1:m
        if T === Float64
            A[i, j] = IN[k += 1]
        else
            re = IN[k += 1]; im = IN[k += 1]
            A[i, j] = T(re, im)
        end
    end
    return A
end
emit!(x::Real) = (push!(OUT, Float64(x)); nothing)
emit!(x::Complex) = (push!(OUT, real(x)); push!(OUT, imag(x)); nothing)
emit!(A::AbstractArray) = (for x in A; emit!(x); end; nothing)

# (an operation: -1 for an input of the wrong length, else its body's status)
macro op(name, T, dims, len, body)
    return esc(quote
        function $name($(dims.args...))
            empty!(OUT)
            length(IN) == nreal($T) * ($len) || return -1
            $body
        end
    end)
end

for (sfx, T) in (("f64", Float64), ("c64", ComplexF64))
    @eval begin
        @op $(Symbol("gemm_", sfx)) $T (m::Int, k::Int, n::Int) (m * k + k * n) begin
            emit!(operand($T, 0, m, k) * operand($T, m * k, k, n)); 0
        end
        @op $(Symbol("solve_", sfx)) $T (n::Int, nrhs::Int) (n * n + n * nrhs) begin
            F = lu(operand($T, 0, n, n); check = false)
            issuccess(F) || return Int(F.info)
            emit!(F \ operand($T, n * n, n, nrhs)); 0
        end
        @op $(Symbol("lu_", sfx)) $T (n::Int,) (n * n) begin
            F = lu(operand($T, 0, n, n); check = false)
            emit!(F.factors); emit!(F.ipiv); Int(F.info)
        end
        @op $(Symbol("inv_", sfx)) $T (n::Int,) (n * n) begin
            F = lu(operand($T, 0, n, n); check = false)
            issuccess(F) || return Int(F.info)
            emit!(inv(F)); 0
        end
        @op $(Symbol("cholesky_", sfx)) $T (n::Int,) (n * n) begin
            F = cholesky(Hermitian(operand($T, 0, n, n), :U); check = false)
            issuccess(F) || return Int(F.info)
            emit!(UpperTriangular(F.U)); 0
        end
        @op $(Symbol("qr_", sfx)) $T (m::Int, n::Int) (m * n) begin
            F = qr(operand($T, 0, m, n))
            emit!(Matrix(F.Q)); emit!(F.R); 0
        end
        @op $(Symbol("svd_", sfx)) $T (m::Int, n::Int) (m * n) begin
            F = svd(operand($T, 0, m, n))
            emit!(F.S); emit!(F.U); emit!(F.Vt); 0
        end
        @op $(Symbol("svdvals_", sfx)) $T (m::Int, n::Int) (m * n) begin
            emit!(svdvals(operand($T, 0, m, n))); 0
        end
        @op $(Symbol("eigen_", sfx)) $T (n::Int,) (n * n) begin
            F = eigen(operand($T, 0, n, n))
            for x in F.values; emit!(complex(x)); end
            for x in F.vectors; emit!(complex(x)); end
            0
        end
        @op $(Symbol("eigen_hermitian_", sfx)) $T (n::Int,) (n * n) begin
            F = eigen(Hermitian(operand($T, 0, n, n), :U))
            emit!(F.values); emit!(F.vectors); 0
        end
        @op $(Symbol("exp_", sfx)) $T (n::Int,) (n * n) begin
            emit!(exp(operand($T, 0, n, n))); 0
        end
    end
end

const OPS = (:gemm => (Int, Int, Int), :solve => (Int, Int), :lu => (Int,), :inv => (Int,), :cholesky => (Int,),
             :qr => (Int, Int), :svd => (Int, Int), :svdvals => (Int, Int), :eigen => (Int,),
             :eigen_hermitian => (Int,), :exp => (Int,))
"The module's exports: `(function, argument types)`, for `WasmCodegen.compile`."
function entries(; ops = first.(OPS), types = ("f64", "c64"))
    es = Any[(input_resize, (Int,)), (input_set, (Int, Float64)), (output_length, ()), (output_get, (Int,))]
    for (op, args) in OPS, sfx in types
        op in ops && push!(es, (getfield(@__MODULE__, Symbol(op, "_", sfx)), args))
    end
    return es
end

end # module
