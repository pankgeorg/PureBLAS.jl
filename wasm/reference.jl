# julia --project=wasm wasm/reference.jl [OUT.json]: wasm/linalg.jl's operations run natively, with LinearAlgebra on
# PureBLAS (`PureBLAS.activate()`), on fixed inputs: the reference wasm/check.mjs compares a module with.
using PureBLAS, LinearAlgebra
include(joinpath(@__DIR__, "linalg.jl"))
const W = PureBLASWasm
out = get(ARGS, 1, joinpath(@__DIR__, "reference.json"))
PureBLAS.activate()

# fixed, well-conditioned inputs with distinct singular values and eigenvalues
# (sin of a product, not of a sum: full rank; a diagonal 1, 2, ..., n separates the eigenvalues)
mat(::Type{Float64}, m, n, s) = [sin(0.37i * j + 0.9i + s) + (i == j ? Float64(i) : 0.0) for i in 1:m, j in 1:n]
mat(::Type{ComplexF64}, m, n, s) = complex.(mat(Float64, m, n, s), mat(Float64, m, n, s + 0.5))
hpd(T, n, s) = (B = mat(T, n, n, s); B' * B + I)
flat(A::AbstractArray{Float64}) = vec(A)
flat(A::AbstractArray{ComplexF64}) = vec(reinterpret(Float64, vec(A)))

function case(io, op, sfx, T, dims, operands...)
    input = reduce(vcat, (flat(A) for A in operands))
    W.input_resize(length(input))
    for (i, x) in enumerate(input); W.input_set(i, x); end
    status = getfield(W, Symbol(op, "_", sfx))(dims...)
    output = copy(W.OUT)
    num(x) = repr(x)
    print(io, "{\"op\":\"", op, "_", sfx, "\",\"dims\":[", join(dims, ","), "],\"status\":", status,
          ",\"input\":[", join(num.(input), ","), "],\"output\":[", join(num.(output), ","), "]}")
end

open(out, "w") do io
    print(io, "[\n")
    first = true
    sep() = (first || print(io, ",\n"); first = false)
    for (sfx, T) in (("f64", Float64), ("c64", ComplexF64))
        sep(); case(io, :gemm, sfx, T, (5, 4, 3), mat(T, 5, 4, 0.1), mat(T, 4, 3, 0.2))
        sep(); case(io, :solve, sfx, T, (6, 2), mat(T, 6, 6, 0.3), mat(T, 6, 2, 0.4))
        sep(); case(io, :lu, sfx, T, (6,), mat(T, 6, 6, 0.5))
        sep(); case(io, :inv, sfx, T, (6,), mat(T, 6, 6, 0.6))
        sep(); case(io, :cholesky, sfx, T, (6,), hpd(T, 6, 0.7))
        sep(); case(io, :qr, sfx, T, (7, 4), mat(T, 7, 4, 0.8))
        sep(); case(io, :svd, sfx, T, (7, 4), mat(T, 7, 4, 0.9))
        sep(); case(io, :svdvals, sfx, T, (4, 7), mat(T, 4, 7, 1.0))
        sep(); case(io, :eigen, sfx, T, (6,), mat(T, 6, 6, 1.1))
        sep(); case(io, :eigen_hermitian, sfx, T, (6,), hpd(T, 6, 1.2))
        sep(); case(io, :exp, sfx, T, (5,), mat(T, 5, 5, 1.3) ./ 4)
    end
    print(io, "\n]\n")
end
println("wrote ", out)
