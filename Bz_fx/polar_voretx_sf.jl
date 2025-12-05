using LazyGrids
using BlockArrays
using Printf
using StaticArrays
#using Interpolations
using SparseArrays
using SparseMatrixDicts
using SpecialFunctions
using FillArrays
using Parameters
using Test
using MAT
using BenchmarkTools
using BasicInterpolators: BicubicInterpolator

using Serialization
#using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

using CairoMakie
using LaTeXStrings
CairoMakie.activate!()
using DelimitedFiles
using ColorSchemes
using ScatteredInterpolation: interpolate, 
                            evaluate, 
                            InverseMultiquadratic, 
                            Multiquadratic
using Statistics
using JLD2
using Dierckx #: Spline2D, evaluate
using ModelingToolkit
using NonlinearSolve
using IterativeSolvers
using LinearAlgebra
using KrylovKit


include("dmsuite.jl")
include("transforms.jl")
include("utils.jl")
include("setBCs.jl")
include("shift_invert.jl")
include("shift_invert_arnoldi.jl")

@with_kw mutable struct TwoDimGrid{Nx, Nz} 
    x = @SVector zeros(Float64, Nx)
    z = @SVector zeros(Float64, Nz)
end

@with_kw mutable struct ChebMarix{Nx, Nz} 
    𝒟ˣ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Nx, Nx))
    𝒟²ˣ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nx, Nx))
    𝒟³ˣ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nx, Nx))
    𝒟⁴ˣ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nx, Nx))

    𝒟ᶻ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟³ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻ::Array{Float64, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟³ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟⁴ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
end

@with_kw mutable struct Operator{N}
"""
    `subperscript with N' means Operator with Neumann boundary condition 
        after kronker product
    `subperscript with D' means Operator with Dirchilet boundary condition
        after kronker product
""" 

    𝒟ˣ::Array{Float64,  2}     = SparseMatrixCSC(Zeros(N, N))
    𝒟²ˣ::Array{Float64, 2}     = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ˣ::Array{Float64, 2}     = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻ::Array{Float64,  2}     = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻ::Array{Float64, 2}     = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴺ::Array{Float64,  2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴺ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴺ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻᴰ::Array{Float64,  2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴰ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᴰ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))

    𝒟ˣᶻᴰ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟ˣᶻᴺ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))

    𝒟ˣ²ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ˣᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟³ˣᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N)) 
    𝒟ˣ³ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟²ˣ²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
end

@with_kw mutable struct MeanFlow{N} 
    B₀::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
  ∇ˣB₀::Array{Float64, 2}   = SparseMatrixCSC(Zeros(N, N))
  ∇ˣˣB₀::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))
  ∇ˣˣˣB₀::Array{Float64, 2} = SparseMatrixCSC(Zeros(N, N))
end


"""
    Construct the derivative operator
"""
function Construct_DerivativeOperator!(diffMatrix, grid, params)
    N = params.Nx * params.Nz

    # ------------- setup differentiation matrices  -------------------
    # Fourier in y-direction: y ∈ [0, L)
    x1, diffMatrix.𝒟ˣ  = FourierDiff(params.Nx, 1)
    _,  diffMatrix.𝒟²ˣ = FourierDiff(params.Nx, 2)
    _,  diffMatrix.𝒟³ˣ = FourierDiff(params.Nx, 3)
    _,  diffMatrix.𝒟⁴ˣ = FourierDiff(params.Nx, 4)

    t1 = @. sin(x1)
    t2 = diffMatrix.𝒟ˣ * t1

    println(t1[1])
    println(t2[1])

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    grid.x         = params.L/2π  * x1
    diffMatrix.𝒟ˣ  = (2π/params.L)^1 * diffMatrix.𝒟ˣ
    diffMatrix.𝒟²ˣ = (2π/params.L)^2 * diffMatrix.𝒟²ˣ
    diffMatrix.𝒟³ˣ = (2π/params.L)^3 * diffMatrix.𝒟³ˣ
    diffMatrix.𝒟⁴ˣ = (2π/params.L)^4 * diffMatrix.𝒟⁴ˣ

    #@assert maximum(grid.y) ≈ params.L && minimum(grid.y) ≈ 0.0

    if params.z_discret == "cheb"
        # Chebyshev in the z-direction
        z1, D1z = chebdif(params.Nz, 1)
        _,  D2z = chebdif(params.Nz, 2)
        _,  D3z = chebdif(params.Nz, 3)
        _,  D4z = chebdif(params.Nz, 4)

        # Transform the domain and derivative operators from [-1, 1] → [0, H]
        grid.z, diffMatrix.𝒟ᶻ, diffMatrix.𝒟²ᶻ  = chebder_transform(z1,  D1z, 
                                                                    D2z, 
                                                                    zerotoL_transform, 
                                                                    params.H)

        _, diffMatrix.𝒟³ᶻ, diffMatrix.𝒟⁴ᶻ      = chebder_transform_ho(z1, D1z, 
                                                                    D2z, 
                                                                    D3z, 
                                                                    D4z, 
                                                                    zerotoL_transform_ho, 
                                                                    params.H)
        
        @printf "size of Chebyshev matrix: %d × %d \n" size(diffMatrix.𝒟ᶻ)[1]  size(diffMatrix.𝒟ᶻ)[2]
        @assert maximum(grid.z) ≈ params.H && minimum(grid.z) ≈ 0.0

    else
        error("Invalid discretization type")
    end

    @testset "checking z-derivative differentiation matrix" begin
        tol = 2.0e-3
        t1 = diffMatrix.𝒟ᶻ * grid.z;
        @test maximum(t1) ≈ 1.0 atol=tol
        @test minimum(t1) ≈ 1.0 atol=tol
        t1 = diffMatrix.𝒟²ᶻ * (grid.z .^ 2);
        @test maximum(t1) ≈ factorial(2) atol=tol
        @test minimum(t1) ≈ factorial(2) atol=tol
        t1 = diffMatrix.𝒟⁴ᶻ * (grid.z .^ 4);
        @test maximum(t1) ≈ factorial(4) atol=tol
        @test minimum(t1) ≈ factorial(4) atol=tol
    end
    return nothing
end

function ImplementBCs_cheb!(Op, diffMatrix, params)
    Iˣ = sparse(Matrix(1.0I, params.Nx, params.Nx)) #Eye{Float64}(params.Ny)
    Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz)) #Eye{Float64}(params.Nz)

    #* Dirichilet boundary condition
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ
    @. diffMatrix.𝒟³ᶻᴰ = diffMatrix.𝒟³ᶻ
    @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ

    n = params.Nz
    for iter ∈ 1:n-1
        diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] = (diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] + 
                                -1.0 * diffMatrix.𝒟⁴ᶻᴰ[1,1] * diffMatrix.𝒟²ᶻᴰ[1,iter+1])

          diffMatrix.𝒟⁴ᶻᴰ[n,iter] = (diffMatrix.𝒟⁴ᶻᴰ[n,iter] + 
                                -1.0 * diffMatrix.𝒟⁴ᶻᴰ[n,n] * diffMatrix.𝒟²ᶻᴰ[n,iter])
    end

    diffMatrix.𝒟ᶻᴰ[1,1]  = 0.0
    diffMatrix.𝒟ᶻᴰ[n,n]  = 0.0

    diffMatrix.𝒟²ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴰ[n,n] = 0.0   

    diffMatrix.𝒟³ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟³ᶻᴰ[n,n] = 0.0   

    diffMatrix.𝒟⁴ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟⁴ᶻᴰ[n,n] = 0.0  

    #* Neumann boundary condition
    @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ

    for iter ∈ 1:n-1
        diffMatrix.𝒟²ᶻᴺ[1,iter+1] = (diffMatrix.𝒟²ᶻᴺ[1,iter+1] + 
                                -1.0 * diffMatrix.𝒟²ᶻᴺ[1,1] * diffMatrix.𝒟ᶻᴺ[1,iter+1]/diffMatrix.𝒟ᶻᴺ[1,1])

        diffMatrix.𝒟²ᶻᴺ[n,iter]   = (diffMatrix.𝒟²ᶻᴺ[n,iter] + 
                                -1.0 * diffMatrix.𝒟²ᶻᴺ[n,n] * diffMatrix.𝒟ᶻᴺ[n,iter]/diffMatrix.𝒟ᶻᴺ[n,n])
    end

    diffMatrix.𝒟²ᶻᴺ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴺ[n,n] = 0.0

    @. diffMatrix.𝒟ᶻᴺ[1,1:end] = 0.0
    @. diffMatrix.𝒟ᶻᴺ[n,1:end] = 0.0
    
    kron!( Op.𝒟ᶻᴰ  ,  Iˣ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iˣ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iˣ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iˣ , diffMatrix.𝒟ᶻᴺ )
    kron!( Op.𝒟²ᶻᴺ ,  Iˣ , diffMatrix.𝒟²ᶻᴺ)

    kron!( Op.𝒟ˣ   ,  diffMatrix.𝒟ˣ  ,  Iᶻ ) 
    kron!( Op.𝒟²ˣ  ,  diffMatrix.𝒟²ˣ ,  Iᶻ )
    kron!( Op.𝒟⁴ˣ  ,  diffMatrix.𝒟⁴ˣ ,  Iᶻ ) 

    kron!( Op.𝒟ˣᶻᴰ   ,  diffMatrix.𝒟ˣ  ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟ˣᶻᴺ   ,  diffMatrix.𝒟ˣ  ,  diffMatrix.𝒟ᶻᴺ  )
    kron!( Op.𝒟ˣ²ᶻᴰ  ,  diffMatrix.𝒟ˣ  ,  diffMatrix.𝒟²ᶻᴰ )

    kron!( Op.𝒟²ˣᶻᴰ  ,  diffMatrix.𝒟²ˣ ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟³ˣᶻᴰ  ,  diffMatrix.𝒟³ˣ ,  diffMatrix.𝒟ᶻᴰ  )

    kron!( Op.𝒟²ˣ²ᶻᴰ ,  diffMatrix.𝒟²ˣ ,  diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟ˣ³ᶻᴰ  ,  diffMatrix.𝒟ˣ  ,  diffMatrix.𝒟³ᶻᴰ )

    return nothing
end


function BasicState!(diffMatrix, mf, grid, params)
    x = grid.x 
    z = grid.z

    B₀ = zeros(length(x), length(z))

    a₀ = 0.15 
    a₁ = 0.85
    c  = 2.0
    δ  = 0.48
    for it in 1:length(x)
        @. B₀[it,:] = a₀ + a₁ * exp(-(x[it]-c)^2/(2δ^2))
    end

    ∂ˣB₀   = similar(B₀)
    ∂ˣˣB₀  = similar(B₀)
    ∂ˣˣˣB₀ = similar(B₀)

    """
    Calculating necessary derivatives of the mean-flow quantities
    """
    ∂ˣB₀    = gradient(  B₀,    grid.x, dims=1)
    ∂ˣˣB₀   = gradient(  ∂ˣB₀,  grid.x, dims=1)
    ∂ˣˣˣB₀  = gradient( ∂ˣˣB₀,  grid.x, dims=1)

    B₀     = transpose(B₀);       B₀    = B₀[:];
    ∂ˣB₀   = transpose(∂ˣB₀);    ∂ˣB₀   = ∂ˣB₀[:];
    ∂ˣˣB₀  = transpose(∂ˣˣB₀);   ∂ˣˣB₀  = ∂ˣˣB₀[:];
    ∂ˣˣˣB₀ = transpose(∂ˣˣˣB₀);  ∂ˣˣˣB₀ = ∂ˣˣˣB₀[:];

    mf.B₀[diagind(mf.B₀)]         = B₀;
    mf.∇ˣB₀[diagind(mf.∇ˣB₀)]     = ∂ˣB₀;
    mf.∇ˣˣB₀[diagind(mf.∇ˣˣB₀)]   = ∂ˣˣB₀;
    mf.∇ˣˣˣB₀[diagind(mf.∇ˣˣˣB₀)] = ∂ˣˣˣB₀;

    return nothing
end


function construct_matrices(Op, mf, params)
    N  = params.Nx * params.Nz
    I⁰ = sparse(Matrix(1.0I, N, N)) #Eye{Float64}(N)
    s₁ = size(I⁰, 1); s₂ = size(I⁰, 2)

    # allocating memory for the LHS and RHS matrices
    𝓛₁ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 5s₂))
    𝓛₂ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 5s₂))
    𝓛₃ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 5s₂))
    𝓛₄ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 5s₂))
    𝓛₅ = SparseMatrixCSC(Zeros{ComplexF64}(s₁, 5s₂))

    ℳ₁ = SparseMatrixCSC(Zeros{Float64}(s₁, 5s₂))
    ℳ₂ = SparseMatrixCSC(Zeros{Float64}(s₁, 5s₂))
    ℳ₃ = SparseMatrixCSC(Zeros{Float64}(s₁, 5s₂))
    ℳ₄ = SparseMatrixCSC(Zeros{Float64}(s₁, 5s₂))
    ℳ₅ = SparseMatrixCSC(Zeros{Float64}(s₁, 5s₂))

    @printf "Start constructing matrices \n"
    # -------------------- construct matrix  ------------------------
    # lhs of the matrix (size := 5 × 5)
    # eigenvectors: [uᶻ ωᶻ θ bᶻ jᶻ]ᵀ
    """
        inverse of the horizontal Laplacian: 
        ∇ₕ² ≡ ∂xx + ∂yy 
        H = (∇ₕ²)⁻¹
        Two methods have been implemented here:
        Method 1: SVD 
        Method 2: QR decomposition 
        Note - Method 2 is probably the `best' option 
                if the matrix, ∇ₕ², is close singular.
    """
    ∇ₕ² = SparseMatrixCSC(Zeros(N, N))
    ∇ₕ² = (1.0 * Op.𝒟²ˣ - 1.0 * params.kₓ^2 * I⁰)

    # Method 1. SVD decmposition 
    # U, Σ, V = svd(∇ₕ²); 
    # H = sparse(V * inv(Diagonal(Σ)) * transpose(U))

    # Method 2. QR decomposition
    Qm, Rm = qr(∇ₕ²)
    invR   = inv(Rm) 
    Qm     = sparse(Qm) # by sparsing the matrix speeds up matrix-matrix multiplication 
    Qᵀ     = transpose(Qm)
    H      = (invR * Qᵀ)

    # difference in L2-norm should be small: ∇ₕ² * (∇ₕ²)⁻¹ - I⁰ ≈ 0 
    @assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-6 "difference in L2-norm should be small"
    @printf "||∇ₕ² * (∇ₕ²)⁻¹ - I||₂ =  %f \n" norm(∇ₕ² * H - I⁰) 

    D⁴  = (1.0 * Op.𝒟⁴ˣ 
        + 1.0 * Op.𝒟⁴ᶻᴰ 
        + 1.0params.kₓ^4 * I⁰ 
        - 2.0params.kₓ^2 * Op.𝒟²ˣ 
        - 2.0 * params.kₓ^2 * Op.𝒟²ᶻᴰ
        + 2.0 * Op.𝒟²ˣ²ᶻᴰ)
        
    D²  = (1.0 * Op.𝒟²ᶻᴰ + 1.0 * ∇ₕ²)
    Dₙ² = (1.0 * Op.𝒟²ᶻᴺ + 1.0 * ∇ₕ²)

    #* 1. uᶻ equation (bcs: w = ∂ᶻᶻw = 0 @ z = 0, 1)
    𝓛₁[:,    1:1s₂] = 1.0params.E * D⁴

    𝓛₁[:,1s₂+1:2s₂] = -1.0 * Op.𝒟ᶻᴺ 
                    
    𝓛₁[:,3s₂+1:4s₂] = 1.0params.Λ * mf.B₀ * D² * Op.𝒟ᶻᴰ 
                    + 1.0params.Λ * mf.∇ˣˣB₀ * Op.𝒟ᶻᴰ
                    + 2.0params.Λ * mf.∇ˣB₀ * Op.𝒟ˣᶻᴰ
                    - 2.0params.Λ * mf.∇ˣˣB₀ * H * Op.𝒟²ˣᶻᴰ
                    - 1.0params.Λ * mf.∇ˣB₀  * H * Op.𝒟³ˣᶻᴰ
                    - 1.0params.Λ * mf.∇ˣˣˣB₀ * H * Op.𝒟ˣᶻᴰ
                    + 1.0params.Λ * params.kₓ^2 * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴰ
                    + 1.0params.Λ * mf.∇ˣB₀ * H * Op.𝒟ˣ³ᶻᴰ
    
    𝓛₁[:,4s₂+1:5s₂] = -2.0im * params.Λ * params.kₓ * mf.∇ˣˣB₀ * H * Op.𝒟ˣ
                    - 1.0im * params.Λ * params.kₓ * mf.∇ˣB₀ * H * Op.𝒟²ˣ
                    - 1.0im * params.Λ * params.kₓ * mf.∇ˣˣˣB₀ * H * I⁰
                    + 1.0im * params.Λ * params.kₓ^3 * mf.∇ˣB₀ * H * I⁰
                    + 1.0im * params.Λ * params.kₓ * mf.∇ˣB₀ * H * Op.𝒟²ᶻᴺ

    #* 2. ωᶻ equation (bcs: ∂ᶻζ = 0 @ z = 0, 1)
    𝓛₂[:,    1:1s₂] = 1.0 * Op.𝒟ᶻᴰ 
    𝓛₂[:,1s₂+1:2s₂] = 1.0params.E * Dₙ²
    𝓛₂[:,3s₂+1:4s₂] = -1.0im * params.kₓ * params.Λ * mf.∇ˣB₀ * H * Op.𝒟²ᶻᴰ     
    𝓛₂[:,4s₂+1:5s₂] = (1.0params.Λ * mf.B₀ * Op.𝒟ᶻᴺ 
                    + 1.0params.Λ * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴺ)

    #* 3. θ equation (bcs: θ = 0 @ z = 0, 1)
    𝓛₃[:,    1:1s₂] = 1.0 * I⁰
    𝓛₃[:,2s₂+1:3s₂] = 1.0params.q * D² 

    #* 4. bᶻ equation (conducting wall: bcs: bᶻ = 0 @ z = 0, 1)
    𝓛₄[:,    1:1s₂] = (1.0 * mf.B₀ * Op.𝒟ᶻᴰ 
                    + 1.0 * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴰ)   
    𝓛₄[:,1s₂+1:2s₂] = 1.0im * params.kₓ * mf.∇ˣB₀ * H * I⁰
    𝓛₄[:,3s₂+1:4s₂] = 1.0 * D² 

    #* 5. jᶻ equation (conducting wall: bcs: ∂ᶻjᶻ = 0 @ z = 0, 1)
    𝓛₅[:,    1:1s₂] = -1.0im * params.kₓ * mf.∇ˣB₀ * H * Op.𝒟²ᶻᴰ
    𝓛₅[:,1s₂+1:2s₂] = (1.0 * mf.B₀ * Op.𝒟ᶻᴺ
                    + 1.0 * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴺ)
    𝓛₅[:,4s₂+1:5s₂] = 1.0 * Dₙ² 

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃; 𝓛₄; 𝓛₅]);

##############

    # rhs of the matrix (size := 5 × 5)
    # [uz, wz, θ, bz, jz] 
    ℳ₁[:,2s₂+1:3s₂] = 1.0 * params.q * (Op.𝒟²ˣ - params.kₓ^2 * I⁰);

    ℳ = ([ℳ₁; ℳ₂; ℳ₃; ℳ₄; ℳ₅]);
    
    #@. 𝓛 *= 1.0/params.kₓ 
    return 𝓛, ℳ
end

"""
Parameters:
"""
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 2π          # horizontal domain size
    H::T        = 1.0          # vertical domain size
    Pr::T       = 1.0          # Prandtl number
    q::T        = 1.0          # Robert number
    Λ::T        = 0.1          # Elsasser number
    kₓ::T       = 0.0          # x-wavenumber
    E::T        = 5.0e-5       # Ekman number 
    Nx::Int64   = 320          # no. of x-grid points
    Nz::Int64   = 20           # no. of z-grid points
    z_discret::String = "cheb"   # option: "cheb", "fdm"
    #method::String    = "feast"
    #method::String    = "shift_invert"
    #method::String   = "arnoldi"
    method::String   = "KrylovKit"
end


function EigSolver(Op, mf, params, σ::ComplexF64)
    printstyled("kₓ: $(params.kₓ) \n"; color=:blue)

    𝓛, ℳ = construct_matrices(Op, mf, params)
    
    N = params.Nx * params.Nz 
    MatrixSize = 5N

    @assert size(𝓛, 1)  == MatrixSize && 
            size(𝓛, 2)  == MatrixSize &&
            size(ℳ, 1)  == MatrixSize &&
            size(ℳ, 2)  == MatrixSize "matrix size does not match!"

    if params.method == "shift_invert"
        printstyled("Eigensolver using Arpack eigs with shift and invert method ...\n"; 
                    color=:red)
        @printf "target eigenvalue: %f \n" σ.re

        λₛ, Χ = Arpack.eigs(𝓛, ℳ, nev=1, tol=1e-10, maxiter=1500, which=:LR, sigma=σ)

        println(λₛ)
        print_evals(λₛ, length(λₛ))

    elseif params.method == "arnoldi"
        printstyled("Arnoldi: based on Implicitly Restarted Arnoldi Method ... \n"; 
                        color=:red)
        @printf "target eigenvalue: %f \n" σ.re

        decomp, history = partialschur(construct_linear_map(𝓛 - σ*ℳ, ℳ), 
                                    nev=10, 
                                    maxdim=500,
                                    tol=1e-10, 
                                    restarts=1500, 
                                    which=:LM)
        @show history
        λₛ⁻¹, Χ = partialeigen(decomp)
        λₛ = @. 1.0 / λₛ⁻¹ + σ

        println(λₛ)
        print_evals(λₛ, length(λₛ))

    elseif params.method == "KrylovKit"
        printstyled("KrylovKit method... \n"; color=:red)
        @printf "target eigenvalue: %f \n" σ.re

        λₛ⁻¹, V1, info = eigsolve(construct_linear_map(𝓛- σ*ℳ, ℳ), 
                                rand(ComplexF64, size(𝓛,1)), 
                                10, :LM, 
                                maxiter=50, krylovdim=300, verbosity=1)

        λₛ⁰ = @. 1.0 / λₛ⁻¹ + σ
        Χ = zeros(ComplexF64, size(𝓛, 1), 1);

        idx = nearestval_idx(real(λₛ⁰), maximum(real(λₛ⁰)));

        Χ  = deepcopy(V1[idx])
        λₛ = λₛ⁰[idx]

        print_evals(λₛ⁰, length(λₛ⁰))

    else
        error("Invalid eigensolver method!")
    end
    # ======================================================================
    @assert length(λₛ) > 0 "No eigenvalue(s) found!"

    # Post Process egenvalues
    #λₛ, Χ = remove_evals(λₛ, Χ, 0.0, 10.0, "M") # `R`: real part of λₛ.

    if length(λₛ) ≥ 2 
        λₛ, Χ = sort_evals(λₛ, Χ, "R") 
    end  
    
    #λₛ = sort_evals_(λₛ, "R")

    #= 
        this removes any further spurious eigenvalues based on norm 
        if you don't need it, just `comment' it!
    =#
    # while norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) > 8e-2 # || imag(λₛ[1]) > 0
    #     @printf "norm (inside while): %f \n" norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
    #     λₛ, Χ = remove_spurious(λₛ, Χ)
    # end
   
    @printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    
    #print_evals(λₛ, length(λₛ))
    @printf "largest growth rate : %1.4e%+1.4eim\n" real(λₛ[1]) imag(λₛ[1])

    𝓛 = nothing
    ℳ = nothing

    #return nothing #
    return λₛ[1], Χ[:,1]
end


function solve_PolarVortex()
    params      = Params{Float64}(kₓ=0.5)
    grid        = TwoDimGrid{params.Nx,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Nx,  params.Nz}()
    Op          = Operator{params.Nx * params.Nz}()
    mf          = MeanFlow{params.Nx * params.Nz}()
    Construct_DerivativeOperator!(diffMatrix, grid, params)
    if params.z_discret == "cheb"
        ImplementBCs_cheb!(Op, diffMatrix, params)
    else
        error("Invalid discretization type!")
    end

    BasicState!(diffMatrix, mf, grid, params)
    N = params.Nx * params.Nz
    MatSize = Int(5N)

    @printf "E: %1.1e \n" params.E
    @printf "min/max of y: %f %f \n" minimum(grid.x) maximum(grid.x)
    @printf "no of y and z grid points: %i %i \n" params.Nx params.Nz

    #kₓ = range(0.01, stop=40.0, length=400)

    kₓ = 35.1
    for it in 1:1 #length(kₓ)
        params.kₓ = kₓ #[it]  
        
        @time λₛ, Χ = EigSolver(Op, mf, params, 0.0+0.0im)
            
        println("==================================================================")
    end

    # Λ  = params.Λ
    # Nx::Int = params.Nx
    # Nz::Int = params.Nz 
    # filename = "benchmark/eigenvals"  * "_elssaer" * string(Λ) * "_" * string(Nz) * string(Nx) * ".jld2"
    # jldsave(filename; kₓ=kₓ, λₛ=λₛ)
end

solve_PolarVortex()

