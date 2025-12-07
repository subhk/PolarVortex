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
    z_cheb = @SVector zeros(Float64, Nz)  # Chebyshev points on [-1,1] for clamped BC
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

    # Neumann BC: ∂z = 0 only (for j_z with conducting walls)
    𝒟ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟³ᶻᴺ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    # Dirichlet BC: u = 0 (for ω_z, θ, b_z)
    𝒟ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟²ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    𝒟³ᶻᴰ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))

    # Clamped BC: u = 0 AND ∂z u = 0 (for u_z with no-slip)
    𝒟ᶻᶜ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(Nz, Nz))   # DN in MATLAB
    𝒟²ᶻᶜ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))   # D2N in MATLAB
    𝒟⁴ᶻᶜ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))   # D4N in MATLAB (Orszag-Patera)
end

@with_kw mutable struct Operator{N}
"""
    `superscript with N' means Operator with Neumann boundary condition 
        after kronecker product
    `superscript with D' means Operator with Dirichlet boundary condition
        after kronecker product
    `superscript with C' means Operator with Clamped (no-slip) boundary condition
        after kronecker product
""" 

    𝒟ˣ::Array{Float64,  2}     = SparseMatrixCSC(Zeros(N, N))
    𝒟²ˣ::Array{Float64, 2}     = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ˣ::Array{Float64, 2}     = SparseMatrixCSC(Zeros(N, N))

    𝒟ᶻ::Array{Float64,  2}     = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻ::Array{Float64, 2}     = SparseMatrixCSC(Zeros(N, N))

    # Neumann operators (∂z = 0)
    𝒟ᶻᴺ::Array{Float64,  2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴺ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟³ᶻᴺ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))

    # Dirichlet operators (u = 0)
    𝒟ᶻᴰ::Array{Float64,  2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᴰ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟³ᶻᴰ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))

    # Clamped operators (u = 0 AND ∂z u = 0) for no-slip
    𝒟ᶻᶜ::Array{Float64,  2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟²ᶻᶜ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))
    𝒟⁴ᶻᶜ::Array{Float64, 2}    = SparseMatrixCSC(Zeros(N, N))

    𝒟ˣᶻᴰ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟ˣᶻᴺ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))
    𝒟ˣᶻᶜ::Array{Float64,  2}   = SparseMatrixCSC(Zeros(N, N))

    𝒟ˣ²ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²ˣᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟³ˣᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N)) 
    𝒟ˣ³ᶻᴰ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟³ˣᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N)) 
    𝒟ˣ³ᶻᴺ::Array{Float64,  2}  = SparseMatrixCSC(Zeros(N, N))

    𝒟²ˣ²ᶻᶜ::Array{Float64, 2}  = SparseMatrixCSC(Zeros(N, N))  # For clamped ∇⁴
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
    # Fourier in x-direction: x ∈ [0, L)
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

    if params.z_discret == "cheb"
        # Chebyshev in the z-direction
        z1, D1z = chebdif(params.Nz, 1)
        _,  D2z = chebdif(params.Nz, 2)
        _,  D3z = chebdif(params.Nz, 3)
        _,  D4z = chebdif(params.Nz, 4)

        # Store Chebyshev points on [-1,1] for clamped BC construction
        grid.z_cheb = z1

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

function ImplementBCs_cheb!(Op, diffMatrix, grid, params)
    Iˣ = sparse(Matrix(1.0I, params.Nx, params.Nx))
    Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz))

    n = params.Nz
    z = grid.z_cheb  # Chebyshev points on [-1, 1]

    # =========================================================================
    # 1. DIRICHLET boundary condition (u = 0 at boundaries)
    #    Used for: ω_z, θ, b_z
    # =========================================================================
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ
    @. diffMatrix.𝒟³ᶻᴰ = diffMatrix.𝒟³ᶻ

    # Zero diagonal at boundaries (MATLAB style)
    diffMatrix.𝒟ᶻᴰ[1,1]  = 0.0
    diffMatrix.𝒟ᶻᴰ[n,n]  = 0.0

    diffMatrix.𝒟²ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟²ᶻᴰ[n,n] = 0.0   

    diffMatrix.𝒟³ᶻᴰ[1,1] = 0.0
    diffMatrix.𝒟³ᶻᴰ[n,n] = 0.0    

    # =========================================================================
    # 2. NEUMANN boundary condition (∂z u = 0 at boundaries)
    #    Used for: j_z (with conducting walls)
    #    Pivot at (1,1) and (n,n)
    # =========================================================================
    Dn = copy(diffMatrix.𝒟ᶻ)
    D2n = copy(diffMatrix.𝒟²ᶻ)
    D3n = copy(diffMatrix.𝒟³ᶻ)

    # Modify D2n for Neumann BC
    for p in 1:n-1
        D2n[1, p+1] = D2n[1, p+1] - D2n[1, 1] * Dn[1, p+1] / Dn[1, 1]
        D2n[n, p]   = D2n[n, p]   - D2n[n, n] * Dn[n, p]   / Dn[n, n]
    end
    D2n[1, 1] = 0.0
    D2n[n, n] = 0.0

    # Modify D3n for Neumann BC
    for p in 1:n-1
        D3n[1, p+1] = D3n[1, p+1] - D3n[1, 1] * Dn[1, p+1] / Dn[1, 1]
        D3n[n, p]   = D3n[n, p]   - D3n[n, n] * Dn[n, p]   / Dn[n, n]
    end
    D3n[1, 1] = 0.0
    D3n[n, n] = 0.0

    # Zero boundary rows of Dn
    Dn[1, :] .= 0.0
    Dn[n, :] .= 0.0

    diffMatrix.𝒟ᶻᴺ  = Dn
    diffMatrix.𝒟²ᶻᴺ = D2n
    diffMatrix.𝒟³ᶻᴺ = D3n

    # =========================================================================
    # 3. CLAMPED boundary condition (u = 0 AND ∂z u = 0 at boundaries)
    #    Used for: u_z (no-slip velocity BC)
    #    This is the KEY difference from simple Neumann!
    # =========================================================================
    
    # --- 3a. DN (first derivative for clamped) ---
    # Pivot at (1,2) and (n,n-1) - different from Neumann!
    DN = copy(diffMatrix.𝒟ᶻ)
    
    # --- 3b. D2N (second derivative for clamped) ---
    # Pivot at (1,2) and (n,n-1)
    D2N = copy(diffMatrix.𝒟²ᶻ)
    
    for p in 1:n-2
        D2N[1, p+2] = D2N[1, p+2] - D2N[1, 2] * DN[1, p+2] / DN[1, 2]
        D2N[n, p]   = D2N[n, p]   - D2N[n, n-1] * DN[n, p] / DN[n, n-1]
    end
    D2N[1, 2]   = 0.0
    D2N[n, n-1] = 0.0

    # --- 3c. D4N (fourth derivative for clamped) - Orszag-Patera method ---
    # This is the special formula for clamped BCs
    D1_raw = copy(diffMatrix.𝒟ᶻ)  # Need raw D1 for the formula
    
    # Scale factor for domain transformation [-1,1] -> [0,H]
    # Since D operators are already transformed, we need to work carefully
    # The MATLAB code works on [-1,1] domain, so we construct D4N there first
    
    # Get raw Chebyshev matrices on [-1,1]
    _, D1_cheb = chebdif(n, 1)
    
    # S matrix: zeros at boundaries, 1/(1-z²) in interior
    S = zeros(n, n)
    for i in 2:n-1
        S[i, i] = 1.0 / (1.0 - z[i]^2)
    end
    
    # D4N using Orszag-Patera formula on [-1,1]
    D4N_cheb = (Diagonal(1.0 .- z.^2) * D1_cheb^4 
              - 8.0 * Diagonal(z) * D1_cheb^3 
              - 12.0 * D1_cheb^2) * S
    
    # Apply clamped BC modification to D4N (pivot at (1,2) and (n,n-1))
    for p in 1:n-2
        D4N_cheb[1, p+2] = D4N_cheb[1, p+2] - D4N_cheb[1, 2] * DN[1, p+2] / DN[1, 2]
        D4N_cheb[n, p]   = D4N_cheb[n, p]   - D4N_cheb[n, n-1] * DN[n, p] / DN[n, n-1]
    end
    D4N_cheb[1, 2]   = 0.0
    D4N_cheb[n, n-1] = 0.0
    
    # Transform D4N from [-1,1] to [0,H]: D4_transformed = (2/H)^4 * D4_cheb
    D4N = (2.0/params.H)^4 * D4N_cheb

    # Zero boundary rows of DN
    DN[1, :] .= 0.0
    DN[n, :] .= 0.0

    diffMatrix.𝒟ᶻᶜ  = DN
    diffMatrix.𝒟²ᶻᶜ = D2N
    diffMatrix.𝒟⁴ᶻᶜ = D4N

    # =========================================================================
    # 4. Kronecker products for 2D operators
    # =========================================================================
    
    # Dirichlet operators
    kron!(Op.𝒟ᶻᴰ,  Iˣ, diffMatrix.𝒟ᶻᴰ)
    kron!(Op.𝒟²ᶻᴰ, Iˣ, diffMatrix.𝒟²ᶻᴰ)
    kron!(Op.𝒟³ᶻᴰ, Iˣ, diffMatrix.𝒟³ᶻᴰ)

    # Neumann operators
    kron!(Op.𝒟ᶻᴺ,  Iˣ, diffMatrix.𝒟ᶻᴺ)
    kron!(Op.𝒟²ᶻᴺ, Iˣ, diffMatrix.𝒟²ᶻᴺ)
    kron!(Op.𝒟³ᶻᴺ, Iˣ, diffMatrix.𝒟³ᶻᴺ)

    # Clamped operators (for no-slip u_z)
    kron!(Op.𝒟ᶻᶜ,  Iˣ, diffMatrix.𝒟ᶻᶜ)
    kron!(Op.𝒟²ᶻᶜ, Iˣ, diffMatrix.𝒟²ᶻᶜ)
    kron!(Op.𝒟⁴ᶻᶜ, Iˣ, diffMatrix.𝒟⁴ᶻᶜ)

    # x-derivatives
    kron!(Op.𝒟ˣ,  diffMatrix.𝒟ˣ,  Iᶻ) 
    kron!(Op.𝒟²ˣ, diffMatrix.𝒟²ˣ, Iᶻ)
    kron!(Op.𝒟⁴ˣ, diffMatrix.𝒟⁴ˣ, Iᶻ) 

    # Mixed derivatives
    kron!(Op.𝒟ˣᶻᴰ,  diffMatrix.𝒟ˣ, diffMatrix.𝒟ᶻᴰ   )
    kron!(Op.𝒟ˣᶻᴺ,  diffMatrix.𝒟ˣ, diffMatrix.𝒟ᶻᴺ   )
    kron!(Op.𝒟ˣᶻᶜ,  diffMatrix.𝒟ˣ, diffMatrix.𝒟ᶻᶜ   )
    kron!(Op.𝒟ˣ²ᶻᴰ, diffMatrix.𝒟ˣ, diffMatrix.𝒟²ᶻᴰ  )

    kron!(Op.𝒟²ˣᶻᴰ, diffMatrix.𝒟²ˣ, diffMatrix.𝒟ᶻᴰ  )
    kron!(Op.𝒟³ˣᶻᴰ, diffMatrix.𝒟³ˣ, diffMatrix.𝒟ᶻᴰ  )

    kron!(Op.𝒟²ˣ²ᶻᶜ, diffMatrix.𝒟²ˣ, diffMatrix.𝒟²ᶻᶜ)
    kron!(Op.𝒟ˣ³ᶻᴰ,  diffMatrix.𝒟ˣ,  diffMatrix.𝒟³ᶻᴰ)

    return nothing
end


function BasicState!(diffMatrix, mf, grid, params)
    x = grid.x 
    z = grid.z

    B₀ = zeros(length(x), length(z))

    a₀ = 0.15 
    a₁ = 0.85
    c  = 0.5 * params.L  # Center of Gaussian
    δ  = 0.4777          # Width (matching MATLAB)
    
    for it in 1:length(x)
        @. B₀[it,:] = a₀ + a₁ * exp(-(x[it]-c)^2/(2δ^2))
    end

    ∂ˣB₀   = similar(B₀)
    ∂ˣˣB₀  = similar(B₀)
    ∂ˣˣˣB₀ = similar(B₀)

    """
    Calculating necessary derivatives of the mean-flow quantities
    Using finite differences (as in MATLAB diffxy function)
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
    I⁰ = sparse(Matrix(1.0I, N, N))
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
    
    # Inverse of horizontal Laplacian: H = (∇ₕ²)⁻¹ = (D²x - k²)⁻¹
    ∇ₕ² = (1.0 * Op.𝒟²ˣ - 1.0 * params.kₓ^2 * I⁰)

    # QR decomposition for inverse
    Qm, Rm = qr(∇ₕ²)
    invR   = inv(Rm) 
    Qm     = sparse(Qm)
    Qᵀ     = transpose(Qm)
    H      = (invR * Qᵀ)

    @assert norm(∇ₕ² * H - I⁰) ≤ 1.0e-6 "difference in L2-norm should be small"
    @printf "||∇ₕ² * (∇ₕ²)⁻¹ - I||₂ =  %f \n" norm(∇ₕ² * H - I⁰) 

    # =========================================================================
    # Composite operators
    # =========================================================================
    
    # ∇⁴ for CLAMPED BC (no-slip u_z): uses 𝒟⁴ᶻᶜ and 𝒟²ᶻᶜ
    Dᶜ⁴ = (1.0 * Op.𝒟⁴ˣ 
         + 1.0 * Op.𝒟⁴ᶻᶜ 
         + 1.0 * params.kₓ^4 * I⁰ 
         - 2.0 * params.kₓ^2 * Op.𝒟²ˣ 
         - 2.0 * params.kₓ^2 * Op.𝒟²ᶻᶜ
         + 2.0 * Op.𝒟²ˣ²ᶻᶜ)
        
    # ∇² for DIRICHLET BC (ω_z, θ, b_z = 0)
    D²  = (1.0 * Op.𝒟²ᶻᴰ + 1.0 * ∇ₕ²)
    
    # ∇² for NEUMANN BC (∂z j_z = 0)
    Dₙ² = (1.0 * Op.𝒟²ᶻᴺ + 1.0 * ∇ₕ²)

    # For b_z terms in u_z equation, need D² acting on b_z (which has Dirichlet BC)
    # but with D_z operators that respect b_z = 0
    D²_bz = (1.0 * Op.𝒟²ᶻᴰ + 1.0 * ∇ₕ²)

    # =========================================================================
    # EQUATION 1: u_z (no-slip: u_z = 0, ∂_z u_z = 0)
    # E∇⁴u_z - D_z ω_z + Λ[Lorentz terms] = Ra·q(D²x - k²)θ
    # =========================================================================
    
    # u_z coefficient: E∇⁴ with CLAMPED BC
    𝓛₁[:, 1:1s₂] = 1.0 * params.E * Dᶜ⁴

    # ω_z coefficient: -D_z (ω_z has Dirichlet BC)
    𝓛₁[:, 1s₂+1:2s₂] = -1.0 * Op.𝒟ᶻᴰ 
                    
    # b_z coefficient (Lorentz terms): b_z has Dirichlet BC
    𝓛₁[:, 3s₂+1:4s₂] = (1.0 * params.Λ * mf.B₀ * D²_bz * Op.𝒟ᶻᴰ 
                      + 1.0 * params.Λ * mf.∇ˣˣB₀ * Op.𝒟ᶻᴰ
                      + 2.0 * params.Λ * mf.∇ˣB₀ * Op.𝒟ˣᶻᴰ
                      - 2.0 * params.Λ * mf.∇ˣˣB₀ * H * Op.𝒟²ˣᶻᴰ
                      - 1.0 * params.Λ * mf.∇ˣB₀  * H * Op.𝒟³ˣᶻᴰ
                      - 1.0 * params.Λ * mf.∇ˣˣˣB₀ * H * Op.𝒟ˣᶻᴰ
                      + 1.0 * params.Λ * params.kₓ^2 * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴰ
                      + 1.0 * params.Λ * mf.∇ˣB₀ * H * Op.𝒟ˣ³ᶻᴰ)
    
    # j_z coefficient (Lorentz terms): j_z has Neumann BC (conducting)
    𝓛₁[:, 4s₂+1:5s₂] = (-2.0im * params.Λ * params.kₓ * mf.∇ˣˣB₀ * H * Op.𝒟ˣ
                      - 1.0im * params.Λ * params.kₓ * mf.∇ˣB₀ * H * Op.𝒟²ˣ
                      - 1.0im * params.Λ * params.kₓ * mf.∇ˣˣˣB₀ * H * I⁰
                      + 1.0im * params.Λ * params.kₓ^3 * mf.∇ˣB₀ * H * I⁰
                      + 1.0im * params.Λ * params.kₓ * mf.∇ˣB₀ * H * Op.𝒟²ᶻᴺ)

    # =========================================================================
    # EQUATION 2: ω_z (no-slip: ω_z = 0)
    # D_z u_z + E∇²ω_z + Λ[Lorentz terms] = 0
    # =========================================================================
    
    # u_z coefficient: D_z with CLAMPED BC (since ∂_z u_z = 0)
    𝓛₂[:, 1:1s₂] = 1.0 * Op.𝒟ᶻᶜ
    
    # ω_z coefficient: E∇² with DIRICHLET BC
    𝓛₂[:, 1s₂+1:2s₂] = 1.0 * params.E * D²
    
    # b_z coefficient: b_z has Dirichlet BC
    𝓛₂[:, 3s₂+1:4s₂] = -1.0im * params.kₓ * params.Λ * mf.∇ˣB₀ * H * Op.𝒟²ᶻᴰ
    
    # j_z coefficient: j_z has Neumann BC (conducting)
    𝓛₂[:, 4s₂+1:5s₂] = (1.0 * params.Λ * mf.B₀ * Op.𝒟ᶻᴺ 
                      + 1.0 * params.Λ * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴺ)

    # =========================================================================
    # EQUATION 3: θ (θ = 0 at boundaries)
    # u_z + q∇²θ = 0
    # =========================================================================
    𝓛₃[:, 1:1s₂] = 1.0 * I⁰
    𝓛₃[:, 2s₂+1:3s₂] = 1.0 * params.q * D² 

    # =========================================================================
    # EQUATION 4: b_z (CONDUCTING wall: b_z = 0)
    # f·D_z u_z + f'·u_x + ∇²b_z = 0
    # =========================================================================
    
    # u_z coefficient: uses CLAMPED BC
    𝓛₄[:, 1:1s₂] = (1.0 * mf.B₀ * Op.𝒟ᶻᶜ 
                  + 1.0 * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᶜ)
    
    # ω_z coefficient
    𝓛₄[:, 1s₂+1:2s₂] = 1.0im * params.kₓ * mf.∇ˣB₀ * H * I⁰
    
    # b_z coefficient: ∇² with DIRICHLET BC
    𝓛₄[:, 3s₂+1:4s₂] = 1.0 * D² 

    # =========================================================================
    # EQUATION 5: j_z (CONDUCTING wall: ∂_z j_z = 0)
    # f·D_z ω_z + f'·D_z u_y + ∇²j_z = 0
    # =========================================================================
    
    # u_z coefficient: uses CLAMPED BC
    𝓛₅[:, 1:1s₂] = -1.0im * params.kₓ * mf.∇ˣB₀ * H * Op.𝒟²ᶻᶜ
    
    # ω_z coefficient: ω_z has Dirichlet BC
    𝓛₅[:, 1s₂+1:2s₂] = (1.0 * mf.B₀ * Op.𝒟ᶻᴰ
                      + 1.0 * mf.∇ˣB₀ * H * Op.𝒟ˣᶻᴰ)
    
    # j_z coefficient: ∇² with NEUMANN BC
    𝓛₅[:, 4s₂+1:5s₂] = 1.0 * Dₙ² 

    𝓛 = ([𝓛₁; 𝓛₂; 𝓛₃; 𝓛₄; 𝓛₅]);

    # =========================================================================
    # RHS matrix (Rayleigh number multiplier)
    # =========================================================================
    ℳ₁[:, 2s₂+1:3s₂] = -1.0 * params.q * (Op.𝒟²ˣ - params.kₓ^2 * I⁰);

    ℳ = ([ℳ₁; ℳ₂; ℳ₃; ℳ₄; ℳ₅]);
    
    return 𝓛, ℳ
end

"""
Parameters:
"""
@with_kw mutable struct Params{T<:Real} @deftype T
    L::T        = 2π          # horizontal domain size (MATCHING MATLAB Asp=4)
    H::T        = 1.0          # vertical domain size
    Pr::T       = 1.0          # Prandtl number
    q::T        = 1.0          # Roberts number
    Λ::T        = 0.04          # Elsasser number (MATCHING MATLAB Els=0.5)
    kₓ::T       = 0.0          # y-wavenumber
    E::T        = 5.0e-5       # Ekman number 
    Nx::Int64   = 180           # no. of x-grid points (MATCHING MATLAB)
    Nz::Int64   = 24           # no. of z-grid points (MATCHING MATLAB)
    z_discret::String = "cheb"
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
                                maxiter=150, krylovdim=300, verbosity=1)

        λₛ = @. 1.0 / λₛ⁻¹ + σ
        Χ = zeros(ComplexF64, size(𝓛, 1), 1);

        print_evals(λₛ, length(λₛ))

    else
        error("Invalid eigensolver method!")
    end

    λₛ = remove_evals_(λₛ, 10.0, 1e10, "R")

    @assert length(λₛ) > 0 "No eigenvalue(s) found!"
   
    @printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1] * ℳ * Χ[:,1])
    @printf "critical Ra : %1.4e \n" real(λₛ[1]) 

    𝓛 = nothing
    ℳ = nothing

    return real(λₛ[1])
end


function solve_PolarVortex()
    params      = Params{Float64}(kₓ=0.5)
    grid        = TwoDimGrid{params.Nx,  params.Nz}()
    diffMatrix  = ChebMarix{ params.Nx,  params.Nz}()
    Op          = Operator{params.Nx * params.Nz}()
    mf          = MeanFlow{params.Nx * params.Nz}()
    
    Construct_DerivativeOperator!(diffMatrix, grid, params)
    
    if params.z_discret == "cheb"
        ImplementBCs_cheb!(Op, diffMatrix, grid, params)
    else
        error("Invalid discretization type!")
    end

    BasicState!(diffMatrix, mf, grid, params)
    N = params.Nx * params.Nz
    MatSize = Int(5N)

    @printf "E: %1.1e \n" params.E
    @printf "Λ: %1.2f \n" params.Λ
    @printf "L: %1.2f \n" params.L
    @printf "min/max of x: %f %f \n" minimum(grid.x) maximum(grid.x)
    @printf "no of x and z grid points: %i %i \n" params.Nx params.Nz

    kₓ = range(0.01, stop=40.0, length=600)
    λₛ = zeros(Float64, length(kₓ))

    #kₓ = 31.1
    for it in 1:length(kₓ)
        params.kₓ = kₓ[it] 
        
        @time λₛ = EigSolver(Op, mf, params, 0.0+0.0im)
            
        println("==================================================================")
    end

    Λ  = params.Λ
    Nx::Int = params.Nx
    Nz::Int = params.Nz 
    filename = "benchmark/eigenvals_ns"  * "_elssaer" * string(Λ) * "_" * string(Nz) * string(Nx) * ".jld2"
    jldsave(filename; kₓ=kₓ, λₛ=λₛ)
end

solve_PolarVortex()