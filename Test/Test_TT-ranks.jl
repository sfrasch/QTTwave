using LinearAlgebra, TensorRefinement, Test

function A_b_c_Gauss(q::Int)
    #= GLRK TABLEAU =#
    #=
    Input:  q = number of GRLK stages.
    Output: A = matrix of size qxq whose entries are the coefficients of GLRK with q stages.
            b = column vector qx1 whose entries are the weights of Gauss-Legendre quadrature in [0,1].
            c = column vector qx1 whose entries are the nodes of Gauss-Legendre quadrature in [0,1].
    =#

    #= Golub–Welsch algorithm =#
    d = (1:(q-1)) ./ sqrt.((4*(1:(q-1)).^2).-1) 
    J = diagm(-1 => d, 1 => d)
    c = (eigvals(J).+1)./2 
    b = (eigvecs(J)[1,:].*eigvecs(J)[1,:]) 

    #= Matrix A=(aⱼᵢ)ⱼᵢ by integrating Lagrange pols ℓᵢ in [0,cⱼ] =#
    A = zeros(Float64,q,q)
    for i ∈ collect(1:1:q)
        for j ∈ collect(1:1:q)
            cs_corr = c[j]*c
            bs_corr = c[j]*b

            num = 1; den = 1;
            for jj ∈ collect(1:1:q)
                if jj != i 
                    num = num.*(cs_corr.-c[jj])
                    den = den*(c[i]-c[jj])
                end
            end
            num = only(sum(num.*bs_corr, dims=1))
            A[j,i] = num/den
        end
    end

    return A,b,c
end

function QTT_IL0(::Type{T}, L::Int) where {T<:FloatRC}
    #= 
    QTT of L cores of the identity matrix of size (2^L-1)x(2^L-1) 
    padded with zeros in the last row and column to get a matrix of size 2^L x 2^L.
    =#

    if L < 1
        throw(ArgumentError("L should be at least one"))
    end

	W = MatrixDec{T}() 
	if L == 1 
		W_1 = T[1 0
				0 0]
		decpush!(W,permutedims(reshape(W_1, 2, 1, 2, 1), [2,1,3,4]))
    else
    	W_1 = T[1 0 0 0
             	0 0 0 1]
    	decpush!(W,permutedims(reshape(W_1, 2, 1, 2, 2), [2,1,3,4]))
    
    	C = [Matrix(1.0I, 2, 2) [0 0; 0 0] 
        	[1 0; 0 0] [0 0; 0 1]]
    	C = permutedims(reshape(C, 2, 2, 2, 2), [2,1,3,4])
    	for ℓ ∈ 1:L-2
        	decpush!(W,copy(C))
    	end

    	W_end = [Matrix(1.0I, 2, 2)
              	[1 0; 0 0]]
    	decpush!(W,permutedims(reshape(W_end, 2, 2, 2, 1), [2,1,3,4]))
	end

    return W
end

function stiffnessbpxdd(::Type{T}, L::Int) where {T<:FloatRC}
    #=
    QTT of L cores of the preconditioned stiffness matrix with 
    homogeneous Dirichlet boundary conditions on (0,1).
    =#
    
	c = [zero(T); ones(T, 1)]
	ΛΛ = dint(T, L, 1, Diagonal(c); major="first")
	Q = diffbpxdd(T, L, 1; major="first")
	B = decmp(Q, 1, decmp(ΛΛ, 2, Q, 1), 1)
	decskp!(B, L+1; path="backward")

	return B
end

function stiffnessbpxdd_left(::Type{T}, L::Int) where {T<:FloatRC}
    #=
    Let A be the stiffness matrix in the L^2-normalized basis 
    with homogeneous Dirichlet boundary conditions on (0,1),
    and C the BPX preconditioner.
    This function provides a QTT of L cores of the matrix C*A. 
    =#
    
	c = [zero(T); ones(T, 1)]
	ΛΛ = dint(T, L, 1, Diagonal(c); major="first")
	Q = diffbpxdd(T, L, 1; major="first")
    M = diffdd(T, L, L, 1; major="first")
	CA = decmp(Q, 1, decmp(ΛΛ, 2, M, 1), 1)
	decskp!(CA, L+1; path="backward")

	return CA
end

function stiffnessdd(::Type{T}, L::Int) where {T<:FloatRC}
    #=
    QTT of L cores of the stiffness matrix with 
    homogeneous Dirichlet boundary conditions on (0,1).
    =#

	c = [zero(T); ones(T, 1)]
	ΛΛ = dint(T, L, 1, Diagonal(c); major="first")
	M = diffdd(T, L, L, 1; major="first")
	A = decmp(M, 1, decmp(ΛΛ, 2, M, 1), 1)
	decskp!(A, L+1; path="backward")
	
    return A
end

function invmassSMW(::Type{T}, L::Int) where {T<:FloatRC}
    #=
    QTT decomposition of L+1 cores of the factors involved in the Sherman–Morrison–Woodbury formula
    for the inverse of the mass matrix in the L^2-normalized basis on (0,1).
    =#

	if L < 1
		throw(ArgumentError("L should be at least one"))
	end
    W = MatrixDec{T}()
    W2 = MatrixDec{T}()
    W22 = MatrixDec{T}()
    C = MatrixDec{T}()
	κ,λ = 4*one(T)/15,-one(T)/15
	U = T[κ λ; λ κ]; U = reshape(U, 1, 2, 2, 1)
	Φ = T[κ,λ,λ,κ];  Φ = reshape(Φ, 2, 2, 1)
    ΦΦ = factormp(Φ, [], Φ, [])
    decpushfirst!(W, factorvcat(U, ΦΦ))

#     ### 2cols
    ΦΦ2 = ΦΦ[1:2,:,:,:]
    Ψ = factorvcat(U, ΦΦ2)
    decpushfirst!(W2, Ψ)

    ### 2cols2rows
    Z = factorvcat(U, ΦΦ)
    Y = Matrix{T}(I, 5, 5)
	for ℓ ∈ L-1:-1:1
		Δ = 1/(κ^2 - 1)
        Φ1 = T[Δ*λ 0; -Δ*κ*λ 1]
        ΦΦ = kron(Φ1, Φ1)
        X = T[1 -κ*Δ 0 0 0; zeros(T, 4, 1) ΦΦ]
        Y = X*Y
		κ,λ = (1-Δ*λ^2)*κ,Δ*λ^2
	end
    Y = Y[1:1,:]
    Z = factorcontract(Y, Z) # of size 1 x 2 x 2 x 1
    decpushfirst!(W22, Z)

    Z = reshape(Z, 2, 2)
    Z = inv(T[0 1; 1 0] - Z)
    Z = reshape(Z, 1, 2, 2, 1)
    decpushfirst!(C, factormp(factormp(Ψ, 2, Z, 1), 2, Ψ, 2))
    
    
	κ,λ = 4*one(T)/15,-one(T)/15
	for ℓ ∈ L-1:-1:1
		Δ = 1/(κ^2 - 1)
        Φ = T[1 -Δ*κ*λ; 0 Δ*λ; Δ*λ 0; -Δ*κ*λ 1]
        Φ = reshape(Φ, 2, 2, 2)
		U = T[1 0; 0 1]
        U = reshape(U, 1, 2, 2, 1)
		V = T[0 0 0 1 0 0 -κ 0; 0 -κ 0 0 1 0 0 0].*Δ
        V = reshape(V, 1, 2, 2, 4)
        ΦΦ = factormp(Φ, [], Φ, [])
		decpushfirst!(W, factorutcat(U, V, ΦΦ))
        ### 2cols
        Φ1 = T[0,Δ*λ]
        Φ1 = reshape(Φ1, 1, 2, 1)
		U = T[0 0; 0 1]
        U = reshape(U, 1, 2, 2, 1)
        V = zeros(T, 1, 2, 2, 2)
		V[1,:,2,:] .= T[0 1; -κ 0].*Δ
        ΦΦ = factormp(Φ, [], Φ1, [])
        Ψ = factorutcat(U, V, ΦΦ)
		decpushfirst!(W2, Ψ)
        ### 2cols2rows
        II = reshape(T[0 0; 0 1], 1, 2, 2, 1)
        decpushfirst!(W22, II)
        
        decpushfirst!(C, factormp(factormp(Ψ, 2, II, 1), 2, Ψ, 2))

		κ,λ = (1-Δ*λ^2)*κ,Δ*λ^2
	end
	U = T[1,0,0,0,0]; U = reshape(U, 1, 1, 1, 5)
	decpushfirst!(W, U)
    ### 2cols
	U2 = T[1,0,0]; U2 = reshape(U2, 1, 1, 1, 3)
	decpushfirst!(W2, U2)
    
    ### 2cols2rows
    U22 = ones(T, 1, 1, 1, 1)
	decpushfirst!(W22, U22)

    Q = factormp(factormp(U2, 2, U22, 1), 2, U2, 2)
    decpushfirst!(C, Q)
    
	return W,W2,W22,C
end

function invmassdd(::Type{T}, L::Int) where {T<:FloatRC}
    #=
    QTT decomposition of L cores of the padded inverse mass matrix
    with respect to the L^2-normalized nodal basis on (0,1).
    =#

    if L < 1
		throw(ArgumentError("L should be at least one"))
	end

    #= Sherman–Morrison–Woodbury formula =#
    W,_,_,C = invmassSMW(T,L)
    Minv = decadd(W,C)
    decskp!(Minv, 1; path="forward")  
    Minv = decscale!(Minv,T(6))  

    return Minv
end

###
T = Float64
L = 20
q = 5
###

@testset "BPX preconditioner" begin
    C = bpxdd(T, L, 1; major="first")
    r_C = decrank(C); r_C = r_C[2:end-1]
    ref_r_C = 13*ones(T,L-1,1)
    @test r_C ≈ ref_r_C rtol = 1e-16
end

@testset "bq^T ⊗ IL" begin
    _,bq,_ = A_b_c_Gauss(q)
    bq_QTT = permutedims(reshape(bq, 1, 1, q, 1), [2,1,3,4])
    IL = QTT_IL0(T,L)
    bq_IL = decpushfirst!(deepcopy(IL), bq_QTT) ## ℓ+1 factors
    r_bq_IL = decrank(bq_IL); r_bq_IL = r_bq_IL[2:end-1]
    ref_r_bq_IL = [1; 2*ones(T,L-1,1)]
    @test r_bq_IL ≈ ref_r_bq_IL rtol = 1e-16
end

@testset "Preconditioned stiffness" begin
    CAC = stiffnessbpxdd(T,L)
    r_CAC = decrank(CAC); r_CAC = r_CAC[2:end-1]
    ref_r_CAC = 169*ones(T,L-1,1)
    @test r_CAC ≈ ref_r_CAC rtol = 1e-16
end

@testset "Left-preconditioned stiffness CL*AL" begin
    CA = stiffnessbpxdd_left(T,L)
    r_CA = decrank(CA); r_CA = r_CA[2:end-1]
    ref_r_CA = 39*ones(T,L-1,1)
    @test r_CA ≈ ref_r_CA rtol = 1e-16
end

@testset "Aq ⊗ IL" begin
    Aq,_,_ = A_b_c_Gauss(q)
    Aq_QTT = permutedims(reshape(Aq, q, 1, q, 1), [2,1,3,4])
    IL = QTT_IL0(T,L)
    Aq_IL = decpushfirst!(deepcopy(IL), Aq_QTT) 
    r_Aq_IL = decrank(Aq_IL); r_Aq_IL = r_Aq_IL[2:end-1]
    ref_r_Aq_IL = [1; 2*ones(T,L-1,1)]
    @test r_Aq_IL ≈ ref_r_Aq_IL rtol = 1e-16
end

@testset "System matrix HL(q)" begin
    Lt = ceil(Int, L/q); τ = 2.0.^(-Lt)
    Iq = Matrix(1.0I, q, q)
    Iq_QTT = permutedims(reshape(Iq, q, 1, q, 1), [2,1,3,4])
    IL = QTT_IL0(T,L)
    Iq_IL = decpushfirst!(deepcopy(IL), Iq_QTT)
    Aq,_,_ = A_b_c_Gauss(q)
    Aq2 = Aq^2
    Aq2_QTT = permutedims(reshape(Aq2, q, 1, q, 1), [2,1,3,4])
    Minv = invmassdd(T,L)
    A = stiffnessdd(T,L)
    MiA = decmp(Minv, 2, A, 1)
    Aq2_MiA = decpushfirst!(deepcopy(MiA), Aq2_QTT)
    HLq = decaxpby(1.0, Iq_IL, τ^2, Aq2_MiA)
    r_HLq = decrank(HLq); r_HLq = r_HLq[2:end-1]
    ref_r_HLq = [2; 128*ones(T,L-1,1)]
    @test r_HLq ≈ ref_r_HLq rtol = 1e-16
end

@testset "Aq ⊗ inv(ML)*AL" begin
    Aq,_,_ = A_b_c_Gauss(q)
    Aq_QTT = permutedims(reshape(Aq, q, 1, q, 1), [2,1,3,4])
    Minv = invmassdd(T,L)
    A = stiffnessdd(T,L)
    MiA = decmp(Minv, 2, A, 1)
    Aq_MiA = decpushfirst!(deepcopy(MiA), Aq_QTT)
    r_Aq_MiA = decrank(Aq_MiA); r_Aq_MiA = r_Aq_MiA[2:end-1]
    ref_r_Aq_MiA = [1; 126*ones(T,L-1,1)]
    @test r_Aq_MiA ≈ ref_r_Aq_MiA rtol = 1e-16
end

@testset  "Iq ⊗ inv(ML)*AL" begin
    Iq = Matrix(1.0I, q, q)
    Iq_QTT = permutedims(reshape(Iq, q, 1, q, 1), [2,1,3,4])
    Minv = invmassdd(T,L)
    A = stiffnessdd(T,L)
    MiA = decmp(Minv, 2, A, 1)
    Iq_MiA = decpushfirst!(deepcopy(MiA), Iq_QTT)
    r_Iq_MiA = decrank(Iq_MiA); r_Iq_MiA = r_Iq_MiA[2:end-1]
    ref_r_Iq_MiA = [1; 126*ones(T,L-1,1)]
    @test r_Iq_MiA ≈ ref_r_Iq_MiA rtol = 1e-16
end



