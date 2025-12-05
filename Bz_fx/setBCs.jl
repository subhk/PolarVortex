using LinearAlgebra
using Printf


function ImplementBCs_Dirchilet_on_D1(𝒟ᶻᴰ::Matrix{T}, z::Vector{T}; 
            order_accuracy::Int) where T
    N   = length(z)
    del = z[2] - z[1] 
    if order_accuracy == 4
        𝒟ᶻᴰ[1,:] .= 0.0;              𝒟ᶻᴰ[1,1] = -(1/12)/del;
        𝒟ᶻᴰ[1,2]  = (2/3)/del;        𝒟ᶻᴰ[1,3] = -(1/12)/del;

        𝒟ᶻᴰ[2,:] .= 0.0;              𝒟ᶻᴰ[2,1] = -(2/3)/del;
        𝒟ᶻᴰ[2,2]  = 0.0;              𝒟ᶻᴰ[2,3] = (2/3)/del;
        𝒟ᶻᴰ[2,4]  = -(1/12)/del;

        𝒟ᶻᴰ[N,  :] .= -1.0 .* 𝒟ᶻᴰ[1,:];               
        𝒟ᶻᴰ[N-1,:] .= -1.0 .* 𝒟ᶻᴰ[2,:];          
    elseif order_accuracy == 2
        𝒟ᶻᴰ[1,:] .= 0;                      
        𝒟ᶻᴰ[1,2]  = 0.5/del;         

        𝒟ᶻᴰ[N,:] .= -1.0 .* 𝒟ᶻᴰ[1,:];                           
    else
        error("Invalid order of accuracy")
    end
    return 𝒟ᶻᴰ
end

function ImplementBCs_Dirchilet_on_D2(𝒟²ᶻᴰ::Matrix{T}, z::Vector{T}; 
            order_accuracy::Int) where T
    N   = length(z)
    del = z[2] - z[1] 
    if order_accuracy == 4
        𝒟²ᶻᴰ[1,:] .= 0;
        𝒟²ᶻᴰ[1,1]  = -2/del^2;         𝒟²ᶻᴰ[1,2] = 1/del^2;  
    
        𝒟²ᶻᴰ[2,:] .= 0;                𝒟²ᶻᴰ[2,1] = (4/3)/del^2; 
        𝒟²ᶻᴰ[2,2]  = -(5/2)/del^2;     𝒟²ᶻᴰ[2,3] = (4/3)/del^2;
        𝒟²ᶻᴰ[2,4]  = -(1/12)/del^2;      

        𝒟²ᶻᴰ[N,  :] .= 1.0 .* 𝒟²ᶻᴰ[1,:];
        𝒟²ᶻᴰ[N-1,:] .= 1.0 .* 𝒟²ᶻᴰ[2,:];  
    elseif order_accuracy == 2
        𝒟²ᶻᴰ[1,:] .= 0;
        𝒟²ᶻᴰ[1,1]  = -2.0/del^2;         
        𝒟²ᶻᴰ[1,2]  = 1.0/del^2;  

        𝒟²ᶻᴰ[N,:] .= 1.0 .* 𝒟²ᶻᴰ[1,:];
    else
        error("Invalid order of accuracy")
    end
    return 𝒟²ᶻᴰ
end

function ImplementBCs_Dirchilet_on_D4(𝒟⁴ᶻᴰ::Matrix{T}, z::Vector{T}; 
        order_accuracy::Int) where T
    N   = length(z)
    del = z[2] - z[1] 
    if order_accuracy == 4
        𝒟⁴ᶻᴰ[1,:] .= 0;                  𝒟⁴ᶻᴰ[1,1] = 5/del^4;
        𝒟⁴ᶻᴰ[1,2]  = -4/del^4;           𝒟⁴ᶻᴰ[1,3] = 1/del^4;
        
        𝒟⁴ᶻᴰ[2,:] .= 0;                  𝒟⁴ᶻᴰ[2,1] = -(38/6)/del^4;
        𝒟⁴ᶻᴰ[2,2]  = (28/3)/del^4;       𝒟⁴ᶻᴰ[2,3] = -(13/2)/del^4;
        𝒟⁴ᶻᴰ[2,4]  = 2/del^4;            𝒟⁴ᶻᴰ[2,5] = -(1/6)/del^4;
        
        𝒟⁴ᶻᴰ[3,:] .= 0;                  𝒟⁴ᶻᴰ[3,1] = 2/del^4;
        𝒟⁴ᶻᴰ[3,2]  = -(13/2)/del^4;      𝒟⁴ᶻᴰ[3,3] = (28/3)/del^4;
        𝒟⁴ᶻᴰ[3,4]  = -(13/2)/del^4;      𝒟⁴ᶻᴰ[3,5] = 2/del^4;
        𝒟⁴ᶻᴰ[3,6]  = -(1/6)/del^4;
        
        𝒟⁴ᶻᴰ[N,  :] .= 1.0 .* 𝒟⁴ᶻᴰ[1,:];                 
        𝒟⁴ᶻᴰ[N-1,:] .= 1.0 .* 𝒟⁴ᶻᴰ[2,:]; 
        𝒟⁴ᶻᴰ[N-2,:] .= 1.0 .* 𝒟⁴ᶻᴰ[3,:]; 
    elseif order_accuracy == 2
        𝒟⁴ᶻᴰ[1,:] .= 0;                  𝒟⁴ᶻᴰ[1,1] = 5.0/del^4;
        𝒟⁴ᶻᴰ[1,2]  = -4.0/del^4;         𝒟⁴ᶻᴰ[1,3] = 1.0/del^4;
 
        𝒟⁴ᶻᴰ[2,:] .= 0;                  𝒟⁴ᶻᴰ[2,1] = -4.0/del^4;
        𝒟⁴ᶻᴰ[2,2]  = 6.0/del^4;          𝒟⁴ᶻᴰ[2,3] = -4.0/del^4;
        𝒟⁴ᶻᴰ[2,4]  = 1.0/del^4;     
        
        𝒟⁴ᶻᴰ[N,  :] .= 1.0 .* 𝒟⁴ᶻᴰ[1,:];
        𝒟⁴ᶻᴰ[N-1,:] .= 1.0 .* 𝒟⁴ᶻᴰ[2,:];      
    else
        error("Invalid order of accuracy")
    end
    return 𝒟⁴ᶻᴰ
end


function ImplementBCs_Neumann_on_D1(𝒟ᶻᴺ::Matrix{T}, z::Vector{T}; 
            order_accuracy::Int) where T
    N   = length(z)
    del = z[2] - z[1] 
    if order_accuracy == 4
        𝒟ᶻᴺ[1,:]    .= 0;              𝒟ᶻᴺ[1,1] = -1/del;
        𝒟ᶻᴺ[1,2]     = 1/del;         
    
        𝒟ᶻᴺ[2,:]    .= 0;              𝒟ᶻᴺ[2,1] = -(7/12)/del;
        𝒟ᶻᴺ[2,2]     = 0;              𝒟ᶻᴺ[2,3] = (2/3)/del;
        𝒟ᶻᴺ[2,4]     = -(1/12)/del;

        𝒟ᶻᴺ[N,  :]  .= -1.0 .* 𝒟ᶻᴺ[1,:];              
        𝒟ᶻᴺ[N-1,:]  .= -1.0 .* 𝒟ᶻᴺ[2,:];             
    elseif order_accuracy == 2
        𝒟ᶻᴺ[1,:]  .= 0;              
        𝒟ᶻᴺ[1,1]   = -0.5/del;
        𝒟ᶻᴺ[1,2]   = 0.5/del;         

        𝒟ᶻᴺ[N,:]  .= -1.0 .* 𝒟ᶻᴺ[1,:];      
    else
        error("Invalid order of accuracy")
    end
    return 𝒟ᶻᴺ
end

function ImplementBCs_Neumann_on_D2(𝒟²ᶻᴺ::Matrix{T}, z::Vector{T}; 
            order_accuracy::Int) where T
    N   = length(z)
    del = z[2] - z[1] 
    if order_accuracy == 4
        𝒟²ᶻᴺ[1,:] .= 0;                  𝒟²ᶻᴺ[1,1] = -1/del^2;
        𝒟²ᶻᴺ[1,2]  = 1/del^2;         
    
        𝒟²ᶻᴺ[2,:] .= 0;                  𝒟²ᶻᴺ[2,1] = (15/12)/del^2;
        𝒟²ᶻᴺ[2,2]  = -(5/2)/del^2;       𝒟²ᶻᴺ[2,3] = (4/3)/del^2;
        𝒟²ᶻᴺ[2,4]  = -(1/12)/del^2;

        𝒟²ᶻᴺ[N,:]   .= 1.0 .* 𝒟²ᶻᴺ[1,:];                 
        𝒟²ᶻᴺ[N-1,:] .= 1.0 .* 𝒟²ᶻᴺ[2,:];  
    elseif order_accuracy == 2
        𝒟²ᶻᴺ[1,:]   .= 0;                  
        𝒟²ᶻᴺ[1,1]    = -1.0/del^2;
        𝒟²ᶻᴺ[1,2]    = 1.0/del^2;
        
        𝒟²ᶻᴺ[N,:]   .= 1.0 .* 𝒟²ᶻᴺ[1,:]; 
    else
        error("Invalid error of accuracy")
    end
    return 𝒟²ᶻᴺ
end


function setBCs(𝒟::Matrix{T}, z::Vector{T}; 
            order_derivate::Int, order_accuracy::Int, bc_type::String) where T
    if bc_type == "dirchilet"
        if order_derivate == 1
            𝒟₁ = ImplementBCs_Dirchilet_on_D1(𝒟, z, order_accuracy=order_accuracy)
        elseif order_derivate == 2
            𝒟₁ = ImplementBCs_Dirchilet_on_D2(𝒟, z, order_accuracy=order_accuracy)
        elseif order_derivate == 4
            𝒟₁ = ImplementBCs_Dirchilet_on_D4(𝒟, z, order_accuracy=order_accuracy)
        else
            error("invalid order of derivative")
        end
    elseif bc_type == "neumann"
        if order_derivate == 1
            𝒟₁ = ImplementBCs_Neumann_on_D1(𝒟, z, order_accuracy=order_accuracy)
        elseif order_derivate == 2
            𝒟₁ = ImplementBCs_Neumann_on_D2(𝒟, z, order_accuracy=order_accuracy)
        else
            error("invalid order of derivative")
        end
    else
        error("Invalid bc type")
    end
    return 𝒟₁
end