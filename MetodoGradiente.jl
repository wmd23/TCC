using LinearAlgebra

include("BuscasLineares.jl") # Inclui o arquivo com as funções de busca linear

function metodo_gradiente(x0, f, gradf; ϵ=1.e-6, maxiter=1.e6)

    xk = copy(x0) 
    dk = -gradf(xk)
    k=0

    while true

        if norm(dk) < ϵ
            return xk, k
        end

        tk = armijo(xk,dk,f,gradf)
        xk = xk + tk * dk
        dk = -gradf(xk)
        k = k + 1

        if k>maxiter
            return xk, -1
        end

    end

end