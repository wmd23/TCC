using LinearAlgebra

include("DerivadasNuméricas.jl")

# h - função de restrição
# gradf - gradiente da função objetivo
# Jacobian_h - matriz jacobiana da função de restrição
# hessf - hessiana da função objetivo
# hess_coordenada_h - vetor contendo as hessianas das funções coordenadas da função de restrição
# (x0,λ0) - ponto inicial

function metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100,tol=1.e-6)
    xk=x0
    λk=λ0
    n=length(x0)
    l=length(λ0)
    k=0

    # gradiente da função Lagrangiana
    gradL(x, λ) = [gradf(x) + Jacobian_h(x)'*λ; h(x)] 

    # hessiana da função Lagrangiana
    hessL(x, λ) = [hessf(x)+sum(λ[i]*hess_coordenada_h[i](x) for i in 1:l) Jacobian_h(x)'; Jacobian_h(x) zeros(l,l)]

    while !(norm(gradL(xk, λk)) < tol) && k < maxiter
        # Resolvendo o sistema linear
        new_point=[xk; λk]-inv(hessL(xk, λk))*gradL(xk, λk)

        # Atualizando x e λ
        xk = new_point[1:n]
        λk = new_point[n+1:end]

        if l==1 # Evitando erro na linha
            λk=λk[1]
        end

        # Atualizando o número de iterações
        k+=1
    end

    return xk, λk, k
end


# Implementação do método de Newton-Lagrange com derivadas numéricas
function metodo_newton_lagrange_num_diff(f, h, x0, λ0;maxiter=100,tol=1.e-6)
    xk=x0; λk=λ0 # Inicializando as sequências xk e λk que esperam convergir para x_barra e λ_barra
    n=length(x0)
    l=length(λ0)

    # Definindo as funções de derivadas numéricas
    gradf(x)=gradf_barra_fd(f,x)
    Jacobian_h(x)=jacobianf_barra_fd(h,x)
    hessf(x)=hessf_barra(f,x)
    hess_coordenada_h=[]
    m=length(h(x0))
    
    # Dimensionando o vetor hess_coordenada_h
    for i in 1:m
        g(x)=h(x)[i] # Determinando as funções coordenadas
        push!(hess_coordenada_h,g)
    end
    k=0

    # gradiente da função Lagrangiana
    gradL(x, λ) = [gradf(x) + Jacobian_h(x)'*λ; h(x)] 

    # hessiana da função Lagrangiana
    hessL(x, λ) = [hessf(x)+sum(λ[i]*hessf_barra(hess_coordenada_h[i],x) for i in 1:l) Jacobian_h(x)'; Jacobian_h(x) zeros(l,l)]

    while !(norm(gradL(xk, λk)) < tol) && k < maxiter
        # Resolvendo o sistema linear
        new_point=[xk; λk]-inv(hessL(xk, λk))*gradL(xk, λk)

        # Atualizando x e λ
        xk = new_point[1:n]
        λk = new_point[n+1:end]

        if l==1 # Evitando erro na linha
            λk=λk[1]
        end

        # Atualizando o número de iterações
        k+=1
    end

    return xk, λk, k
end