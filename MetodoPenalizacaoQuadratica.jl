using LinearAlgebra

include("Metodos.jl")

function metodo_penalizacao_quadratica(y0,f,h,gradf,Jacobian_h,c; ϵ=1.e3) #c deve ser positivo
    
    k=0
    t=c                   
    phi(x;ck=c)=f(x)+0.5*ck*norm(h(x))^2
    gradphi(x;ck=c)=gradf(x)+ck*(Jacobian_h(x)'*h(x))
    x0,iter=metodo_gradiente(y0,phi,gradphi)

    if iter == -1
        return xk,-1
    end

    xk=x0
    t=1.5*t
    while t<ϵ
        phi(x)=phi(x,ck=t) 
        gradphi(x)=gradphi(x,ck=t)
        xk, iter = metodo_gradiente(xk,phi,gradphi)
        println(xk)
        if iter == -1
            return xk, -1
        end
        t=1.5*t
        k+=1
    end

    return xk, k

end