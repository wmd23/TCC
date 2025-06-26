using LinearAlgebra

function armijo(x, d, f, gradf; η=.6, γ=.5)

    t = 1
    while f(x + t * d) > f(x) + η * t * dot(gradf(x),d)
        t = t * γ
    end

    return t

end