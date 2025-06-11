# Método de Newton puro aplicado no sistema Lagrange

## Exemplo I (uma restrição) - $\min f(x_1,x_2)=-x_1x_2$ sujeito a $x_1+x_2=2$

```julia
include("MetodoNewtonLagrange.jl")

f(x)=-x[1]*x[2]
h(x)=x[1]+x[2]-2
gradf(x)=[-x[2]; -x[1]]
gradh(x)=[1; 1]; Jacobian_h(x)=gradh(x)' # A matriz jacobiana de uma função escalar é gradiente transposto
hessf(x)=[0 -1; -1 0]
hessh(x)=[0 0; 0 0]; hess_coordenada_h=[hessh] # A função coordenada de uma função escalar é ela própria
x0=1.e3*rand(2); λ0=1.e3*rand() # Determinando o ponto inicial

x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100)
```

## Exemplo II (mais do que uma restrição) - $\min -x_1-2x_2-3x_3$ sujeito a $x_1-x_2+x_3=1$ e $x_1^2+x_2^2=1$

```julia
include("MetodoNewtonLagrange.jl")

f(x)=-x[1]-2*x[2]-3*x[3]
h(x)=[x[1]-x[2]+x[3]-1;x[1]^2+x[2]^2-1]
gradf(x)=[-1;-2;-3]
Jacobian_h(x)=[1 -1 1; 2x[1] 2x[2] 0]
hessf(x)=zeros(3,3)
hessh1(x)=hessf(x); hessh2(x)=[2 0 0; 0 2 0; 0 0 0]; hess_coordenada_h=[hessh1;hessh2]
x0=rand(3); λ0=rand(2)

x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100)
```

## Exemplo III - $\min f(x,y)=y^2-x^2$ sujeito a $\frac{1}{4x^2}+y^2=1$

```julia
include("MetodoNewtonLagrange.jl")

# Máximo f(0,+-1)=1, mínimo f(+-2,0)=-4
f(x)=x[2]^2-x[1]^2
h(x)=0.25*x[1]^2+x[2]^2-1
gradf(x)=[-2*x[1]; 2*x[2]]
gradh(x)=[0.5*x[1]; 2*x[2]]; Jacobian_h(x)=gradh(x)' # A matriz jacobiana de uma função escalar é gradiente transposto
hessf(x)=[-2 0; 0 2]
hessh(x)=[1 0; 0 2]; hess_coordenada_h=[hessh] # A função coordenada de uma função escalar é ela própria
x0=rand(2); λ0=rand() # Determinando o ponto inicial

x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=40000)
```

```julia
include("MetodoNewtonLagrange.jl")

# Máximo f(1,√2,-√2)=1+2√2 Mínimo f(1,√2,-√2)=1-2√2
f(x)=x[1]+2*x[2]
h(x)=[x[1]+x[2]+x[3]-1;x[2]^2+x[3]^2-4]
gradf(x)=[1;2;0]
Jacobian_h(x)=[1 1 1; 0 2*x[2] 2*x[3]]
hessf(x)=zeros(3,3)
hessh1(x)=hessf(x); hessh2(x)=[0 0 0; 0 2 0; 0 0 2]; hess_coordenada_h=[hessh1;hessh2]
x0=rand(3); λ0=rand(2)

x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100)
```
