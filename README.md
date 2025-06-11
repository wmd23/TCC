# Método de Newton puro aplicado no sistema Lagrange

## Exemplo I (uma restrição) - $min f(x_1,x_2)=-x_1x_2$ sujeito a $x_1+x_2=2$

```julia
include("testfunctions.jl"); include("projections.jl")

x0 = ones(100) # guess
F = gradsumsquares # see testfunctions.jl for more details
proj = projRplus # see projections.jl for more details

x,k,t,nFx,Fevals,error=ding(x0,F,proj, maxiter=1.e4) # see search.jl for more details
```

#=
# Exemplo I (uma restrição) - min f(x1,x2)=-x1x2 sujeito a x1+x2=2
f(x)=-x[1]*x[2]
h(x)=x[1]+x[2]-2
gradf(x)=[-x[2]; -x[1]]
gradh(x)=[1; 1]; Jacobian_h(x)=gradh(x)' # A matriz jacobiana de uma função escalar é gradiente transposto
hessf(x)=[0 -1; -1 0]
hessh(x)=[0 0; 0 0]; hess_coordenada_h=[hessh] # A função coordenada de uma função escalar é ela própria
x0=1.e3*rand(2); λ0=1.e3*rand() # Determinando o ponto inicial

x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100)
x_barra2,λ_barra2,k2=metodo_newton_lagrange_num_diff(f, h, hess_coordenada_h, x0, λ0;maxiter=100)
=#

#=
#Exemplo II (mais do que uma restrição) - min -x1-2x2-3x3 sujeito a x1-x2+x3=1 e x1^2+x2^2=1
f(x)=-x[1]-2*x[2]-3*x[3]
h(x)=[x[1]-x[2]+x[3]-1;x[1]^2+x[2]^2-1]
gradf(x)=[-1;-2;-3]
Jacobian_h(x)=[1 -1 1; 2x[1] 2x[2] 0]
hessf(x)=zeros(3,3)
hessh1(x)=hessf(x); hessh2(x)=[2 0 0; 0 2 0; 0 0 0]; hess_coordenada_h=[hessh1;hessh2]
x0=rand(3); λ0=rand(2)
x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100)
x_barra2,λ_barra2,k2=metodo_newton_lagrange_num_diff(f, h, x0, λ0;maxiter=100)
=#

# include("MetodoPenalizacaoQuadratica.jl")

# y0=1.e3*rand(2)
# f(x)=-x[1]*x[2]
# h(x)=x[1]+x[2]-2
# gradf(x)=[-x[2]; -x[1]]
# hessh(x)=[0 0; 0 0]
# c=1
# x_barra=metodo_penalizacao_quadratica(y0,f,h,gradf,hessh,c,ϵ=1.e3)

#=
# Exemplo I (uma restrição) - min f(x1,x2)=x_1²+x_2² sujeito a x1*x2=1 (número máximo de iteradas atingido)
f(x)=x[1]^2+x[2]^2
h(x)=x[1]*x[2]-1
gradf(x)=[2*x[1]; 2*x[2]]
gradh(x)=[x[2]; x[1]]; Jacobian_h(x)=gradh(x)' # A matriz jacobiana de uma função escalar é gradiente transposto
hessf(x)=[2 0; 0 2]
hessh(x)=[0 0; 0 0]; hess_coordenada_h=[hessh] # A função coordenada de uma função escalar é ela própria
x0=[1.1; 0.8]; λ0=-2 # Determinando o ponto inicial
x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=40000)
=#

# Exemplo II - min f(x,y)=y²-x² sujeito a 1/4x²+y²=1 14.8 Ex:5
# Máximo f(0,+-1)=1, mínimo f(+-2,0)=-4
f(x)=x[2]^2-x[1]^2
h(x)=0.25*x[1]^2+x[2]^2-1
gradf(x)=[-2*x[1]; 2*x[2]]
gradh(x)=[0.5*x[1]; 2*x[2]]; Jacobian_h(x)=gradh(x)' # A matriz jacobiana de uma função escalar é gradiente transposto
hessf(x)=[-2 0; 0 2]
hessh(x)=[1 0; 0 2]; hess_coordenada_h=[hessh] # A função coordenada de uma função escalar é ela própria
x0=rand(2); λ0=rand() # Determinando o ponto inicial
x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=40000)

# Exemplo III - min f(x,y,z)=x+2y sujeito a x+y+z=1 e y²+z²=4 14.8 Ex:15
# Máximo f(1,√2,-√2)=1+2√2 Mínimo f(1,√2,-√2)=1-2√2
f(x)=x[1]+2*x[2]
h(x)=[x[1]+x[2]+x[3]-1;x[2]^2+x[3]^2-4]
gradf(x)=[1;2;0]
Jacobian_h(x)=[1 1 1; 0 2*x[2] 2*x[3]]
hessf(x)=zeros(3,3)
hessh1(x)=hessf(x); hessh2(x)=[0 0 0; 0 2 0; 0 0 2]; hess_coordenada_h=[hessh1;hessh2]
x0=rand(3); λ0=rand(2)
x_barra1,λ_barra1,k1=metodo_newton_lagrange(h, gradf, Jacobian_h, hessf, hess_coordenada_h, x0, λ0;maxiter=100)
