# algorithme ADMM   
#### EXEMPLE 2 : taux de chomage ####

from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt

n = 2
r = 0.1
epsilon = 10**(-6)
#fonction sto 

a = np.array(n)

def sto_efficace(lamba,a):
 return np.sign(a) * np.maximum(0, np.abs(a) - lamba)

# donnée


# imoprtation 
d1 = np.loadtxt('C:/Users/rania/OneDrive/Documents/UnemploymentRateFR24x2.dat.txt') # matrice de donner dont la derriniere ligne est la matrice b et le rest est A

A = np.delete(d1,2,1)
print(A)
b = d1[:,-1]


x_k = np.zeros(n)
z_k = np.zeros(np.size(b))
u_k = np.zeros(np.size(b))
x= np.zeros(n) 
z = np.zeros(np.size(b)) 
u = np.zeros(np.size(b))
stop = 1
norminfini=1
k=0
while stop > epsilon and norminfini > 1e-10 and k < 5000:

    x_k = np.linalg.solve(np.transpose(A) @ A , np.transpose(A) @ (b + z_k - u_k))

    z_k = sto_efficace(1/r,A @ x_k - b + u_k )

    u_k = u_k + A @ x_k - z_k - b

    err_rel_x = la.norm(x - x_k)
    err_rel_z = la.norm(z - z_k)
    err_rel_u = la.norm(u-u_k)
    norm_x = la.norm(x_k)
    norm_z = la.norm(z_k)
    norm_u = la.norm(u_k)
    stop = (err_rel_x + err_rel_z + err_rel_u) / (norm_x+ norm_z + norm_u)
    x = x_k
    z = z_k
    u = u_k
    norminfini = la.norm(z-A@x + b )
    k= k+1
    print(k,'k')
    print(norminfini,'norminfini')
    print(stop,'stop')
    print (x_k,'x_k')
    print (z_k,'z_k')
    print (u_k,'u_k')


t = d1[:,1]
print(t)

plt.plot(t,x_k[0]+t* x_k[1],label='moindre valeur absolue')


# MOINDRE CARRÉ 
# on resout directement 
a1,b1 = np.linalg.lstsq(A,b)[0]
t = d1[:,1]#on tabule t
# X, Y, Z = [tmin1, tmax1], [a1 + tmin1 * b1, a1 + tmax1 * b1], [ac1 + tmin1 * bc1, ac1 + tmax1 * bc1]
# plt.figure()

plt.plot(t,b, 'o', label = "donné")
plt.plot(t,a1+t*b1,label='moindre carré')

plt.legend()
plt.title('comparaison moindre valeur absolue et moidre carré appliquer au taux de chomage')
plt.grid(True)
plt.show()
