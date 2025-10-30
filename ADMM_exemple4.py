# algorithme ADMM   
#### EXEMPLE  4 : prix de vente  ####

from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt

r = 0.1
epsilon = 10**(-6)

#fonction sto 


def sto_efficace(lamba,a):
 return np.sign(a) * np.maximum(0, np.abs(a) - lamba)

# donnée


# imoprtation 
d2 = np.loadtxt('C:/Users/rania/OneDrive/Documents/SellingPrice28x11.dat.txt') # matrice de donner dont la derriniere ligne est la matrice b et le rest est A
n2 = len(d2)
A = np.concatenate((np.ones((n2, 1)), d2[:, 0:-1]), axis=1)
b = d2[:, -1].reshape(n2, 1) 
n = n2

a = np.array(n)


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
    # print(k,'k')
    # print(norminfini,'norminfini')
    # print(stop,'stop')
    # print (x_k,'x_k')
    # print (z_k,'z_k')
    # print (u_k,'u_k')
    
print('résultat avec ADMM:')
print('b=',x_k[0][0])
print('a=',x_k[1:,0])

# # MOINDRE CARRÉ 

# beta est l'estimateur qu'on a deduit des donner 
beta = np.linalg.lstsq(A, b, rcond=None)[0]
print('résultat avec les moindres carrés:')
print('b=',beta[0])
print('a=',beta[1:])


# estimation 
b_esti = A @ beta
b_esti_ADMM = A @x_k[:,0]
# Affichage des résultats
print("Coefficients beta :")
print(beta)


# Tracé des valeurs observées vs prédites
plt.scatter(b, b_esti, color='blue',label = 'estimateur moindre carré')
plt.scatter(b, b_esti_ADMM, color='red',label = 'estimateur ADMM')
plt.plot([min(b), max(b)], [min(b), max(b)], color='black', linestyle='--', label='x=y')
plt.xlabel("Valeurs réelles ")
plt.ylabel("Valeurs estimées ")
plt.title("Commparaison des deux estimateurs")
plt.legend()
plt.grid(True)
plt.show()

normb = np.linalg.norm(b)

normbestimoindrecarré = np.linalg.norm(b-b_esti)
normbestimoindrecarréord1 = np.linalg.norm(b-b_esti,ord=1)
normbestiADMM= np.linalg.norm(b-b_esti_ADMM)
normbestiADMMord1= np.linalg.norm(b-b_esti_ADMM,ord=1)
print(normbestiADMM,'norme 2 de la différence entre estimateur ADMM et la valeur réel' )
print(normbestimoindrecarré,'norme 2 de la différence entre estimateur moidre carré et la valeur réel' )
print(normbestiADMMord1,'norme 1 de la différence entre estimateur ADMM et la valeur réel' )
print(normbestimoindrecarréord1,'norme 1 de la différence entre estimateur moindre carré et la valeur réel' )

