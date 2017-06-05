#!/usr/bin/python
#-*-coding:utf8-*-

from pylab import *

def J(W,C,Y):
    """
    Función de costos
    """
    E = 0.0
    m = len(C)
    for i in range(len(C)):
        y = Y[i]
        x = C[i]
        hp = sigmoide(W.T.dot(x));
        hn = sigmoide(W.T.dot(-x));
        e = -y*log(hp)-(1-y)*log(hn);
        E += e
    return (1./m)*E


def sigmoide(X):
    """
    Función sigmoide
    """
    num = 1.
    den = 1+exp(-X)
    return num/den


def h(W,X):
    q = X.shape
    if len(q)==1:
        Y = W.dot(X)
        return sigmoide(Y)
    else:
        Y = zeros((q[0],))
        for i in range(len(X)):
            Y[i] = W.dot(X[i])
        return sigmoide(Y)

def normalize(X):
    mu = X.mean(axis=0)
    Smin = amin(X,axis=0)
    Smax = amax(X,axis=0)
    x = (X-mu)/(Smax-Smin)
    return x


def paso_gradiente(Wa, C, Y):
    N = len(C)
    grad = zeros(Wa.size)
    n = grad.size
    delta = Y- sigmoide(Wa.dot(C.T))
    permutacion = range(N)
    shuffle(permutacion)
    for j in range(n-1):
        for i in permutacion:
            grad[j] += delta[i]*C[i,j]
    return -(1./N)*grad

def grad_desc( W0,P, Y, ca, iters, verbose=True):
    pesos = [W0[:]]
    costos = [J(W0,P,Y)]
    tamanos = [norm(W0)]
    for i in range(iters):
        pa = pesos[-1]
        grad = paso_gradiente(pa, P, Y)
        pa = pa - ca*grad
        costo = J(pa,P,Y)
        pesos.append(pa[:])
        costos.append(costo)
        tamanos.append(norm(grad))
        if verbose:
            print i,":",costo
    return pesos[-1], costos, tamanos

def ejemplo():
    P = genfromtxt('conjen_logit.csv', delimiter=',')
    X = P[:,1:3]
    Xn = normalize(X)
    Xn = append( ones( (len(Xn),1) ), Xn, axis=1 )
    Y = P[:,-1]
    ca = 0.3
    W0 = ranf((3,))
    iters = 10000
    error_i = J(W0, Xn, Y)
    print "Descenso en w0={0}, w1={1}, w2={2},  error={3}".format(W0[0],W0[1],W0[2], error_i)
    Wf, costos, tams = grad_desc( W0, Xn, Y, ca, iters )
    error_f = J(Wf,Xn, Y)
    print u"Después de {0} iteraciones W_0={1}, W_1={2}, W_2={3}, error={4}".format(iters, Wf[0],Wf[1],Wf[2] ,error_f)

if __name__ =='__main__':
    ejemplo()
