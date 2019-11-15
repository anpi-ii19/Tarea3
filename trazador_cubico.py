from relajacion import *
import numpy as np

def evaluar_trazador_cubico(x_eval,x,a,b,c,d):
    """
    Funcion evaluar un trazador cubico sobre un vector x.
    :param x_eval: Vector sobre el que sera evaluado el trazador cubico
    :param x: vector de pre-imagenes con el que se construyo el trazador cubico
    :param a: vector de imagenes con el que se construyo el trazador cubico
    :param b: Vector b generado por el metedo del trazador cubico
    :param c: Vector c generado por el metedo del trazador cubico
    :param d: Vector d generado por el metedo del trazador cubico

    :return: Numpy array correspindiente a S(x_eval)
    """
    y=np.zeros(len(x_eval))
    for i in range(0, len(x_eval)):
        flag=False
        for j in range(0, len(x)-1):
            if( (x[j]<=x_eval[i]) and ( x_eval[i]<=x[j+1])):
                y[i]=a[j]+b[j]*(x_eval[i]-x[j])+ c[j]*(x_eval[i]-x[j])**2+d[j]*(x_eval[i]-x[j])**3
                flag=True
                break
        if(not flag):
            raise ValueError(x_eval[i],"esta fuera de rango")
    return y
            
def trazador_cubico(x,y,tol=10**-10):
    """
    Funcion evaluar un trazador cubico sobre un vector x.
    :param x: vector de pre-imagenes 
    :param y: vector de imagenes f(x)
    
    :return: Funcion correspindiente al trazador cubico S(x)
    """
    x= np.array(x)
    y=np.array(y)
    # evaluar que x y y tengan el mismo tamano
    if(len(x)!= len(y)):
        raise ValueError("x y y tienen dimensiones diferentes")
    n = len(x)-1
    h= x[1:] - x[:-1] # h[i]= x[i+1]-x[i] (i=0,1,2,...,n-1)
    
    alpha=np.append(np.append(0,(3/h[1:])*(y[2:]-y[1:-1])-(3/h[:-1])*(y[1:-1]-y[:-2])),0)
    
    # contruir matriz tridiagonal A
    upper=np.append(0,h[1:])
    lower=np.append(h[:-1],0)
    center=np.append(np.append(1,2*(h[:-1]+h[1:] )),1)
    A=np.diag(center)+np.diag(upper,1)+np.diag(lower,-1)
    
    # resolver la matrix A*x=alpha mediante el metodo de relajacion(sor)
    c= np.asarray(np.transpose(relajacion(A, np.transpose([alpha]), tol)))[0]
    b= np.zeros(n+1)
    d=np.zeros(n+1)
    #construir los vectores b y d
    for i in reversed(range(0,n)):
        b[i]=(y[i+1]-y[i])/h[i]- h[i]*(c[i+1]+2*c[i])/3
        d[i]=(c[i+1]-c[i])/(3*h[i])
    f = lambda x_eval : evaluar_trazador_cubico(x_eval,x,y,b,c,d)
    return f