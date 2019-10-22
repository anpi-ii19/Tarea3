import numpy as np
import sympy as sp


def verificar_simetria(matriz):
    """
    Metodo que verifica si una matriz es simetrica
    :param matriz: Numpy Matrix que debe ser verificada
    :return: True si la matriz es simetrica, False en caso contrario
    """
    # Se verifica que las entradas cumplen que Ai,j = Aj,i (simetria)
    n, m = matriz.shape  # Dimensiones de la matriz
    for i in range(0, n):
        for j in range(0, m):
            elemento1 = matriz[i, j]
            elemento2 = matriz[j, i]

            # Si elemento1 es diferente de elemento2 la matriz no es simetrica
            if elemento1 != elemento2:
                return False

    return True


def verificar_positiva_definida(matriz):
    """
    Funcion para verificar que una matriz es positiva definida
    :param matriz: Numpy Matrix que debe ser verificada
    :return: True si la matriz es positiva definida, False en caso contrario
    """
    n = matriz.shape[0]  # Cantidad de filas de la matriz
    for i in range(1, n + 1):
        sub_matriz = matriz[:i, :i]
        det = np.linalg.det(sub_matriz)

        # Determinante de la submatriz negativo, matriz no es positivida definida
        if det < 0:
            return False

    return True


def calcular_autovalores(matriz):
    """
    Funcion para calcular los autovalores de una matriz
    :param matriz: Numpy Matrix a la cual calcularle los autovalores
    :return: Lista con los autovalores encontrados
    """
    n = matriz.shape[0]  # Cantidad de filas
    identidad = sp.Matrix(sp.eye(n))  # Matriz identidad
    sim_l = sp.Symbol('l')  # Variable simbolica
    matriz = matriz - sim_l * identidad
    det = matriz.det()  # Calculo del determinante

    # Se encuentran las soluciones del determinante igualado a cero
    auto_val = sp.solve(det, sim_l)

    return auto_val


def calcular_omega_optimo(matriz_l, matriz_d, matriz_u):
    """
    Funcion que encuentra el omega optimo para el metodo de relajacion
    :param matriz_u: Numpy Matrix triangular superior de la matriz A
    :param matriz_d: Numpy Matrix diagonal de la matriz A
    :param matriz_l: Numpy Matrix triangular inferior de la matriz A
    :return: Float, omega optimo calculado
    """
    # Se calcula D**-1 * -(L + U)
    matriz = np.linalg.inv(matriz_d) * (-matriz_l - matriz_u)

    # Se calculan los autovalores de la matriz
    auto_val = calcular_autovalores(matriz)

    # Se obtiene el maximo autovalor en valor absoluto
    max_autoval = abs(max(auto_val, key=abs))

    # Se calcula el omega optimo
    omega_opt = 2 / (1 + (1 - (max_autoval ** 2)) ** 0.5)

    return float(omega_opt)


def calcular_matriz_diagonal(matriz):
    """
    Funcion para obtener la matriz diagonal de una matriz ingresada
    :param matriz: Numpy Matrix de la cual obtener la matriz diagonal
    :return: Numpy Matrix con la diagonal obtenida
    """
    n = matriz.shape[0]  # Cantidad de filas de matriz

    # Se crea una matriz con ceros
    diagonal = np.matrix(np.zeros((n, n)))

    # Se cambian los valores en la diagonal por los valores en la matriz ingresada
    for i in range(0, n):
        diagonal[i, i] = matriz[i, i]

    return diagonal


def calcular_triangular_superior(matriz):
    """
    Funcion para calcular la matriz triangular superior de una matriz ingresada
    :param matriz: Numpy Matrix de la cual obtener la triangular superior
    :return: Numpy Matrix con la triangular superior obtenida
    """
    n = matriz.shape[0]  # Cantidad de filas de matriz

    # Se crea una matriz con ceros
    t_superior = np.matrix(np.zeros((n, n)))

    # Se cambian los valores en la triangular superior por los valores en la matriz
    for i in range(0, n - 1):
        t_superior[i,  i + 1:] = matriz[i, i + 1:]

    return t_superior


def calcular_triangular_inferior(matriz):
    """
    Funcion para calcular la matriz triangular inferior de una matriz ingresada
    :param matriz: Numpy Matrix de la cual obtener la triangular inferior
    :return: Numpy Matrix con la triangular inferior obtenida
    """
    n = matriz.shape[0]  # Cantidad de filas de matriz

    # Se crea una matriz con ceros
    t_inferior = np.matrix(np.zeros((n, n)))

    # Se cambian los valores en la triangular inferior por los valores en la matriz
    for i in range(1, n):
        t_inferior[i, :i] = matriz[i, :i]

    return t_inferior


def relajacion(matriz_a, matriz_b, tol):
    """
    Funcion que implementa el metodo de relajacion para encontrar la solucion de un sistema
    de la forma A x = b
    :param matriz_a: Numpy Matrix o matriz con listas genericas, matriz A del sistema A x = b
    :param matriz_b: Numpy Matrix o matriz con listas genericas, matriz b del sistema A x = b
    :param tol: Numero mayor a cero, corresponde a la tolerancia al fallo que debe tener el resultado
    :return: Numpy Matrix, matriz x que resuelve el sistema A x = b
    """
    # Se convierten las matrices ingresadas a matrices de Numpy
    matriz_a = np.asmatrix(matriz_a)
    matriz_b = np.asmatrix(matriz_b)

    # Dimensiones de la matriz_a
    n, m = matriz_a.shape

    # Se  verifica que la matriz_a sea cuadrada
    if n != m:
        return "La matriz_a debe ser cuadrada"

    if tol < 0:
        return "El parametro tol debe un numero mayor a cero"

    # Se verifica que la matriz_a sea simetrica
    simetria = verificar_simetria(matriz_a)
    if not simetria:
        return "La matriz_a debe ser simetrica"

    # Se verifica que la matriz_a sea positiva definida
    positiva_definida = verificar_positiva_definida(matriz_a)
    if not positiva_definida:
        return "La matriz_a debe ser positiva definida"

    # Se calcula la descomposicion A = L + D + U
    matriz_l = calcular_triangular_inferior(matriz_a)
    matriz_d = calcular_matriz_diagonal(matriz_a)
    matriz_u = calcular_triangular_superior(matriz_a)

    omega = calcular_omega_optimo(matriz_l, matriz_d, matriz_u)

    try:
        # Se calcula el inverso de la matriz (D + w * L)
        matriz_dwl_inv = np.linalg.inv(matriz_d + omega * matriz_l)
    except:
        return "El termino (D + w * L) debe ser invertible"

    # Se calculan las matrices Br y Cr
    matriz_br = matriz_dwl_inv * ((1 - omega) * matriz_d - omega * matriz_u)
    matriz_cr = omega * matriz_dwl_inv * matriz_b

    # Matriz x anterior que almacena el resultado de la iteracion anterior
    matriz_x_ant = np.matrix(np.zeros((n, 1)))

    while 1:
        # Calculo de la matriz x de la iteracion actual
        matriz_x = matriz_br * matriz_x_ant + matriz_cr

        # Calculo del error relativo de la iteracion actual
        numerador = np.linalg.norm(matriz_x - matriz_x_ant)
        denominador = np.linalg.norm(matriz_x)
        error = numerador / denominador

        # Se verifica la condicion de parada
        if error < tol:
            break

        matriz_x_ant = matriz_x.copy()

    return matriz_x


# A = np.matrix([[4, 3, 0], [3, 4, -1], [0, -1, 4]])
# b = np.matrix([[7], [7], [-1]])
# x = relajacion(A, b, 10 ** -10)
# print(A * x)

# A = [[10, -1, 0], [-1, 10, -2], [0, -2, 10]]
# b = [[9], [7], [6]]
# x = relajacion(A, b, 10 ** -10)
# print(A * x)

# A = [[3, -1, 1], [-1, 3, -1], [1, -1, 3]]
# b = [[-1], [7], [-7]]
# x = relajacion(A, b, 10 ** -10)
# print(A * x)
