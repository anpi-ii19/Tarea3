from relajaciontavo import *

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


n= 10
A=np.diag(np.ones(n),0)*6+np.diag(np.ones(n-1),-1)*2+np.diag(np.ones(n-1)*2,1)
b=np.ones(n)*15
b[0]=12
b[-1]=12


print(A)
print(b)
print(relajacion(A, np.transpose([b]), 10 ** -10))

