from trazador_cubico import *
from matplotlib import pyplot as plt 

x_ref = np.linspace(-5, 5, num=21)
y_ref = [0,0.0707,0,-0.0909,0,0.1273,0,-0.2122,0,0.6366,1,0.6366,0,-0.2122,0,0.1273,0,-0.0909,0,0.0707,0]
print("X_n:")
print(x_ref)
print("Y_n:")
print(y_ref)


sc = trazador_cubico(x_ref,y_ref,tol=1)
x_sc=np.linspace(-5, 5, num=100)
y_sc=sc(x_sc)


fig, ax = plt.subplots()
ax.plot(x_ref, y_ref, 'bo', label='Pares Ordenados')
# ax.plot(x_f, y_f, 'k--', label='fun')
ax.plot(x_sc, y_sc, 'k:', label='Trazador Cubico')

legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()