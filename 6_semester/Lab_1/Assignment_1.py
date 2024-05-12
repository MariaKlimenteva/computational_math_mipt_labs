import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the reaction rate constants
k1 = 0.04
k2 = 3e7
k3 = 1e4

# Define the system of ODEs
def dydt(t, y):
    y1, y2, y3 = y
    dy1dt = -k1 * y1**2 + k3 * y2 * y3
    dy2dt = k1 * y1**2 - k2 * y2**2 - k3 * y2 * y3
    dy3dt = k2 * y2**2
    return [dy1dt, dy2dt, dy3dt]

# Define the Butcher tableau for the implicit Runge-Kutta method
A = np.array([[1/2, 0], [-1/2, 2]])
b = np.array([1/2, 3/2])
c = np.array([-1/2, 3/2])

# Define the initial conditions
y0 = [1, 0, 0]

# Define the time span
t_span = (0, 0.3)

# Solve the system of ODEs using solve_ivp with the Radau method
sol = solve_ivp(dydt, t_span, y0, method='Radau', jac=None, dense_output=True)


t = sol.t
y1, y2, y3 = sol.y


plt.plot(t, y1, label='A')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration of Substances over Time')
plt.legend()
plt.grid(True)
plt.show()
plt.plot(t, y2, label='B')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration of Substances over Time')
plt.legend()
plt.grid(True)
plt.show()
plt.plot(t, y3, label='C')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration of Substances over Time')
plt.legend()
plt.grid(True)
plt.show()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration of Substances over Time')
plt.legend()
plt.grid(True)
# plt.show()