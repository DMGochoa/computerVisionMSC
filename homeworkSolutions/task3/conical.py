"""This module demonstrates the creation and usage of conics class
on the P2 space to fit a conic from N points.

Diego A Moreno G
MSc Student in Electrical Engineering
Universidad Tecnológica de Pereira
20/08/2023
"""
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
from geometry.conicalGP2 import ConicalGP2
# List to store the coordinates of the points
collected_points = []

def onclick(event):
    global collected_points
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        collected_points.append((x, y))
        plt.scatter(x, y, c='red')
        plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        plt.draw()

# Create a figure and connect it to the click event
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
plt.grid(True)
plt.title('Please click on the canvas to collect points to fit a conic (close the window when you are done)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Connect the click event with the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Create a ConicalGP2 object
conic_fitted = ConicalGP2(puntos=collected_points)

# Display the conic's equation
print(f"Equation of the conic: {conic_fitted.conic}")

# Plot the conic using the class method
conic_fitted.plot()

# Plot the collected points
for point in collected_points:
    plt.scatter(*point, c='red')
    plt.annotate(f'({point[0]:.2f}, {point[1]:.2f})', (point[0], point[1]), textcoords="offset points", xytext=(0,10), ha='center')

# Set up the plot
plt.grid(True)
plt.title('Fitted Conic and Collected Points')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.pause(6)