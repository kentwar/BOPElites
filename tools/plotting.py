import matplotlib.pyplot as plt
import numpy as np

# Creating a swirl function
def swirl_function(x, y):
    # Convert to polar coordinates
    adj = 0.2
    r = np.sqrt((x+adj)**2 + (y-adj)**2)
    theta = np.arctan2((y-adj), (x+adj))
    
    # Swirl pattern
    z = np.exp(-r) * np.sin(4 * np.pi * r + 4 * theta)
    return z

# Define the grid
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = swirl_function(X, Y)

# # Plot
# plt.figure(figsize=(8, 6))
# plt.contourf(X, Y, Z, levels=50, cmap='viridis')
# plt.colorbar()

# # Drawing the grid
# plt.axhline(0, color='k', linestyle='--')
# plt.axvline(0, color='k', linestyle='--')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# Define points near the central grid edges
points = [(-0.75, 0), (-0.5, 0.7), (0, -0.5), (0.5, 0.5), (0.75, 0), (0.5, -0.5), (0, -0.75), (-0.5, -0.5)]

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()

# Draw the grid
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')

# # Calculate the gradient of the swirl function
# def gradient_swirl(x, y):
#     # Small change for numerical gradient
#     h = 1e-5
#     grad_x = (swirl_function(x + h, y) - swirl_function(x - h, y)) / (2 * h)
#     grad_y = (swirl_function(x, y + h) - swirl_function(x, y - h)) / (2 * h)
#     return grad_x, grad_y

# for (px, py) in points:
#     # Calculate gradient
#     grad_x, grad_y = gradient_swirl(px, py)
    
#     # Normalize gradient for consistent arrow size
#     norm = np.sqrt(grad_x**2 + grad_y**2)
#     grad_x, grad_y = grad_x / norm * 0.2, grad_y / norm * 0.2  # Make the lines a bit longer by scaling
    
#     # Plotting points
#     plt.plot(px, py, 'ro')  # Red point
    
#     # Plotting gradients
#     plt.quiver(px, py, grad_x, grad_y, color='r', scale=1, width=0.005)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# plt.show()