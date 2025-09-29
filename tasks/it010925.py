import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 1.8, 2.4, 2.8, 3.0, 3.1])
log_x = np.log(x)

a, b = 0.0, 0.0
learning_rate = 0.01
epochs = 5000
losses = []

for i in range(epochs):
    y_pred = a + b * log_x
    
    da = (2 / len(y)) * np.sum(y_pred - y)
    db = (2 / len(y)) * np.sum((y_pred - y) * log_x)
    
    a -= learning_rate * da
    b -= learning_rate * db
    
    sse = np.sum((y - y_pred)**2)
    losses.append(sse)

print(f"Optimized: y = {a:.3f} + {b:.3f} * ln(x)")
print(f"Sum Sqr e: {sse:.6f}")

plt.figure(figsize=(9, 6))
plt.scatter(x, y, color='red', label='Data')
x_smooth = np.linspace(1, 6, 100)
plt.plot(x_smooth, a + b * np.log(x_smooth), 'b-', label=f'Iterative fit: y = {a:.2f} + {b:.2f} ln(x)')
plt.legend()
plt.title('Iterative Log Fit')
plt.xlabel('x'); plt.ylabel('y')
plt.grid(True, alpha=0.3)
plt.show()