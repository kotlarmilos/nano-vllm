import matplotlib.pyplot as plt
import numpy as np

x = np.array([5, 1, -3])
y = np.array([-3, 0, 1])

w_vals = np.linspace(-3, 3, 100)
loss_vals = []

print (w_vals)

for w in w_vals:
    y_pred = x * w
    loss = np.mean((y - y_pred)**2)
    loss_vals.append(loss)

print(loss_vals)

plt.plot(w_vals, loss_vals)
plt.xlabel("w vals")
plt.ylabel("loss(w) vals")

plt.show()