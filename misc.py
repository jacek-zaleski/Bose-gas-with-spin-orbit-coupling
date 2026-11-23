import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0, 1.5, 1000)
ys = 0.5 * xs**2 - xs
fig, ax = plt.subplots()
ax.plot(xs, ys, label='$E_-$')
ys = 0.5 * xs**2 + xs
ax.plot(xs, ys, label='$E_+$')
ax.grid()
ax.legend()
ax.set_xlabel('$|\\boldsymbol{p}|$')
fig.savefig('x2.pdf')
