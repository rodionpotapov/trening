import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def f(x):
    return x * x - 5 * x + 5


def df(x):
    return 2 * x - 5


N = 20
xx = 0
lmd = 0.3

x_plt = np.arange(0, 5.0, 0.1)
f_plt = [f(x) for x in x_plt]

for i in range(N + 1):
    xx = xx - lmd * np.sign(df(xx))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(True)
    ax.plot(x_plt, f_plt, label='f(x)')
    ax.set_title(" ".join(["Gradient Descent", f"step №{i}"]))

    if i == N:
        ax.scatter(xx, f(xx), c='blue', label='Converged Point')
    else:
        ax.scatter(xx, f(xx), c='red', label='Current Point')

    ax.text(xx + 0.5, f(xx) + 0.5, f'xx = {xx:.2f}', fontsize=10, ha='right')

    ax.legend()
    plt.show()

    time.sleep(0.02)
    display.clear_output(wait=True)