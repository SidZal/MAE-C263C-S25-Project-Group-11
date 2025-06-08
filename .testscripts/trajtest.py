from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import numpy as np

spl = CubicSpline(
    x=np.asarray([0, 5]), 
    y= np.asarray(([5, 1], [2, 2])).T, 
    axis = 1, 
    bc_type=((1, [-1, 0]), (1, np.zeros(2)))
)

t = np.linspace(0, 5, 50)

fig = plt.figure()
ax = fig.add_subplot(111)

print(f"{spl(t)=}")

ax.plot(t, spl(t)[0], label="S")
ax.plot(t, spl(t, 1)[0], label="Sd")
ax.plot(t, spl(t, 2)[0], label="Sdd")
ax.plot(t, spl(t)[1], label="S1")
ax.plot(t, spl(t, 1)[1], label="S1d")
ax.plot(t, spl(t, 2)[1], label="S1dd")
ax.legend()

plt.show()
