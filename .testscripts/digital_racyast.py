# get modules from project file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import matplotlib.pyplot as plt

from pongBot import pongBot

raytest = pongBot([1, 1, 0.1])

arena_width = 10
ball_radius = .4
ball_p0 = [10, 0]
ball_v0 = [-.5, 1.4]

coords = raytest.ball_raycast(arena_width, ball_radius, ball_p0, ball_v0, dt = 1)
# print(coords)


fig, ax = plt.subplots()

plt.axhline(y=arena_width/2, color='black')
plt.axhline(y=-arena_width/2, color='black')

ax.add_artist(plt.Circle(ball_p0, ball_radius, color="green", fill=False))
ax.annotate("", xytext=(ball_p0[0], ball_p0[1]), xy=(ball_p0[0]+.5*ball_v0[0], ball_p0[1]+.5*ball_v0[1]), arrowprops=dict(arrowstyle="->"))
ax.annotate("Starting Position", xytext=(ball_p0[0], ball_p0[1]), xy=(ball_p0[0]+.5*ball_v0[0], ball_p0[1]+.5*ball_v0[1]))


for coord in coords:
    clr = 'blue'
    if coord[2] != 0:
        clr = 'r'
    ax.add_artist(plt.Circle(coord, ball_radius, color=clr, fill=False ))
plt.scatter(coords.T[0], coords.T[1], marker="+", color="gray")

plt.axis('scaled')
plt.xlim([0, 10])
plt.ylim([-5, 5])
plt.grid()
plt.show()