import math

import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from RunningEnv import RunningEnv

x=0
FRAMES = 500
pref_pace = 181
target_pace = pref_pace * 1.1
env = RunningEnv(pref_pace, 1)
pacing = 0
total_reward = 0
state = np.empty(0)
x_acc = np.empty(0)

current_plt = plt.figure()
def onclick(event):
    global pacing
    pacing = not(pacing)
    print('Pace %s' % ('Activated' if pacing else 'Deactivated'))
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))

cid = current_plt.canvas.mpl_connect('button_press_event', onclick)

def animate(iFrame):
    global x, x_acc, state, total_reward
    x=x+1
    current_pace, new_state, reward, real_pacing, done = env.step(pacing, target_pace)
    total_reward = total_reward + reward
    x_acc = np.append(x_acc, x)
    state = np.append(state, (new_state[0]+1)*target_pace)
    pace_color = 'b'
    if real_pacing: pace_color = 'g'
    plt.scatter(x, current_pace, marker="x", color=pace_color)
    plt.plot(x_acc, state, 'r-', linewidth=1)
    plt.annotate("Reward: %s"%(np.round(reward[0], 1)), xy=(1, 1), xycoords='figure points',
                 bbox=dict(fc='white', ec='white'))
    plt.annotate("Avg. Pace: %s"%(np.round((new_state[0]+1)*target_pace, 0)), xy=(100, 1), xycoords='figure points',
                 bbox=dict(fc='white', ec='white'))
    plt.annotate("Total Reward: %s"%(np.round(total_reward[0], 2)), xy=(200, 1), xycoords='figure points',
                 bbox=dict(fc='white', ec='white'))
    plt.annotate("Remaining time: %s%%"%(np.round(new_state[0], 2)), xy=(300, 1), xycoords='figure points',
                 bbox=dict(fc='white', ec='white'))

ani = FuncAnimation(current_plt, animate, interval=1000, frames=FRAMES, repeat=False)
plt.axhline(y=pref_pace*1.1, color='k', linestyle='--', label='Target Pace')
plt.title("Pace simulator")
plt.xlabel("time")
plt.ylabel("spm")

plt.show()

# ani.save(r'animation.gif', fps=10)


