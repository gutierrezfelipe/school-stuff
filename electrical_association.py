import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import use


use('TkAgg')


# 2 Resistors Parallel association
points = 500
starting_value = 250e3
R_1_0 = np.linspace(0.1, starting_value, points)
# R_1_0 = 250e3
R_2_0 = starting_value

fig_sig = plt.figure()
ax_s = fig_sig.add_subplot(111)
fig_sig.subplots_adjust(left=0.25, bottom=0.25)
plt.title('R parallel')
R_1_0 = np.linspace(0, starting_value, points)
R_2_0 = starting_value
delta_r = 100
R_t = R_1_0 * R_2_0 / (R_1_0 + R_2_0)
print(f'Maximum Resistance: {R_t[-1]} Ohms')
im1 = ax_s.plot(R_1_0, R_t, label=f'R_total')
ax_s.legend()
#ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'
ax_R1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
s_R1 = Slider(ax_R1, 'R1', 0.1, 1e6, valinit=starting_value, valstep=delta_r)
ax_R2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
s_R2 = Slider(ax_R2, 'R2', 0.1, 1e6, valinit=starting_value, valstep=delta_r)


def update_R_t(val):
    R1 = np.linspace(0, s_R1.val, points)
    R2 = s_R2.val
    R_t = R1 * R2 / (R1 + R2)

    print(f'Maximum Resistance: {R_t[-1]} Ohms')
    ax_s.cla()
    ax_s.plot(R1, R_t, label=f'R_total')
    ax_s.legend()
    fig_sig.canvas.draw_idle()


s_R1.on_changed(update_R_t)
s_R2.on_changed(update_R_t)


