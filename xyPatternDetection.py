import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
from collections import deque
import random
import math

#--parameters--
I = 0.1
k = 1
T = 0
dt = 0.005
J = 100
N = 100
gam = 1.5
W = 3250

# magnetic field params
H0 = 100
NH = 20
alpha = 0

# detector params
RECORD_EVERY_CYCLES = 4
MAX_CYCLE_SNAPSHOTS = 10
COMPARE_CYCLES_BACK = 2

cycle_snapshots = deque(maxlen=MAX_CYCLE_SNAPSHOTS)
repeat_history = []
fft2d_history = []
order_history = []
time_history = []

last_snapshot_cycle = [-1]
last_repeat_text = "Repeat: waiting"
last_fft2d_strength = 0.0
last_order = 0.0


def get_drive_period():
    if W <= 0:
        return None
    return 360 / W


def get_current_cycle():
    drive_period = get_drive_period()
    if drive_period is None:
        return 0
    return int(totalt[0] // drive_period)


def get_field_mask():
    return ((H > 0) & (mask == 1)).astype(int)


def update_metric_plots():
    if len(time_history) == 0:
        return

    repeat_line.set_data(time_history, repeat_history)
    fft2d_line.set_data(time_history, fft2d_history)
    m_line.set_data(time_history, order_history)

    ax_repeat.relim()
    ax_repeat.autoscale_view()

    ax_fft2d.relim()
    ax_fft2d.autoscale_view()

    ax_m.relim()
    ax_m.autoscale_view()

    fig_metrics.canvas.draw_idle()


def reset_detector_history():
    global last_repeat_text, last_fft2d_strength, last_order

    cycle_snapshots.clear()
    repeat_history.clear()
    fft2d_history.clear()
    order_history.clear()
    time_history.clear()

    repeat_line.set_data([], [])
    fft2d_line.set_data([], [])
    m_line.set_data([], [])
    fig_metrics.canvas.draw_idle()

    last_snapshot_cycle[0] = -1
    last_repeat_text = "Repeat: waiting"
    last_fft2d_strength = 0.0
    last_order = 0.0


def store_current_cycle_snapshot():
    current_cycle = get_current_cycle()
    cycle_snapshots.append((current_cycle, spins.copy()))
    last_snapshot_cycle[0] = current_cycle


def reset_detector_and_store_snapshot():
    reset_detector_history()
    store_current_cycle_snapshot()


def wrapped_angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def angular_rms_difference(a, b, mask=None):
    diff = wrapped_angle_difference(a, b)

    if mask is not None:
        diff = diff[mask == 1]

    if diff.size == 0:
        return np.nan

    return np.sqrt(np.mean(diff ** 2))


def fft2d_structure_strength(spins, mask=None):
    z = np.exp(1j * spins)

    if mask is not None:
        valid = mask == 1

        if np.sum(valid) == 0:
            return 0.0

        mean_z = np.mean(z[valid])

        z = z.copy()
        z[~valid] = mean_z

    z = z - np.mean(z)

    F = np.fft.fft2(z)
    power = np.abs(F) ** 2

    power[0, 0] = 0

    total_power = np.sum(power)

    if total_power == 0:
        return 0.0

    return np.max(power) / total_power


def get_order_parameter(spins, mask=None):
    if mask is None:
        theta = spins
    else:
        theta = spins[mask == 1]

    if theta.size == 0:
        return 0.0

    return abs(np.mean(np.exp(1j * theta)))


def is_stable_repeat(history, window=20, tolerance=0.005, max_error=0.15):
    if len(history) < 2 * window:
        return False

    recent = np.array(history[-window:])
    previous = np.array(history[-2 * window:-window])

    recent_mean = np.mean(recent)
    previous_mean = np.mean(previous)
    change = abs(recent_mean - previous_mean)

    return recent_mean < max_error and change < tolerance


#---main simulation window--------------------
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.40)

#---live detector plots in a separate window---
fig_metrics, (ax_repeat, ax_fft2d, ax_m) = plt.subplots(3, 1, figsize=(7, 7))
fig_metrics.tight_layout(pad=2.0)

repeat_line, = ax_repeat.plot([], [], marker="o", markersize=3)
fft2d_line, = ax_fft2d.plot([], [], marker="o", markersize=3)
m_line, = ax_m.plot([], [], marker="o", markersize=3)

ax_repeat.set_title("2T Repeat Error")
ax_repeat.set_xlabel("Time")
ax_repeat.set_ylabel("Error")
ax_repeat.grid(True)

ax_fft2d.set_title("Field 2D FFT")
ax_fft2d.set_xlabel("Time")
ax_fft2d.set_ylabel("FFT2D")
ax_fft2d.grid(True)

ax_m.set_title("Field M")
ax_m.set_xlabel("Time")
ax_m.set_ylabel("M")
ax_m.grid(True)

# controls stay attached to main window
axboxT = fig.add_axes([0.2, 0.01, 0.25, 0.035])
axboxH0 = fig.add_axes([0.6, 0.01, 0.25, 0.035])

axboxNH = fig.add_axes([0.2, 0.06, 0.25, 0.035])
axboxJ = fig.add_axes([0.6, 0.06, 0.25, 0.035])

axboxGAM = fig.add_axes([0.2, 0.11, 0.25, 0.035])
axboxI = fig.add_axes([0.6, 0.11, 0.25, 0.035])

axboxW = fig.add_axes([0.2, 0.16, 0.25, 0.035])

boxT = TextBox(axboxT, "T", initial=str(T))
boxH0 = TextBox(axboxH0, "H0", initial=str(H0))
boxNH = TextBox(axboxNH, "Width", initial=str(NH))
boxJ = TextBox(axboxJ, "J", initial=str(J))
boxGAM = TextBox(axboxGAM, "Gamma", initial=str(gam))
boxI = TextBox(axboxI, "I", initial=str(I))
boxW = TextBox(axboxW, "W", initial=str(W))


def update_T(text):
    global T
    try:
        T = float(text)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


def update_H0(text):
    global H0, H, alpha
    try:
        H0 = float(text)
        H, alpha = set_H(NH, alpha, H0, update_spins=False)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


def update_NH(text):
    global NH, H, alpha
    try:
        NH = int(float(text))
        NH = max(0, min(NH, N // 2 - 1))
        H, alpha = set_H(NH, alpha, H0, update_spins=False)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


def update_J(text):
    global J
    try:
        J = float(text)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


def update_GAM(text):
    global gam
    try:
        gam = float(text)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


def update_I(text):
    global I
    try:
        I = float(text)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


def update_W(text):
    global W
    try:
        W = float(text)
        reset_detector_and_store_snapshot()
    except ValueError:
        pass


boxT.on_submit(update_T)
boxH0.on_submit(update_H0)
boxNH.on_submit(update_NH)
boxJ.on_submit(update_J)
boxGAM.on_submit(update_GAM)
boxI.on_submit(update_I)
boxW.on_submit(update_W)

#----------------------------------------------
spins = np.full((N, N), np.pi / 2)
prevSpins = spins.copy()

button1_ax = fig.add_axes([0.2, 0.23, 0.25, 0.05])
button2_ax = fig.add_axes([0.5, 0.23, 0.25, 0.05])
button3_ax = fig.add_axes([0.35, 0.30, 0.30, 0.05])

button1 = Button(button1_ax, "Align Spins")
button2 = Button(button2_ax, "Randomize Spins")
button3 = Button(button3_ax, "Reset Detector")


def reset1(event):
    global spins, prevSpins
    spins[:] = np.pi / 2
    prevSpins = spins.copy()
    reset_detector_and_store_snapshot()


def reset2(event):
    global spins, prevSpins
    spins[:] = np.random.uniform(-np.pi, np.pi, size=(N, N))
    prevSpins = spins.copy()
    reset_detector_and_store_snapshot()


def reset3(event):
    reset_detector_and_store_snapshot()


button1.on_clicked(reset1)
button2.on_clicked(reset2)
button3.on_clicked(reset3)

my_cmap = plt.get_cmap("hsv").copy()
my_cmap.set_over("black")

im = ax.imshow(
    spins,
    cmap=my_cmap,
    vmin=-np.pi,
    vmax=np.pi,
    interpolation="nearest"
)

plt.colorbar(im, ax=ax)


def get_Mag(spins):
    y = np.mean(np.sin(spins))
    x = np.mean(np.cos(spins))
    return np.atan2(y, x)


def set_H(a, h_theta, H0, update_spins):
    a = int(a)
    H = np.zeros((N, N))

    if a == 0:
        return H, h_theta
    else:
        for i in range(N // 2 - a, N // 2 + a + 1):
            for j in range(N // 2 - a, N // 2 + a + 1):
                H[i][j] = H0
                if update_spins:
                    spins[i][j] = alpha

        return H, h_theta


H, alpha = set_H(NH, 0, H0, update_spins=False)

mask = np.ones((N, N))
totalt = [0]

# Store the first snapshot immediately when the program starts.
store_current_cycle_snapshot()


def step(frame, spins, im, totalt):
    global last_repeat_text, last_fft2d_strength, last_order

    eta = (2 * I * gam * k * T / dt) ** 0.5 * np.random.randn(N, N) * mask

    sTop = np.roll(spins, 1, axis=1)
    sBot = np.roll(spins, -1, axis=1)
    sLeft = np.roll(spins, 1, axis=0)
    sRight = np.roll(spins, -1, axis=0)

    exch = -J * (
        np.sin(spins - sTop) * np.roll(mask, 1, axis=1)
        + np.sin(spins - sBot) * np.roll(mask, -1, axis=1)
        + np.sin(spins - sLeft) * np.roll(mask, 1, axis=0)
        + np.sin(spins - sRight) * np.roll(mask, -1, axis=0)
    ) * mask

    Happ = H * math.sin(math.radians(W * totalt[0]))
    totalt[0] += dt

    hInt = -Happ * np.sin(spins - alpha) * mask

    pot = exch + eta + hInt

    newSpins = 1 / (1 + gam / (2 * I) * dt) * (
        (pot) / I * dt ** 2
        + 2 * spins
        - (1 - gam / (2 * I) * dt) * prevSpins
    )

    prevSpins[:] = spins[:]
    spins[:] = newSpins

    if W > 0:
        current_cycle = get_current_cycle()

        if current_cycle != last_snapshot_cycle[0]:
            cycle_snapshots.append((current_cycle, spins.copy()))
            last_snapshot_cycle[0] = current_cycle

            if current_cycle % RECORD_EVERY_CYCLES == 0:
                field_mask = get_field_mask()

                last_order = get_order_parameter(spins, field_mask)
                last_fft2d_strength = fft2d_structure_strength(spins, field_mask)

                snapshots_by_cycle = dict(cycle_snapshots)
                target_cycle = current_cycle - COMPARE_CYCLES_BACK

                if target_cycle in snapshots_by_cycle:
                    old_spins = snapshots_by_cycle[target_cycle]
                    repeat_error = angular_rms_difference(spins, old_spins, field_mask)

                    repeat_history.append(repeat_error)
                    fft2d_history.append(last_fft2d_strength)
                    order_history.append(last_order)
                    time_history.append(totalt[0])

                    update_metric_plots()

                    stable_text = "stable" if is_stable_repeat(repeat_history) else "settling"

                    last_repeat_text = (
                        f"Repeat: 2T err={repeat_error:.6f} {stable_text} "
                        f"| cycle {current_cycle}"
                    )
                else:
                    last_repeat_text = (
                        f"Repeat: waiting for cycle {COMPARE_CYCLES_BACK} "
                        f"| cycle {current_cycle}"
                    )

    else:
        last_repeat_text = "Repeat: W must be > 0"

    ax.set_title(
        f"{last_repeat_text} | Field FFT2D={last_fft2d_strength:.3f} | Field M={last_order:.3f}",
        fontsize=9
    )

    display_spins = (spins + np.pi) % (2 * np.pi) - np.pi
    display_spins = display_spins.copy()
    display_spins[mask == 0] = 10
    im.set_data(display_spins)

    spins[mask == 0] = 10

    return [im]


def onclick(event):
    if event.inaxes == ax:
        ix, iy = int(round(event.xdata)), int(round(event.ydata))

        if 0 <= ix < N and 0 <= iy < N:
            r = 10

            x_start, x_end = max(0, ix - r), min(N, ix + r + 1)
            y_start, y_end = max(0, iy - 10 * r), min(N, iy + 10 * r + 1)

            mask[y_start:y_end, x_start:x_end] = 0
            reset_detector_and_store_snapshot()

            display_spins = (spins + np.pi) % (2 * np.pi) - np.pi
            display_spins = display_spins.copy()
            display_spins[mask == 0] = 10
            im.set_data(display_spins)
            fig.canvas.draw_idle()


ani = animation.FuncAnimation(
    fig,
    step,
    fargs=(spins, im, totalt),
    frames=10,
    interval=1,
    blit=False
)

cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
