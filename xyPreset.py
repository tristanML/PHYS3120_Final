import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button, TextBox
import math
import os

#--parameters--
I = 0.1
k = 1
T = 0
dt = 0.005
J = 100
gam = 1.5
W = 50

PRESET_FILE = "frame.csv"

# magnetic field params
H0 = 0
NH = 30
alpha = 0

# detector params
RECORD_EVERY_CYCLES = 2

fft2d_history = []
order_history = []
time_history = []

last_recorded_cycle = [-1]
last_fft2d_strength = 0.0
last_order = 0.0


def wrap_angles(arr):
    return (arr + np.pi) % (2 * np.pi) - np.pi


def load_spin_preset():
    loaded_spins = np.loadtxt(PRESET_FILE, delimiter=",", ndmin=2)

    if loaded_spins.ndim != 2:
        raise ValueError("frame.csv must be a 2D array")

    loaded_spins = wrap_angles(loaded_spins)

    return loaded_spins


def clamp_NH_to_shape(value, shape):
    rows, cols = shape
    max_width = max(0, min(rows, cols) // 2 - 1)
    return max(0, min(int(value), max_width))


spins = load_spin_preset()
ROWS, COLS = spins.shape
NH = clamp_NH_to_shape(NH, spins.shape)

prevSpins = spins.copy()
mask = np.ones(spins.shape)
totalt = [0]

print(f"Loaded starting state from: {os.path.abspath(PRESET_FILE)}")
print(f"CSV shape: {ROWS} rows x {COLS} columns")


def set_H(a, h_theta, H0, update_spins=False):
    a = int(a)
    H = np.zeros(spins.shape)

    if a <= 0:
        return H, h_theta

    center_row = ROWS // 2
    center_col = COLS // 2

    row_start = max(0, center_row - a)
    row_end = min(ROWS, center_row + a + 1)

    col_start = max(0, center_col - a)
    col_end = min(COLS, center_col + a + 1)

    H[row_start:row_end, col_start:col_end] = H0

    if update_spins:
        spins[row_start:row_end, col_start:col_end] = h_theta

    return H, h_theta


H, alpha = set_H(NH, 0, H0, update_spins=False)


def get_drive_period():
    if W <= 0:
        return None
    return 2 * np.pi / W


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

    fft2d_line.set_data(time_history, fft2d_history)
    m_line.set_data(time_history, order_history)

    ax_fft2d.relim()
    ax_fft2d.autoscale_view()

    ax_m.relim()
    ax_m.autoscale_view()

    fig_metrics.canvas.draw_idle()


def reset_metric_history():
    global last_fft2d_strength, last_order

    fft2d_history.clear()
    order_history.clear()
    time_history.clear()

    fft2d_line.set_data([], [])
    m_line.set_data([], [])
    fig_metrics.canvas.draw_idle()

    last_recorded_cycle[0] = -1
    last_fft2d_strength = 0.0
    last_order = 0.0


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


#---main simulation window--------------------
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.40)

#---live detector plots in a separate window---
fig_metrics, (ax_fft2d, ax_m) = plt.subplots(2, 1, figsize=(7, 5))
fig_metrics.tight_layout(pad=2.0)

fft2d_line, = ax_fft2d.plot([], [], marker="o", markersize=3)
m_line, = ax_m.plot([], [], marker="o", markersize=3)

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
        reset_metric_history()
    except ValueError:
        pass


def update_H0(text):
    global H0, H, alpha
    try:
        H0 = float(text)
        H, alpha = set_H(NH, alpha, H0, update_spins=False)
        reset_metric_history()
    except ValueError:
        pass


def update_NH(text):
    global NH, H, alpha
    try:
        NH = clamp_NH_to_shape(int(float(text)), spins.shape)
        H, alpha = set_H(NH, alpha, H0, update_spins=False)
        reset_metric_history()
    except ValueError:
        pass


def update_J(text):
    global J
    try:
        J = float(text)
        reset_metric_history()
    except ValueError:
        pass


def update_GAM(text):
    global gam
    try:
        gam = float(text)
        reset_metric_history()
    except ValueError:
        pass


def update_I(text):
    global I
    try:
        I = float(text)
        reset_metric_history()
    except ValueError:
        pass


def update_W(text):
    global W
    try:
        W = float(text)
        reset_metric_history()
    except ValueError:
        pass


boxT.on_submit(update_T)
boxH0.on_submit(update_H0)
boxNH.on_submit(update_NH)
boxJ.on_submit(update_J)
boxGAM.on_submit(update_GAM)
boxI.on_submit(update_I)
boxW.on_submit(update_W)

button1_ax = fig.add_axes([0.2, 0.23, 0.25, 0.05])
button2_ax = fig.add_axes([0.5, 0.23, 0.25, 0.05])
button3_ax = fig.add_axes([0.35, 0.30, 0.30, 0.05])

button1 = Button(button1_ax, "Reset CSV")
button2 = Button(button2_ax, "Randomize Spins")
button3 = Button(button3_ax, "Reset Metrics")

my_cmap = plt.get_cmap("hsv").copy()
my_cmap.set_over("black")


def get_display_spins():
    display_spins = wrap_angles(spins).copy()
    display_spins[mask == 0] = 10
    return display_spins


im = ax.imshow(
    get_display_spins(),
    cmap=my_cmap,
    vmin=-np.pi,
    vmax=np.pi,
    interpolation="nearest"
)

plt.colorbar(im, ax=ax)


def update_main_image():
    im.set_data(get_display_spins())

    im.set_extent((-0.5, COLS - 0.5, ROWS - 0.5, -0.5))
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)

    fig.canvas.draw_idle()


def reset1(event):
    global spins, prevSpins, mask, H, ROWS, COLS, NH, alpha

    loaded_spins = load_spin_preset()

    spins = loaded_spins.copy()
    prevSpins = spins.copy()

    ROWS, COLS = spins.shape
    NH = clamp_NH_to_shape(NH, spins.shape)

    mask = np.ones(spins.shape)
    H, alpha = set_H(NH, alpha, H0, update_spins=False)

    reset_metric_history()
    totalt[0] = 0

    update_main_image()

    ax.set_title(f"Reloaded {ROWS} x {COLS} state from frame.csv")
    fig.canvas.draw_idle()

    print(f"Reloaded state from: {os.path.abspath(PRESET_FILE)}")
    print(f"CSV shape: {ROWS} rows x {COLS} columns")


def reset2(event):
    global spins, prevSpins

    spins[:] = np.random.uniform(-np.pi, np.pi, size=spins.shape)
    prevSpins = spins.copy()

    reset_metric_history()
    totalt[0] = 0

    update_main_image()


def reset3(event):
    reset_metric_history()


button1.on_clicked(reset1)
button2.on_clicked(reset2)
button3.on_clicked(reset3)


def get_Mag(spins):
    y = np.mean(np.sin(spins))
    x = np.mean(np.cos(spins))
    return np.atan2(y, x)


def step(frame):
    global prevSpins, last_fft2d_strength, last_order

    eta = (2 * I * gam * k * T / dt) ** 0.5 * np.random.randn(*spins.shape) * mask

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

    Happ = H * math.sin(W * totalt[0])
    totalt[0] += dt

    hInt = -Happ * np.sin(spins - alpha) * mask

    pot = exch + eta + hInt

    newSpins = 1 / (1 + gam / (2 * I) * dt) * (
        pot / I * dt ** 2
        + 2 * spins
        - (1 - gam / (2 * I) * dt) * prevSpins
    )

    prevSpins[:] = spins[:]
    spins[:] = newSpins

    if W > 0:
        current_cycle = get_current_cycle()

        if current_cycle != last_recorded_cycle[0]:
            last_recorded_cycle[0] = current_cycle

            if current_cycle % RECORD_EVERY_CYCLES == 0:
                field_mask = get_field_mask()

                last_order = get_order_parameter(spins, field_mask)
                last_fft2d_strength = fft2d_structure_strength(spins, field_mask)

                fft2d_history.append(last_fft2d_strength)
                order_history.append(last_order)
                time_history.append(totalt[0])

                update_metric_plots()

    ax.set_title(
        f"Field FFT2D={last_fft2d_strength:.3f} | Field M={last_order:.3f}",
        fontsize=9
    )

    im.set_data(get_display_spins())

    spins[mask == 0] = 10

    return [im]


def onclick(event):
    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        ix, iy = int(round(event.xdata)), int(round(event.ydata))

        if 0 <= ix < COLS and 0 <= iy < ROWS:
            r = 10

            x_start = max(0, ix - r)
            x_end = min(COLS, ix + r + 1)

            y_start = max(0, iy - 10 * r)
            y_end = min(ROWS, iy + 10 * r + 1)

            mask[y_start:y_end, x_start:x_end] = 0
            reset_metric_history()

            update_main_image()


ani = animation.FuncAnimation(
    fig,
    step,
    frames=10,
    interval=1,
    blit=False
)

cid = fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()