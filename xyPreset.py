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
N = 100
gam = 1.5
W = 50

PRESET_FILE = "spin_preset.csv"

# magnetic field params
H0 = 100
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


def load_spin_preset():
    loaded_spins = np.loadtxt(PRESET_FILE, delimiter=",")

    if loaded_spins.ndim != 2:
        raise ValueError("spin_preset.csv must be a 2D array")

    if loaded_spins.shape[0] != loaded_spins.shape[1]:
        raise ValueError("spin_preset.csv must be square, like 100 x 100")

    loaded_spins = (loaded_spins + np.pi) % (2 * np.pi) - np.pi

    return loaded_spins


spins = load_spin_preset()
N = spins.shape[0]
NH = max(0, min(NH, N // 2 - 1))

print(f"Loaded starting state from: {os.path.abspath(PRESET_FILE)}")


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
        NH = int(float(text))
        NH = max(0, min(NH, N // 2 - 1))
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

#----------------------------------------------
prevSpins = spins.copy()

button1_ax = fig.add_axes([0.2, 0.23, 0.25, 0.05])
button2_ax = fig.add_axes([0.5, 0.23, 0.25, 0.05])
button3_ax = fig.add_axes([0.35, 0.30, 0.30, 0.05])

button1 = Button(button1_ax, "Reset CSV")
button2 = Button(button2_ax, "Randomize Spins")
button3 = Button(button3_ax, "Reset Metrics")


def reset1(event):
    global prevSpins

    loaded_spins = load_spin_preset()

    if loaded_spins.shape != spins.shape:
        print("CSV shape does not match current simulation shape.")
        print(f"CSV shape: {loaded_spins.shape}, current shape: {spins.shape}")
        return

    spins[:] = loaded_spins
    prevSpins[:] = spins[:]

    reset_metric_history()
    totalt[0] = 0

    display_spins = (spins + np.pi) % (2 * np.pi) - np.pi
    display_spins = display_spins.copy()
    display_spins[mask == 0] = 10
    im.set_data(display_spins)

    ax.set_title("Reloaded state from spin_preset.csv")
    fig.canvas.draw_idle()

    print(f"Reloaded state from: {os.path.abspath(PRESET_FILE)}")


def reset2(event):
    global prevSpins
    spins[:] = np.random.uniform(-np.pi, np.pi, size=(N, N))
    prevSpins[:] = spins[:]
    reset_metric_history()
    totalt[0] = 0


def reset3(event):
    reset_metric_history()


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


def step(frame, spins, im, totalt):
    global last_fft2d_strength, last_order

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
            reset_metric_history()

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