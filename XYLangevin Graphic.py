import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import math

#--parameters--
I = 0.1
k = 1
#----
T = 0.4
dt = 0.01
J = 100
N = 100
gam = 1.5
W = 40

# magnetic field params
H0 = 100
NH = 50
alpha = 0
#--------------

n_frames = 100000

spins = np.random.uniform(-np.pi, np.pi, size=(N, N))
prevSpins = spins.copy()

totalt = [0]


def wrap_angles(arr):
    return (arr + np.pi) % (2 * np.pi) - np.pi


def set_H(a, h_theta, H0):
    H = np.zeros((N, N))

    if a == 0:
        return H, h_theta

    a = int(a)

    for i in range(N // 2 - a, N // 2 + a + 1):
        for j in range(N // 2 - a, N // 2 + a + 1):
            if 0 <= i < N and 0 <= j < N:
                H[i][j] = H0

    return H, h_theta


H, alpha = set_H(NH, 0, H0)


# exact 100x100 output
dpi = 100
fig = plt.figure(figsize=(1, 1), dpi=dpi, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])

im = ax.imshow(
    spins,
    cmap="hsv",
    vmin=-np.pi,
    vmax=np.pi,
    interpolation="nearest",
    resample=False,
    aspect="equal",
    animated=True
)

ax.set_axis_off()


def step(frame, spins, im, totalt):
    global prevSpins

    eta = (2 * gam * I * k * T / dt) ** 0.5 * np.random.randn(N, N)

    sTop = np.roll(spins, 1, axis=1)
    sBot = np.roll(spins, -1, axis=1)
    sLeft = np.roll(spins, 1, axis=0)
    sRight = np.roll(spins, -1, axis=0)

    exch = -J * (
        np.sin(spins - sTop)
        + np.sin(spins - sBot)
        + np.sin(spins - sLeft)
        + np.sin(spins - sRight)
    )

    Happ = H * math.sin(W * totalt[0])
    totalt[0] += dt

    hInt = -Happ * np.sin(spins - alpha)
    pot = exch + eta + hInt

    newSpins = 1 / (1 + gam / (2 * I) * dt) * (
        (pot / I) * dt**2
        + 2 * spins
        - (1 - gam / (2 * I) * dt) * prevSpins
    )

    prevSpins[:] = spins[:]
    spins[:] = newSpins

    spins[:] = wrap_angles(spins)
    prevSpins[:] = wrap_angles(prevSpins)

    im.set_data(spins)
    return [im]


ani = animation.FuncAnimation(
    fig,
    step,
    fargs=(spins, im, totalt),
    frames=n_frames,
    interval=20,
    blit=True
)

writer = animation.FFMpegWriter(
    fps=30,
    codec="libx264",
    extra_args=["-pix_fmt", "yuv420p"]
)

ani.save("spins_0.4_40_100k.mp4", writer=writer, dpi=dpi)
print("Download Complete")
plt.close(fig)