import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Slider,Button
import random
#--parameters--
I = 0.1
k = 1
#----
T = 0.4
dt = 0.005
J = 5
N = 100
gam = 1.5

#magnetic field params
H0 = 0 #magnitude of magnetic field
NH = 0 #width of magnetic field, measured in cells
alpha = 0
#--------------


#---axis and sliders --------------------
fig, ax = plt.subplots(figsize = (6,6))
plt.subplots_adjust(bottom=0.35)
axsliderT = plt.axes([0.2, 0.01, 0.6, 0.02])
axsliderH0 = plt.axes([0.2, 0.04, 0.6, 0.02])
axsliderNH = plt.axes([0.2, 0.07, 0.6, 0.02])
axsliderJ = plt.axes([0.2, 0.10, 0.6, 0.02])
axsliderGAM = plt.axes([0.2, 0.13, 0.6, 0.02])
axsliderI = plt.axes([0.2,0.16,0.6,0.02])

sliderT = Slider(axsliderT,'T', 0, 10.0, valinit = T)
sliderH0 = Slider(axsliderH0,'H0', 0, 100.0, valinit = H0)
sliderNH = Slider(axsliderNH,'Width of H', 0, N//2-1, valinit = 0,valstep =1)
sliderJ = Slider(axsliderJ,'J', 0, 100.0, valinit = J)
sliderGAM = Slider(axsliderGAM,'Gamma', 0.1, 5.0, valinit = gam)
sliderI = Slider(axsliderI, 'I', 0.01, 10, valinit = I)
def update_T(val):
    global T
    T = sliderT.val

def update_H0(val):
    global H,H0,NH
    H0 = sliderH0.val
    H,alpha = set_H(NH,0,H0,update_spins = False)

def update_NH(val):
    global NH,H
    NH = sliderNH.val
    H, alpha = set_H(NH,0,H0,update_spins = False)

def update_J(val):
    global J
    J = sliderJ.val

def update_GAM(val):
    global gam
    gam = sliderGAM.val

def update_I(val):
    global I
    I = sliderI.val

sliderT.on_changed(update_T)
sliderH0.on_changed(update_H0)
sliderNH.on_changed(update_NH)
sliderJ.on_changed(update_J)
sliderGAM.on_changed(update_GAM)
sliderI.on_changed(update_I)


#----------------------------------------------
spins = np.full((N,N), np.pi-2*np.pi*random.random())
prevSpins = spins.copy()

button1_ax = plt.axes([0.2, 0.21, 0.25, 0.05])
button2_ax = plt.axes([0.5, 0.21, 0.25, 0.05])
button1 = Button(button1_ax, 'Align Spins')
button2 = Button(button2_ax, 'Randomize Spins')

def reset1(event):
    global spins, prevSpins
    spins[:] = np.full((N,N), np.pi-2*np.pi*random.random())
    prevSpins = spins.copy()

def reset2(event):
    global spins, prevSpins
    spins[:] = np.random.uniform(-np.pi,np.pi, size = (N,N))
    prevSpins = spins.copy()

    
button1.on_clicked(reset1)
button2.on_clicked(reset2)
my_cmap = plt.get_cmap('hsv').copy()
my_cmap.set_over('black')
im = ax.imshow(spins, cmap=my_cmap, vmin=-np.pi, vmax=np.pi, interpolation='nearest')
plt.colorbar(im)

def get_Mag(spins):
    y = np.mean(np.sin(spins))
    x = np.mean(np.cos(spins))
    return np.atan2(y,x)

def set_H(a, h_theta, H0, update_spins):
    H = np.zeros((N,N)) 
    if a == 0:
        #H[N//2][N//2] = H0
        return H, h_theta
    else:
        for i in range(N//2-a,N//2+a+1):
            for j in range(N//2-a,N//2+a+1):
                H[i][j] = H0
                if update_spins:
                    spins[i][j] = alpha
        return H, h_theta
    
H, alpha = set_H(NH,0,H0,update_spins = False)
mask = np.ones((N, N))
# Let's put a "nonmagnetic" square in the middle
# mask[N//4:3*N//4, N//4:3*N//4] = 0
def step(frame,spins,im):
    eta = (2*gam*k*T/dt)**(1/2)*np.random.randn(N,N)
    sTop = np.roll(spins, 1, axis = 1)
    sBot = np.roll(spins, -1, axis = 1)
    sLeft = np.roll(spins, 1, axis = 0)
    sRight = np.roll(spins, -1, axis = 0)

    exch = -J*(np.sin(spins-sTop)+np.sin(spins-sBot)+np.sin(spins-sLeft)+np.sin(spins-sRight))
    hInt = -H*np.sin(spins-alpha)
    pot = exch+eta+hInt
    #spins[:] = spins + (exch+eta+hInt)*dt/gam
    newSpins = 1/(1+gam/(2*I)*dt)*( (pot)/I*dt**2 + 2*spins  - (1-gam/(2*I)*dt)*prevSpins )
    prevSpins[:] = spins[:]
    spins[:] = newSpins

    spins[mask == 0] = 10

    im.set_data((spins+np.pi)%(2*np.pi)-np.pi)
    return [im]

def onclick(event):
    if event.inaxes == ax:
        # Get coordinates
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        
        # Check bounds
        if 0 <= ix < N and 0 <= iy < N:
            r = 2  # This creates a 5x5 square (ix-2 to ix+2)
        
            # Calculate bounds and ensure they stay within the grid [0, N]
            x_start, x_end = max(0, ix - r), min(N, ix + r + 1)
            y_start, y_end = max(0, iy - r), min(N, iy + r + 1)
            
            # Apply to the mask and spins simultaneously
            mask[y_start:y_end, x_start:x_end] = 0
            spins[y_start:y_end, x_start:x_end] = 0
            
            # Update display immediately
            im.set_data(spins)
            fig.canvas.draw_idle()


ani = animation.FuncAnimation(fig,step,fargs=(spins, im),frames = 10, interval=1,blit=True)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
