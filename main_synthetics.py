from scipy.signal import butter, lfilter
#from scipy.stats import norm
from numpy import linalg as LA
from matplotlib import rc, font_manager
import math
import numpy as np
import os

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as colors

rc('text', usetex=True)
plt.rc('font', family='serif')


def synth_generator(dt,ns,nr,R,t0s,S,f0,vp,vs,rho,M):
        # Change units
        pi = np.pi;
        R  = R/1e3;  # receiver coordinates
        S[:] = [x / 1e3 for x in S] # Source coordinates
        vp = vp/1e3; # P-velocity
        vs = vs/1e3; # S-velocity
        vp3 = vp*vp*vp;  # elevate to ^3 power, once and for all.
        vs3 = vs*vs*vs;  # elevate to ^3 power, once and for all.
        rw  = Ricker(dt=dt, f0=f0); # computes riker wavelet
        nrw = len(rw);              # get the length of rw
        nf  = 4 * nextpow2(ns);     # get next power of 2, for padding
        dw  = 2.0 * pi/(nf*dt);     # angular frec
        zero_pad = np.zeros(nf-nrw)                       # zero tail
        rwpad    = np.concatenate((rw,zero_pad),axis=0);  # padd with tail.
        RW       = np.fft.fft(rwpad);                     # discrete FFT.
        U        = np.zeros((ns, 3*nr));
        Ux       = np.zeros((ns, nr));
        Uy       = np.zeros((ns, nr));
        Uz       = np.zeros((ns, nr));
        for rec in range(nr):#range(nr):
                SR = -(S - R[rec, :])    ; # vector pointing from Source to Receiver jth.
                r  = LA.norm(SR)         ; # distance from source to Receiver jth
                v  = np.transpose(np.array([ [x / r for x in SR] ])); # 3 X 1 column vector
                cp = 1.0/(4*pi*rho*vp3*r); # aux value
                cs = 1.0/(4*pi*rho*vs3*r); # aux value
                vt = np.transpose(v)     ; # transpose v. 1X3 row vector
                m1 = np.dot(v,cp*vt)     ; # v*cp*vt);
                m2 = np.dot(M,v)         ; # M,v    ;
                Ap = np.dot(m1,m2)       ; # v * cp * vt * M * v;
                m3 = np.dot(v,vt) ;
                m4 = np.dot(m3,m2);
                As = cs*(m2 - m4)      ;  # cs*(M*v - v*vt*M*v);
                tp = r/vp              ;  # arrival time p- for jth receiver
                ts = r/vs              ;  # arrival time sh- for jth receiver
                Uj  = np.array(np.zeros((nf, 3),dtype=complex)); # initialize fft matrix
                Ujp = np.array(np.zeros((nf, 3),dtype=complex)); # initialize p-fft matrix
                Ujs = np.array(np.zeros((nf, 3),dtype=complex)); # initialize sh-fft matrix
                nf2 = int(math.floor(nf/2))          ; # nyquist frec
                for k in np.arange(2,nf2+1):# from 2 to nf2
                        w    = (k-1)*dw   ;    # angular frec
                        imw  = 1j*w       ;
                        imwR = imw*RW[k-1];    # first evaluate in 2nd index k=2-1=1
                        tsp  = t0s + tp   ;
                        tss  = t0s + ts   ;
                        Ujp[k-1, :] = np.conj(np.transpose(np.array([up/np.exp(-imw*tsp) for up in Ap]))); # first frec is saved in index = 1, Not 0!
                        Ujs[k-1, :] = np.conj(np.transpose(np.array([us/np.exp(-imw*tss) for us in As])));
                        Uj[k-1, :]  = imwR * (Ujp[k-1,:] + Ujs[k-1,:]);
                        Uj[nf-(k-1), :] = np.conj(Uj[k-1, :]);
                #end
                ujt    = (np.fft.ifft(Uj, axis=0)).real
                ujcrop = ujt[0:350,:]
                jx = 3*(rec-1) + 3
                U[0:350,jx+0] = ujcrop[:,0]; # signal in xyz order
                U[0:350,jx+1] = ujcrop[:,1];
                U[0:350,jx+2] = ujcrop[:,2];

                Ux[0:350,rec] = ujcrop[:,0]; # separate signal channels
                Uy[0:350,rec] = ujcrop[:,1];
                Uz[0:350,rec] = ujcrop[:,2];

        return U,Ux,Uy,Uz

def Ricker(dt=0.002,f0=20.0):
        nw = 2.0/(f0*dt);
        nc = math.floor(nw/2);
        t  = dt*np.arange(-nc,nc+1,1);
        pf = f0;
        pi = np.pi;
        rick1  = pi*pf*t
        rick12 = np.power(rick1,2)        # (pi*pf*t)^2;
        rick2  = (1.0-2.0*rick12)         # (1.0-2.0*(pi*pf*t)^2)
        rick3  = np.exp(-rick12)          # exp(-(pi*pf*t)^2);
        rick   = np.multiply(rick2,rick3) #(1.0-2.0*(pi*pf*t)^2)*exp(-(pi*p[1]*t)^2);
        return rick;

def nextpow2(x):
        absx  = abs(x);
        lg2   = math.log(absx,2);
        nxtp2 = int(2**math.ceil(lg2));
        return nxtp2



from scipy.signal import butter, lfilter
#from scipy.stats import norm
from numpy import linalg as LA
from matplotlib import rc, font_manager
import math
import numpy as np
import os

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.colors as colors

rc('text', usetex=True)
plt.rc('font', family='serif')


rec_file = "receptores/receptores_coord.txt" ;
tmin  = 0.000   ;  # first sample in processing window [s]
tmax  = 0.350   ;  # last sample in processing window [s]
dt    = 0.001   ;  # sampling interval [s]
t0s   = 0.000   ;  # -0.037; # t0 of source [s]
f0    = 100.0   ;  # Ricker wavelet central frequency [Hz]
vp    = 4500.0  ;  # 3500.0; # P-wave velocity [m/s]
vs    = 2500.0  ;  # 2400.0; # S-wave velocity [m/s]
rho   = 2700    ;  # 1.0;    # density of the medium [kg/m³]
S     = [240.0, 320.0, -353.5]
R        = np.loadtxt(rec_file);
nr       = np.shape(R)[0]                 ; # number of receivers
ns       = int(math.ceil(tmax/dt))        ; # number of samples in data

M        = np.array([[0,-1,0],
[-1,0,0],
[0,0,0]])                      ; # Moment tensor. Symetric 3x3 matrix;

signal,ux,uy,uz = synth_generator(dt,ns,nr,R,t0s,S,f0,vp,vs,rho,M)
#np.savetxt('gathers/signal.txt', signal)
#np.savetxt('gathers/datax.txt', ux)
#np.savetxt('gathers/datay.txt', uy)
#np.savetxt('gathers/dataz.txt', uz)
