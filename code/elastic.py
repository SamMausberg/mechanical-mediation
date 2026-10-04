"""Shared-recoil interferometer calculations, SI units.

All channel formulae refer to the explicitly specified nondemolition Gaussian
force model. No claim about an uncharacterized experimental support is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
import mpmath as mp
from numpy.polynomial import Polynomial
from scipy.integrate import quad

G = 6.67430e-11
HBAR = 1.0545718176461565e-34  # h/(2 pi), h exact in SI
KB = 1.380649e-23
C_LIGHT = 299792458.0
SIGNS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
S = Polynomial([0, 0, 0, 0, 35, -84, 70, -20])

@dataclass(frozen=True)
class Protocol:
    mass: float = 1e-14
    separation: float = 450e-6
    split: float = 250e-6
    ramp: float = 0.5
    hold: float = 2.5
    def __post_init__(self):
        if min(self.mass, self.separation, self.split, self.ramp) <= 0 or self.hold < 0:
            raise ValueError('Require positive masses/lengths/ramp, nonnegative hold.')
        if self.split >= self.separation:
            raise ValueError('The paths must not cross.')
    @property
    def duration(self): return 2*self.ramp + self.hold
    def envelope(self, t: float, shape: str = 'septic') -> float:
        if not 0 <= t <= self.duration: return 0.0
        if t <= self.ramp: u = t/self.ramp
        elif t <= self.ramp+self.hold: return 1.0
        else: u = (self.duration-t)/self.ramp
        if shape == 'septic': return float(S(u))
        if shape == 'bang': return 2*u*u if u <= .5 else 1-2*(1-u)**2
        raise ValueError('Unknown shape')
    def branches(self, t: float, shape='septic'):
        return self.split*self.envelope(t, shape)
    def recoil_phase(self, reservoir_mass: float, shape='septic') -> float:
        if reservoir_mass <= 0: raise ValueError('Positive reservoir mass required')
        if shape not in ('septic', 'bang'): raise ValueError('Unknown shape')
        norm = 700/429 if shape == 'septic' else 4/3
        return -self.mass**2/(reservoir_mass*HBAR) * 2*self.split**2/self.ramp * norm
    def force_norms(self):
        m,d,t = self.mass,self.split,self.ramp
        return dict(l1=35/8*m*d/t,
                    l2sq=140/11*m*m*d*d/t**3,
                    d1l1=336*math.sqrt(5)/25*m*d/t**2,
                    d2l1=273*m*d/t**3)
    def segments(self):
        """Force f(t)=m/2 * d''(t); coefficients use seconds within segment."""
        f = S.deriv(2) * (self.mass*self.split/(2*self.ramp**2))
        f = Polynomial([a/self.ramp**k for k,a in enumerate(f.coef)])
        return [(0.0, self.ramp, f),
                (self.ramp, self.hold, Polynomial([0.])),
                (self.ramp+self.hold, self.ramp, -f)]

def phase_pair(protocol: Protocol, shape='septic', hold_only=False, mp_dps=60):
    with mp.workdps(mp_dps):
        g,h=mp.mpf(str(G)),mp.mpf(str(HBAR))
        m,r,d,t,H=map(lambda x:mp.mpf(str(x)),
                      (protocol.mass,protocol.separation,protocol.split,protocol.ramp,protocol.hold))
        def k(e):
            x=d*e
            return 2/r-1/(r-x)-1/(r+x)
        if hold_only: return g*m*m/h*H*k(1)
        def env(u):
            if shape=='septic': return 35*u**4-84*u**5+70*u**6-20*u**7
            return 2*u*u if u<=mp.mpf('.5') else 1-2*(1-u)**2
        return g*m*m/h*(H*k(1)+2*t*mp.quad(lambda u:k(env(u)),[0,.5,1]))

def phase_casimir(protocol: Protocol, radius=1e-6, dielectric=5.7,
                  shape='septic', mp_dps=60):
    """Leading retarded electric-dipole sphere--sphere Casimir--Polder model.
    Alpha_vol = a^3*(eps-1)/(eps+2); U=-23 hbar c/(4pi)*Alpha_vol^2/r^7.
    This is NOT a certified finite-temperature/material error estimate.
    """
    with mp.workdps(mp_dps):
        r,d,t,H,a,e=map(lambda x:mp.mpf(str(x)),
          (protocol.separation,protocol.split,protocol.ramp,protocol.hold,radius,dielectric))
        pref=23*mp.mpf(str(C_LIGHT))/(4*mp.pi)*a**6*((e-1)/(e+2))**2
        def k(en):
            x=d*en
            return 2/r**7-1/(r-x)**7-1/(r+x)**7
        def env(u):
            if shape=='septic': return 35*u**4-84*u**5+70*u**6-20*u**7
            return 2*u*u if u<=mp.mpf('.5') else 1-2*(1-u)**2
        return pref*(H*k(1)+2*t*mp.quad(lambda u:k(env(u)),[0,.5,1]))

def support_phase(protocol: Protocol, M=1., height=.1, kind='sphere', length=1.,
                  shape='septic', mp_dps=70):
    """Rigid translating support, both branch labels included exactly.
    Sphere: exact external spherical potential. Line: uniform thin rod.
    Neither object is asserted to be part of Bose et al.'s apparatus.
    """
    with mp.workdps(mp_dps):
        m,R,D,tr,H,Mass,h,L=map(lambda x:mp.mpf(str(x)),
          (protocol.mass,protocol.separation,protocol.split,protocol.ramp,
           protocol.hold,M,height,length))
        if Mass <= 0 or h <= 0: raise ValueError('Positive mass and height required.')
        grav,HB=mp.mpf(str(G)),mp.mpf(str(HBAR))
        if kind=='sphere':
            phi=lambda x:-grav*Mass/mp.sqrt(x*x+h*h)
        elif kind=='line':
            phi=lambda x:-grav*Mass/L*(mp.asinh((L/2-x)/h)+mp.asinh((L/2+x)/h))
        else: raise ValueError('Unknown support potential')
        def k(en):
            d=D*en; eta=m/Mass; xp=(R+d)/2; xm=(R-d)/2
            return -2*m/HB*(phi(xp+eta*d)-phi(xp)+phi(xm-eta*d)-phi(xm))
        def env(u):
            if shape=='septic':return 35*u**4-84*u**5+70*u**6-20*u**7
            return 2*u*u if u<=mp.mpf('.5') else 1-2*(1-u)**2
        return H*k(1)+2*tr*mp.quad(lambda u:k(env(u)),[0,.5,1])

def sphere_bound(protocol:Protocol,M=1.,height=.1,shape='septic'):
    """Mixed-Hessian bound K<=2 G M / h^3, no small-displacement expansion."""
    n=quad(lambda u: (float(S(u)) if shape=='septic' else
             (2*u*u if u<=.5 else 1-2*(1-u)**2))**2,0,1,epsabs=1e-13)[0]
    integral=protocol.split**2*(protocol.hold+2*protocol.ramp*n)
    return 4*G*protocol.mass**2/HBAR*(1+protocol.mass/M)*integral/height**3

def exp_poly_integral(poly:Polynomial, duration:float, omega:float)->complex:
    """Exact endpoint formula, with ordinary quadrature near omega=0."""
    if omega==0: return complex(poly.integ()(duration)-poly.integ()(0))
    if abs(omega*duration)<20:
        re=quad(lambda t:poly(t)*math.cos(omega*t),0,duration,
                epsabs=1e-40,epsrel=5e-12,limit=150)[0]
        im=quad(lambda t:poly(t)*math.sin(omega*t),0,duration,
                epsabs=1e-40,epsrel=5e-12,limit=150)[0]
        return complex(re,im)
    e=np.exp(1j*omega*duration)
    total=0j
    for k in range(poly.degree()+1):
        der=poly.deriv(k)
        total+=(-1)**k*(der(duration)*e-der(0))/(1j*omega)**(k+1)
    return total

def force_transform(protocol:Protocol,omega:float)->complex:
    return sum(np.exp(1j*omega*start)*exp_poly_integral(p,T,omega)
               for start,T,p in protocol.segments())

def mode_action(protocol:Protocol,omega:float)->float:
    """Integral f(t) R(t) dt where R''+omega^2 R=f, R(0)=R'(0)=0.
    Exact piecewise-polynomial particular solution; no oscillatory time mesh.
    """
    if omega<=0: raise ValueError('Positive mode frequency required.')
    if omega*protocol.ramp < 20: return float(mode_action_mp(protocol,omega))
    pos=vel=action=0.
    for start,T,f in protocol.segments():
        particular=Polynomial([0.])
        for k in range(f.degree()//2+1):
            particular=particular+(-1)**k*f.deriv(2*k)/omega**(2*k+2)
        dp=particular.deriv()
        a=pos-particular(0); b=(vel-dp(0))/omega
        fp=(f*particular).integ()
        integ=exp_poly_integral(f,T,omega)
        action+=fp(T)-fp(0)+a*integ.real+b*integ.imag
        co,si=math.cos(omega*T),math.sin(omega*T)
        pos=particular(T)+a*co+b*si
        vel=dp(T)+omega*(-a*si+b*co)
    return float(action)

@dataclass(frozen=True)
class Rod:
    mass:float=1.
    length:float=1.
    speed:float=5000.
    temperature:float=4.
    port_separation:float=450e-6
    port_width:float=100e-6
    def __post_init__(self):
        if min(self.mass,self.length,self.speed,self.port_width) <= 0 or self.temperature<0:
            raise ValueError('Require positive rod parameters and nonnegative temperature.')
        if self.speed >= C_LIGHT: raise ValueError('The causal rod model requires sound speed below c.')
        if not 0<self.port_separation or self.port_separation+self.port_width>=self.length:
            raise ValueError('Both positive-width ports must lie inside the rod.')
        if self.port_width>=self.port_separation:
            raise ValueError('The two port supports must be disjoint.')
    @property
    def omega1(self):return math.pi*self.speed/self.length
    def c(self,n:int):
        xA=(self.length-self.port_separation)/2
        xB=(self.length+self.port_separation)/2
        profile=np.sinc(n*self.port_width/(2*self.length))
        return math.sqrt(2)*profile*np.cos(n*math.pi*np.array([xA,xB])/self.length)
    def phase_bound(self,p:Protocol,start=1):
        a=p.force_norms()
        E=2*(a['l2sq']+a['l1']*a['d1l1'])
        return 4*E/(self.mass*HBAR*self.omega1**2)*float(mp.zeta(2,start))
    def noise_bound(self,p:Protocol,start=1):
        a=p.force_norms()
        return 2*a['d2l1']**2/(self.mass*HBAR)*(float(mp.zeta(5,start))/self.omega1**5
          +2*KB*self.temperature/HBAR*float(mp.zeta(6,start))/self.omega1**6)
    def calculate(self,p:Protocol,N=100):
        phase=0.; V=np.zeros((2,2)); per_mode=[]
        for n in range(1,N+1):
            omega=n*self.omega1; co=self.c(n)
            da=4*co[0]*co[1]/(self.mass*HBAR)*mode_action(p,omega)
            alpha=-1j*co*force_transform(p,omega)/math.sqrt(2*self.mass*HBAR*omega)
            nu=1.0 if self.temperature==0 else 1/math.tanh(HBAR*omega/(2*KB*self.temperature))
            vn=nu*np.real(np.outer(alpha,alpha.conj()))
            phase+=da;V+=vn
            per_mode.append([n,da,float(vn.trace())])
        return dict(phase=phase, V=V, per_mode=per_mode,
          phase_tail_bound=self.phase_bound(p,N+1),
          noise_tail_bound=self.noise_bound(p,N+1))

def density_matrix(delta:float,V:np.ndarray|None=None)->np.ndarray:
    V=np.zeros((2,2)) if V is None else np.asarray(V,float)
    if V.shape!=(2,2) or np.linalg.eigvalsh(V).min() < -1e-10:
        raise ValueError('Noise covariance must be positive semidefinite, size 2x2.')
    phase=delta/4*SIGNS[:,0]*SIGNS[:,1]
    rho=np.empty((4,4),complex)
    for i,z in enumerate(SIGNS):
        for j,w in enumerate(SIGNS):
            dz=z-w
            rho[i,j]=np.exp(1j*(phase[i]-phase[j])-.5*dz@V@dz)/4
    return rho

def partial_transpose(rho):
    return np.asarray(rho).reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)

def negativity(rho):
    return float(np.maximum(-np.linalg.eigvalsh(partial_transpose(rho)),0).sum())

def trace_distance(a,b):return float(np.linalg.svd(np.asarray(a)-np.asarray(b),compute_uv=False).sum()/2)

def witness_matrix(sign=1):
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1])
    return (np.eye(4)-np.kron(X,X)+sign*(np.kron(Y,Z)+np.kron(Z,Y)))/4

def witness_value(delta,V):
    vA=math.exp(-2*V[0,0]);vB=math.exp(-2*V[1,1])
    vp=math.exp(-2*(V[0,0]+V[1,1]+2*V[0,1]))
    vm=math.exp(-2*(V[0,0]+V[1,1]-2*V[0,1]))
    return ((vp+vm)/2-1+(vA+vB)*abs(math.sin(delta/2)))/4

def synchronous_noise_distance(noise_trace:float)->float:
    """Trace-distance bound for real Gaussian random local-Z dephasing.
    The CP semigroup estimate is 2 Tr(V); purification also gives sqrt(Tr V).
    This sharpening is NOT used for general complex Weyl overlap phases.
    """
    if noise_trace < 0: raise ValueError('A noise trace cannot be negative.')
    return min(1., 2*noise_trace, math.sqrt(noise_trace))


def rod_green_images(t:float,x:float,y:float,rod:Rod)->float:
    """Point-port retarded Neumann Green function, exact finite image sum.
    Values precisely on wavefronts use the symmetric Heaviside convention.
    Finite-width port response is the average of this kernel over both ports.
    """
    if not (0<=x<=rod.length and 0<=y<=rod.length):
        raise ValueError('Point ports must be inside the rod.')
    if t<0:return 0.
    L,v=rod.length,rod.speed
    K=math.ceil((v*t+2*L)/(2*L))+1
    count=0.
    for k in range(-K,K+1):
        for distance in [abs(x-y+2*k*L),abs(x+y+2*k*L)]:
            arrival=distance/v
            count+=1. if t>arrival else (.5 if t==arrival else 0.)
    return L/(2*rod.mass*v)*count

def mode_action_mp(protocol:Protocol,omega:float,dps=70):
    """High-precision independent polynomial implementation of the mode action.
    Used at small omega to avoid cancellation in the particular solution.
    """
    if omega<=0:raise ValueError('Positive mode frequency required.')
    with mp.workdps(dps):
        w=mp.mpf(str(omega));pos=vel=action=mp.mpf(0)
        def der(a,k=1):
            b=a[:]
            for _ in range(k):b=[(i+1)*b[i+1] for i in range(len(b)-1)] or [mp.mpf(0)]
            return b
        def val(a,t):return mp.polyval(a[::-1],t)
        for _,duration,poly in protocol.segments():
            T=mp.mpf(str(duration));f=[mp.mpf(str(c)) for c in poly.coef]
            p=[mp.mpf(0)]*len(f)
            for k in range((len(f)-1)//2+1):
                for j,c in enumerate(der(f,2*k)):p[j]+=(-1)**k*c/w**(2*k+2)
            dp=der(p);a=pos-p[0];b=(vel-dp[0])/w
            integral=mp.mpc(0);e=mp.exp(1j*w*T)
            for k in range(len(f)):
                df=der(f,k)
                integral+=(-1)**k*(val(df,T)*e-df[0])/(1j*w)**(k+1)
            fpint=sum(f[i]*p[j]*T**(i+j+1)/(i+j+1) for i in range(len(f)) for j in range(len(p)))
            action+=fpint+a*integral.real+b*integral.imag
            co,si=mp.cos(w*T),mp.sin(w*T)
            pos=val(p,T)+a*co+b*si;vel=val(dp,T)+w*(-a*si+b*co)
        return +action
