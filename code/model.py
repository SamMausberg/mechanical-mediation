"""Response-based recoil calibration in SI units.

Quadratic Gaussian reservoirs, prescribed local nondemolition forces.
Numerical mounting models are reference models, not measured QGEM cross responses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

HBAR = 6.62607015e-34 / (2*math.pi)
G = 6.67430e-11
C = 299792458.0
EPS0 = 8.8541878188e-12
E_CHARGE = 1.602176634e-19
S = Polynomial([0,0,0,0,35,-84,70,-20])
SIGNS = np.array([[1,1],[1,-1],[-1,1],[-1,-1]],dtype=float)

@dataclass(frozen=True)
class Pulse:
    """Closed inertial separation. Its net reaction force is -m d''/2."""
    mass: float = 1e-14
    split: float = 250e-6
    ramp: float = .5
    hold: float = 2.5
    delay: float = 0.
    def __post_init__(self):
        if self.mass <= 0 or self.ramp <= 0 or self.hold < 0 or self.delay < 0:
            raise ValueError('Positive mass/ramp and nonnegative hold/delay required.')
    @property
    def end(self): return self.delay+2*self.ramp+self.hold
    @property
    def knots(self):
        return [self.delay,self.delay+self.ramp,self.delay+self.ramp+self.hold,self.end]
    @property
    def scale(self): return abs(self.mass*self.split/(2*self.ramp**2))
    def d(self,t,derivative=0):
        t=t-self.delay
        if t<0 or t>2*self.ramp+self.hold: return 0.
        if t<=self.ramp:
            return self.split*float(S.deriv(derivative)(t/self.ramp))/self.ramp**derivative
        if t<=self.ramp+self.hold:
            return self.split if derivative==0 else 0.
        u=(t-self.ramp-self.hold)/self.ramp
        if derivative==0: return self.split*(1-float(S(u)))
        return -self.split*float(S.deriv(derivative)(u))/self.ramp**derivative
    def force(self,t): return -.5*self.mass*self.d(t,2)
    def norms(self):
        m,d,t=self.mass,abs(self.split),self.ramp
        return {'l1':35/8*m*d/t,
                'l2sq':140/11*m*m*d*d/t**3,
                'velocity2':1400/429*d*d/t,
                'force_peak':42*math.sqrt(5)/25*m*d/t**2}

@dataclass(frozen=True)
class Wave:
    """A real linear combination of closed separation loops."""
    terms: tuple[tuple[float,Pulse], ...]
    @property
    def end(self): return max(p.end for _,p in self.terms)
    @property
    def knots(self): return sorted(set(k for _,p in self.terms for k in p.knots))
    @property
    def ramp(self): return min(p.ramp for _,p in self.terms)
    @property
    def scale(self): return sum(abs(a)*p.scale for a,p in self.terms)
    def d(self,t,derivative=0): return sum(a*p.d(t,derivative) for a,p in self.terms)
    def force(self,t): return sum(a*p.force(t) for a,p in self.terms)

@dataclass(frozen=True)
class Mount:
    mass: float
    frequency: float=0.
    damping: float=0.
    def __post_init__(self):
        if self.mass<=0 or self.frequency<0 or self.damping<0:
            raise ValueError('Positive mass and nonnegative frequency/damping required.')
    @property
    def omega(self): return 2*math.pi*self.frequency
    def susceptibility(self,w):
        return 1/(self.mass*(self.omega**2-w*w-2j*self.damping*self.omega*w))
    def hinf(self):
        if self.frequency==0 or self.damping==0: return math.inf
        a=1. if self.damping>=1/math.sqrt(2) else 2*self.damping*math.sqrt(1-self.damping**2)
        return 1/(self.mass*self.omega**2*a)

def directed_actions(a: Pulse|Wave,b: Pulse|Wave,mount: Mount,rtol=2e-11,atol=2e-13):
    """I_AB and I_BA using a scaled causal initial-value problem.

    Zero-force intervals propagate exactly. All nonzero intervals use DOP853.
    No assumed equilibrium initial displacement enters the retarded response.
    """
    tau=min(a.ramp,b.ramp)
    fs=max(a.scale,b.scale)
    if fs==0: return (0.,0.)
    end=max(a.end,b.end)
    knots=sorted(set([0.,end]+a.knots+b.knots))
    knots=[k/tau for k in knots if 0<=k<=end]
    om=mount.omega*tau; gam=mount.damping*om
    state=np.zeros(6) # qa, qa', qb, qb', Iab, Iba in dimensionless time
    def rhs(u,y):
        fa=a.force(u*tau)/fs; fb=b.force(u*tau)/fs
        return [y[1],fa-2*gam*y[1]-om*om*y[0],
                y[3],fb-2*gam*y[3]-om*om*y[2],fa*y[2],fb*y[0]]
    mat=np.array([[0.,1.],[-om*om,-2*gam]])
    for lo,hi in zip(knots[:-1],knots[1:]):
        if hi-lo<1e-13: continue
        points=[(lo+(hi-lo)*v)*tau for v in (.17,.37,.73)]
        empty=all(abs(a.force(t))+abs(b.force(t))<fs*1e-25 for t in points)
        if empty:
            prop=expm(mat*(hi-lo))
            state[:2]=prop@state[:2];state[2:4]=prop@state[2:4]
        else:
            sol=solve_ivp(rhs,(lo,hi),state,method='DOP853',rtol=rtol,atol=atol,
                          max_step=min((hi-lo)/12, .7/max(om,1e-12)))
            if not sol.success: raise RuntimeError(sol.message)
            state=sol.y[:,-1]
    scale=fs*fs*tau**3/mount.mass
    return float(state[4]*scale),float(state[5]*scale)

def phase_and_bound(actions):
    ab,ba=actions
    return {'phase':2*(ab+ba)/HBAR,'weyl':(ab-ba)/(2*HBAR),
            'beta':min(1.,max(abs(ab),abs(ba))/HBAR),
            'Iab':ab,'Iba':ba}

def free_phase(p:Pulse,M:float):
    return -p.mass**2*p.norms()['velocity2']/(M*HBAR)

def integrate_knots(fun,knots):
    out=0.;error=0.
    for lo,hi in zip(knots[:-1],knots[1:]):
        if hi>lo:
            v,e=quad(fun,lo,hi,epsabs=1e-13,epsrel=2e-12,limit=200)
            out+=v;error+=e
    return out,error

def pair_phase(a:Pulse|Wave,b:Pulse|Wave,separation:float,geometry='collinear',distance=None):
    """Prescribed-path Newtonian invariant for equal branch midpoints.

    distance(t) supplies the changing transverse gap of a screened geometry.
    For a Wave all component masses must agree.
    """
    ma=a.mass if isinstance(a,Pulse) else a.terms[0][1].mass
    mb=b.mass if isinstance(b,Pulse) else b.terms[0][1].mass
    def kernel(t):
        da,db=a.d(t),b.d(t)
        r=separation if distance is None else distance(t)
        u=(da-db)/2;v=(da+db)/2
        if geometry=='collinear':
            if r<=max(abs(u),abs(v)): raise ValueError('Branches intersect.')
            # Stable difference: S(u)-S(v), S(x)=2r/(r^2-x^2).
            return 2*r*(u*u-v*v)/((r*r-u*u)*(r*r-v*v))
        if geometry=='parallel':
            x=math.sqrt(r*r+u*u);y=math.sqrt(r*r+v*v)
            return 2*(v*v-u*u)/(x*y*(x+y))
        raise ValueError('Unknown geometry')
    knots=sorted(set([0.,max(a.end,b.end)]+a.knots+b.knots))
    # Integrate a dimensionless kernel to keep quadrature tolerances meaningful.
    val,err=integrate_knots(lambda t:kernel(t)*separation,knots)
    return G*ma*mb/(HBAR*separation)*val

def screened_gap(duration=1.,mass=1e-14,z0=41e-6,thickness=1e-6,
                 dielectric=5.1,density=3500.,dipole=.01*E_CHARGE*.01):
    """Schut et al. Eqs. (14),(16), including worst-case aligned dipoles.

    Newtonian trajectory under the published ideal point-sphere/plane model;
    not a material or Casimir error certificate.
    """
    ac=3*HBAR*C/(2*math.pi)*(dielectric-1)/(dielectric+2)*3/(4*math.pi*density)
    ad=(1/(4*math.pi*EPS0))*3*dipole**2/(8*mass)
    def rhs(t,y):
        return [y[1],-ac/y[0]**5-ad/y[0]**4]
    sol=solve_ivp(rhs,(0.,duration),[z0,0.],rtol=5e-12,atol=1e-16,
                  max_step=.001,dense_output=True)
    if not sol.success or sol.y[0,-1]<=0: raise RuntimeError('Plate collision or failed integration')
    return lambda t:float(2*sol.sol(t)[0]+thickness), float(sol.y[0,-1])

def multiplier(delta:float,sigma:float,V:np.ndarray):
    V=np.asarray(V,dtype=float)
    if V.shape!=(2,2): raise ValueError('V must be 2 by 2.')
    z=SIGNS
    diff=z[:,None,:]-z[None,:,:]
    gate=delta/4*((z[:,0]*z[:,1])[:,None]-(z[:,0]*z[:,1])[None,:])
    wedge=sigma*(z[:,0,None]*z[None,:,1]-z[:,1,None]*z[None,:,0])
    noise=np.einsum('abi,ij,abj->ab',diff,V,diff)/2
    return np.exp(1j*(gate+wedge)-noise)

def trace_distance(rho,sigma):
    a=(rho-sigma);a=(a+a.conj().T)/2
    return float(np.abs(np.linalg.eigvalsh(a)).sum()/2)

def negativity(rho):
    pt=rho.reshape(2,2,2,2).transpose(0,3,2,1).reshape(4,4)
    return float(max(0.,(-np.linalg.eigvalsh(pt)).clip(min=0).sum()))

def witness(sign=1):
    x=np.array([[0,1],[1,0]],complex)
    y=np.array([[0,-1j],[1j,0]],complex)
    z=np.diag([1.,-1.])
    return (np.eye(4)-np.kron(x,x)+sign*(np.kron(y,z)+np.kron(z,y)))/4

def calibration_bound(Iab,Iba,error_ab=0.,error_ba=0.):
    if min(error_ab,error_ba)<0: raise ValueError('Error bounds must be nonnegative.')
    return min(1.,max(abs(Iab)+error_ab,abs(Iba)+error_ba)/HBAR)
