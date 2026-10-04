"""Passive compliance bounds and explicitly specified horizontal mount.

Every geometric choice here is a design input, not measured QGEM hardware.
Young's modulus is NIST's 304 stainless fit evaluated at 293 K.
The density 8000 kg/m^3 is a nominal design choice.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad
from numpy.polynomial import Polynomial
from model import Pulse, S, HBAR, G, pair_phase

def poly_abs_integral(p: Polynomial) -> float:
    roots=[r.real for r in p.roots() if abs(r.imag)<1e-8 and 0<r.real<1]
    points=[0.]+sorted(roots)+[1.]
    ip=p.integ()
    return sum(abs(ip(b)-ip(a)) for a,b in zip(points[:-1],points[1:]))

def pulse_extra(p: Pulse) -> dict[str,float]:
    m,d,t=p.mass,abs(p.split),p.ramp
    # Extension by zero is used. f'' can jump at the four ramp endpoints.
    return dict(d1l2sq=560*m*m*d*d/t**5, d2l1=273*m*d/t**3, d1l1=m*d/t**2*poly_abs_integral(S.deriv(3)),
        d2l2=math.sqrt(m*m*d*d/(2*t**7)*(S.deriv(4)**2).integ()(1)),
        tvd2=m*d/t**4*poly_abs_integral(S.deriv(5))
              +2*m*d/t**4*abs(S.deriv(4)(0)))

def static_bounds(p: Pulse,cross:float,ca:float,cb:float,omega1:float) -> dict[str,float]:
    n=p.norms(); e=pulse_extra(p)
    weighted=math.sqrt(ca*cb)
    absolute=weighted*(n['l2sq']+n['l1']*e['d1l1'])
    # Exact all-mode remainder under a strict gap and spectral passivity.
    remainder=weighted/omega1**2*(e['d1l2sq']+e['d1l1']*e['d2l1'])
    leading=cross*n['l2sq']
    return dict(action_static=leading,action_error=remainder,
                phase_static=4*leading/HBAR,phase_error=4*remainder/HBAR,
                beta_passive=absolute/HBAR)

@dataclass(frozen=True)
class FixedBar:
    length: float=.1
    area: float=1e-4
    density: float=8000.
    young: float=(210.0593+.1534883*293-.001617390*293**2+5.117060e-6*293**3-6.154600e-9*293**4)*1e9
    port_width: float=.001
    def __post_init__(self):
        if min(self.length,self.area,self.density,self.young,self.port_width)<=0:
            raise ValueError('Positive material and geometric inputs required.')
        if self.port_width>=self.length/3:raise ValueError('Ports must be disjoint and interior.')
    @property
    def mass(self): return self.density*self.area*self.length
    @property
    def speed(self): return math.sqrt(self.young/self.density)
    @property
    def omega1(self): return math.pi*self.speed/self.length
    @property
    def cross(self): return self.length/(9*self.young*self.area)
    @property
    def self_compliance(self):return (2*self.length/9-self.port_width/6)/(self.young*self.area)
    def coupling(self,n):
        q=np.sinc(n*self.port_width/(2*self.length))
        return math.sqrt(2)*q*np.sin(n*math.pi*np.array([1/3,2/3]))
    def static_mode_sum(self,N):
        return sum(np.prod(self.coupling(n))/(self.mass*(n*self.omega1)**2) for n in range(1,N+1))

def gravity_quadratic_interval(p:Pulse,R:float):
    # True collinear phase magnitude lies in this analytic interval.
    s2=(S*S).integ()(1)
    q=2*G*p.mass**2*p.split**2*(p.hold+2*p.ramp*s2)/(HBAR*R**3)
    return q,q/(1-(p.split/R)**2)

def design_row(p:Pulse,R:float,bar:FixedBar):
    dg=pair_phase(p,p,R)
    wg=abs(math.sin(dg/2))/2
    b=static_bounds(p,bar.cross,bar.self_compliance,bar.self_compliance,bar.omega1)
    return {**b,'ramp':p.ramp,'gravity_phase':dg,'gravity_witness':wg,
        'one_percent_compliance':.01*wg*HBAR/p.norms()['l2sq'],
        'ratio_static':abs(b['phase_static']/dg),
        'ratio_error':abs(b['phase_error']/dg)}

