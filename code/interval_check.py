"""Outward interval checks for the damped-mode actions and pulse null.

Uses mpmath.iv interval elementary functions and exact rational forcing
polynomials. This is numerical interval verification, not formal verification
of the interval library. No ODE tolerance is interpreted as a rigorous bound.
"""
from fractions import Fraction as F
from math import comb
import json
from pathlib import Path
import mpmath as mp
from model import Pulse

iv=mp.iv
iv.dps=70

def R(x):
    x=x if isinstance(x,F) else F(str(x))
    return iv.mpf(x.numerator)/x.denominator

def poly_eval(p,x):
    s=iv.mpf(0)
    for a in reversed(p):s=s*x+a
    return s

def poly_deriv(p):return [(j+1)*p[j+1] for j in range(len(p)-1)] or [iv.mpf(0)]

def exact_force(p,t0,mid):
    """Force coefficients in seconds from t0; branch choices use rational mid."""
    m,d,t,H,start=map(lambda x:F(str(x)),(p.mass,p.split,p.ramp,p.hold,p.delay))
    if start<=mid<start+t:shift=(t0-start)/t;sign=-1
    elif start+t+H<=mid<start+2*t+H:shift=(t0-start-t-H)/t;sign=1
    else:return [iv.mpf(0)]*6
    coeff=[0,0,420,-1680,2100,-840] # S''
    out=[F(0)]*6
    for j,c in enumerate(coeff):
        for k in range(j+1):
            out[k]+=sign*m*d/(2*t*t)*c*comb(j,k)*shift**(j-k)/t**k
    return [R(a) for a in out]

def particular(f,om,gamma):
    q=[iv.mpf(0)]*(len(f)+2)
    for j in reversed(range(len(f))):
        q[j]=(f[j]-2*gamma*(j+1)*q[j+1]-(j+2)*(j+1)*q[j+2])/(om*om)
    return q[:-2]

def moments(z,L,n):
    e=iv.exp(z*L);out=[(e-1)/z]
    for k in range(1,n+1):out.append(L**k*e/z-k*out[-1]/z)
    return out

def integral_product(a,b,L):
    return sum((x*y*L**(i+j+1)/(i+j+1)
                for i,x in enumerate(a) for j,y in enumerate(b)),iv.mpf(0))

def actions(a,b,mass='187',freq='.42',damping='.58'):
    M=R(mass);om=2*iv.pi*R(freq);ga=R(damping)*om
    wd=om*iv.sqrt(1-R(damping)**2);z=iv.mpc(-ga,wd)
    def k(p):
        s,t,H=map(lambda x:F(str(x)),(p.delay,p.ramp,p.hold))
        return [s,s+t,s+t+H,s+2*t+H]
    knots=sorted(set([F(0)]+k(a)+k(b)))
    qa=va=qb=vb=iab=iba=iv.mpf(0)
    for lo,hi in zip(knots[:-1],knots[1:]):
        L=R(hi-lo);mid=(lo+hi)/2
        fa=exact_force(a,lo,mid);fb=exact_force(b,lo,mid)
        pa=particular(fa,om,ga);pb=particular(fb,om,ga)
        da=poly_deriv(pa);db=poly_deriv(pb)
        Aa=qa-pa[0];Ba=(va-da[0]+ga*Aa)/wd
        Ab=qb-pb[0];Bb=(vb-db[0]+ga*Ab)/wd
        mom=moments(z,L,5)
        ea=sum((f*x for f,x in zip(fa,mom)),iv.mpc(0))
        eb=sum((f*x for f,x in zip(fb,mom)),iv.mpc(0))
        iab+=integral_product(fa,pb,L)+Ab*ea.real+Bb*ea.imag
        iba+=integral_product(fb,pa,L)+Aa*eb.real+Ba*eb.imag
        co=iv.cos(wd*L);si=iv.sin(wd*L);ex=iv.exp(-ga*L)
        qa=poly_eval(pa,L)+ex*(Aa*co+Ba*si)
        va=poly_eval(da,L)+ex*((-ga*Aa+wd*Ba)*co+(-ga*Ba-wd*Aa)*si)
        qb=poly_eval(pb,L)+ex*(Ab*co+Bb*si)
        vb=poly_eval(db,L)+ex*((-ga*Ab+wd*Bb)*co+(-ga*Bb-wd*Ab)*si)
    hb=R('6.62607015e-34')/(2*iv.pi)
    return iab/(M*hb),iba/(M*hb)

def bounds(x):
    # repr of interval endpoints: strings retain the interval library's digits.
    return {'lower':str(x.a),'upper':str(x.b)}

def main():
    a=Pulse();b=Pulse(delay=.5)
    gs=[Pulse(ramp=.25,hold=0),Pulse(ramp=.5,hold=0,delay=3)]
    b0,b1=actions(a,b);c00,c10=actions(a,gs[0]);c01,c11=actions(a,gs[1])
    det=c00*c11-c01*c10
    assert det.a>0
    x0=(-c11*b0+c01*b1)/det
    x1=(c10*b0-c00*b1)/det
    assert x0.a>0 and x0.b<1 and x1.a>-1 and x1.b<0
    # Eight-decimal controls, with rigorous action error from rounding.
    x0r=R('0.42502996');x1r=R('-0.45682594')
    r0=b0+c00*x0r+c01*x1r;r1=b1+c10*x0r+c11*x1r
    sync=actions(a,a)
    out={'precision_digits':iv.dps,'I_over_hbar_staggered':[bounds(b0),bounds(b1)],
         'matrix':[[bounds(c00),bounds(c01)],[bounds(c10),bounds(c11)]],
         'determinant':bounds(det),'coefficient_1':bounds(x0),'coefficient_2':bounds(x1),
         'rounded_residuals_over_hbar':[bounds(r0),bounds(r1)],
         'sync_phase':bounds(2*(sync[0]+sync[1]))}
    path=Path(__file__).resolve().parents[1]/'results/intervals.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print('Interval determinant:',det)
    print('Coefficients:',x0,x1)
    print('Rounded action residuals / hbar:',r0,r1)
    print('Synchronized phase:',2*(sync[0]+sync[1]))

if __name__=='__main__':main()
