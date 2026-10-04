"""Independent checks of the passive-compliance bounds and local bar model.

The tests check formulas and numerical implementations. The proofs are in the
paper; none of these checks establishes that a fabricated mount is this model.
"""
import math
import unittest
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import zeta
from numpy.testing import assert_allclose
from model import Pulse, S, HBAR, G, pair_phase, multiplier, witness
from design import FixedBar, pulse_extra, static_bounds, gravity_quadratic_interval
from elastic import Protocol, mode_action, force_transform, Rod


def osc_action(g, f, omega, derivative=0):
    """Independent causal ODE for J(g,f), using unit-mass mode response."""
    knots = sorted(set([0., max(g.end, f.end)] + g.knots + f.knots))
    y = np.zeros(3)
    def force(p, t):
        return -p.mass * p.d(t, 2 + derivative) / 2
    def rhs(t, q):
        return [q[1], force(f,t) - omega**2*q[0], force(g,t)*q[0]]
    for lo, hi in zip(knots[:-1], knots[1:]):
        if hi <= lo:
            continue
        sol=solve_ivp(rhs, (lo,hi), y, method='DOP853', rtol=3e-12,
                      atol=2e-13, max_step=min((hi-lo)/20, .1/omega))
        if not sol.success:
            raise RuntimeError(sol.message)
        y=sol.y[:,-1]
    return y[-1]


def overlap(g, f):
    knots=sorted(set([0., max(g.end,f.end)]+g.knots+f.knots))
    return sum(quad(lambda t:g.force(t)*f.force(t),a,b,
                    epsabs=1e-10, epsrel=2e-12)[0]
               for a,b in zip(knots[:-1],knots[1:]) if b>a)


class PassiveTests(unittest.TestCase):
    def setUp(self):
        self.a=Pulse(mass=1,split=1,ramp=.4,hold=.3)
        self.b=Pulse(mass=1,split=.7,ramp=.3,hold=.2,delay=.2)
    def test_derivative_identity(self):
        for omega in [1., 4.3, 15.]:
            j=osc_action(self.a,self.b,omega)
            jd=osc_action(self.a,self.b,omega,derivative=1)
            assert_allclose(j, (overlap(self.a,self.b)+jd)/omega**2,
                            rtol=2e-9, atol=2e-10)
    def test_mode_absolute_bound(self):
        na,nb=self.a.norms(),self.b.norms()
        db=pulse_extra(self.b)['d1l1']
        for omega in [1.,4.3,15.]:
            j=osc_action(self.a,self.b,omega)
            bound=(math.sqrt(na['l2sq']*nb['l2sq'])+na['l1']*db)/omega**2
            self.assertLess(abs(j),bound)
    def test_mode_static_remainder(self):
        na,nb=pulse_extra(self.a),pulse_extra(self.b)
        for omega in [1.,4.3,15.]:
            remainder=abs(osc_action(self.a,self.b,omega)-overlap(self.a,self.b)/omega**2)
            bound=(math.sqrt(na['d1l2sq']*nb['d1l2sq'])+na['d1l1']*nb['d2l1'])/omega**4
            self.assertLess(remainder,bound)
    def test_spectral_cauchy(self):
        rng=np.random.default_rng(6021)
        for _ in range(20):
            c=rng.normal(size=(2,100));w=rng.uniform(.1,10,size=100)
            weighted=np.sum(np.abs(c[0]*c[1])/w**2)
            self.assertLessEqual(weighted,np.sqrt(np.sum(c[0]**2/w**2)*np.sum(c[1]**2/w**2)))
    def test_mode_formula_vs_independent_ode(self):
        p=Protocol(mass=1,separation=2,split=1,ramp=.4,hold=.3)
        for w in [7.,60.]:
            assert_allclose(mode_action(p,w),osc_action(self.a,self.a,w),rtol=2e-8,atol=2e-10)


class SmoothNormTests(unittest.TestCase):
    def test_polynomial_squared_derivative_constants(self):
        for derivative, exact in [(1,700/429),(2,280/11),(3,1120),(4,100800)]:
            p=(S.deriv(derivative)**2).integ()
            assert_allclose(p(1)-p(0),exact,rtol=1e-10)
    def test_total_variations(self):
        roots=[0.,.5-math.sqrt(5)/10,.5+math.sqrt(5)/10,1.]
        value=sum(quad(lambda u:abs(S.deriv(3)(u)),lo,hi,epsabs=1e-9)[0]
                  for lo,hi in zip(roots[:-1],roots[1:]))
        assert_allclose(value,336*math.sqrt(5)/25,rtol=2e-12)
        roots=[0.,.5-math.sqrt(15)/10,.5,.5+math.sqrt(15)/10,1.]
        value=sum(quad(lambda u:abs(S.deriv(4)(u)),lo,hi,epsabs=1e-9)[0]
                  for lo,hi in zip(roots[:-1],roots[1:]))
        assert_allclose(value,273,rtol=2e-12)
    def test_integrated_separation(self):
        p=(S*S).integ()
        assert_allclose(2*(p(1)-p(0)),1042/1287,rtol=1e-10)
    def test_derivative_scaling(self):
        p=Pulse(mass=2,split=3,ramp=.4,hold=.2)
        e=pulse_extra(p)
        exact=560*p.mass**2*p.split**2/p.ramp**5
        assert_allclose(e['d1l2sq'],exact,rtol=1e-14)
        assert_allclose(e['d2l1'],273*p.mass*p.split/p.ramp**3,rtol=1e-14)


class BarTests(unittest.TestCase):
    def setUp(self):self.bar=FixedBar()
    def test_material_numbers(self):
        b=self.bar
        assert_allclose(b.young,199533626924.58542,rtol=1e-14)
        assert_allclose(b.mass,.08,rtol=1e-14)
        assert_allclose(b.omega1/(2*math.pi),24970.8346708,rtol=1e-10)
    def test_static_cross_vs_modes(self):
        b=self.bar;N=10000
        err=2*float(zeta(2,N+1))/(b.mass*b.omega1**2)
        self.assertLess(abs(b.static_mode_sum(N)-b.cross),err)
    def test_static_self_vs_modes(self):
        b=self.bar;N=10000
        val=sum(b.coupling(n)[0]**2/(b.mass*(n*b.omega1)**2) for n in range(1,N+1))
        err=2*float(zeta(2,N+1))/(b.mass*b.omega1**2)
        self.assertLess(abs(val-b.self_compliance),err)
    def test_static_patch_integral(self):
        b=self.bar;x=b.length/3;w=b.port_width
        # Integrate the two triangular pieces of the Dirichlet static kernel.
        def inner(y):
            left=quad(lambda z:z-z*y/b.length,x-w/2,y,epsabs=1e-18)[0]
            right=quad(lambda z:y-z*y/b.length,y,x+w/2,epsabs=1e-18)[0]
            return left+right
        val=quad(inner,x-w/2,x+w/2,epsabs=1e-18)[0]/(w*w*b.young*b.area)
        assert_allclose(val,b.self_compliance,rtol=2e-12)
    def test_dirichlet_causal_cone(self):
        # Infinite image series has only these terms at the short times used.
        L=1.;v=2.;x=.2;y=.8
        def images(t):
            return sum(float(t>=abs(x-y+2*k*L)/v)-float(t>=abs(x+y+2*k*L)/v)
                       for k in range(-8,9))
        for t in [.01,.1,.29]:self.assertEqual(images(t),0)
        self.assertEqual(images(.31),1)
        # Fixed-end boundary values vanish exactly after every arrival.
        x=0.
        for t in [.3,.7,1.1,2.1]:self.assertEqual(images(t),0)
    def test_designed_fast_bounds(self):
        p=Pulse(split=1e-6,ramp=.001,hold=.14);b=self.bar
        r=static_bounds(p,b.cross,b.self_compliance,b.self_compliance,b.omega1)
        g=abs(pair_phase(p,p,100e-6))
        self.assertGreater((r['phase_static']-r['phase_error'])/g,1.42)
        self.assertLess((r['phase_static']+r['phase_error'])/g,1.60)
        p=Pulse(split=1e-6,ramp=.01,hold=.14)
        r=static_bounds(p,b.cross,b.self_compliance,b.self_compliance,b.omega1)
        self.assertLess((r['phase_static']+r['phase_error'])/abs(pair_phase(p,p,100e-6)),.00144)
    def test_gravity_interval(self):
        for p,r in [(Pulse(),450e-6),(Pulse(split=1e-6,ramp=.001,hold=.14),100e-6)]:
            lo,hi=gravity_quadratic_interval(p,r);g=abs(pair_phase(p,p,r))
            self.assertLess(lo,g);self.assertLess(g,hi)
    def test_anchor_actuator_loads(self):
        p=Pulse(mass=2,split=3,ramp=.4,hold=.2);w=4.43
        for t in np.linspace(0,p.end,71):
            fa=-p.mass*(p.d(t,2)+w*w*p.d(t))/2
            fh=p.mass*w*w*p.d(t)/2
            assert_allclose(fa+fh,p.force(t),atol=2e-13,rtol=3e-14)
    def test_coordinate_invariance(self):
        rng=np.random.default_rng(434)
        f,g=rng.normal(size=(2,4));chi=rng.normal(size=(4,4))
        L=rng.normal(size=(4,4))+3*np.eye(4)
        assert_allclose(np.linalg.solve(L.T,f)@(L@chi@L.T)@np.linalg.solve(L.T,g),f@chi@g,rtol=1e-13)
    def test_invalid_design(self):
        for kw in [{'length':0},{'area':-1},{'port_width':.05}]:
            with self.assertRaises(ValueError):FixedBar(**kw)
    def test_covariance_witness_upper_bound(self):
        rng=np.random.default_rng(178)
        for _ in range(30):
            x=rng.normal(size=(2,2))*.1;V=x@x.T;S0=np.trace(V)
            delta=rng.uniform(-3,3);sign=1 if delta>=0 else -1
            rho=multiplier(delta,0,V)/4
            measured=-np.trace(witness(sign)@rho).real
            upper=(np.exp(-4*S0)-1)/8+abs(np.sin(delta/2))/2
            self.assertLessEqual(measured,upper+1e-14)
    def test_free_elastic_negativity(self):
        p=Protocol();M=abs(p.recoil_phase(1))/math.pi
        r=Rod(mass=M,length=1.1,speed=5000,temperature=4)
        self.assertGreater(.5-r.phase_bound(p)/4-2*r.noise_bound(p),.49998)

if __name__=='__main__':unittest.main(verbosity=2)
