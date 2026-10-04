"""Executed consistency tests. These do not replace the analytic proofs."""
import math
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from numpy.testing import assert_allclose
from model import *

class PulseTests(unittest.TestCase):
    def setUp(self): self.p=Pulse()
    def test_endpoint_closure(self):
        p=self.p
        for t in (0,p.end):
            for k in range(4): self.assertLess(abs(p.d(t,k)),1e-12)
    def test_force_moments(self):
        p=self.p;scale=p.scale
        for k in (0,1):
            v,_=integrate_knots(lambda t:t**k*p.force(t)/scale,p.knots)
            self.assertLess(abs(v),1e-11)
    def test_force_l1(self):
        p=self.p
        v=quad(lambda u:abs(float(S.deriv(2)(u))),0,1,points=[.5],epsabs=1e-12)[0]
        assert_allclose(p.scale*2*p.ramp*v,p.norms()['l1'],rtol=1e-12)
    def test_force_l2(self):
        p=self.p;v=(S.deriv(2)**2).integ()(1)
        assert_allclose(2*p.scale**2*p.ramp*v,p.norms()['l2sq'],rtol=1e-10)
    def test_peak(self):
        t=self.p.ramp*(.5-math.sqrt(5)/10)
        assert_allclose(abs(self.p.force(t)),self.p.norms()['force_peak'],rtol=1e-12)
    def test_velocity_norm(self):
        v,_=integrate_knots(lambda t:(self.p.d(t,1)/self.p.split)**2,self.p.knots)
        assert_allclose(v*self.p.split**2,self.p.norms()['velocity2'],rtol=1e-11)
    def test_invalid_mount(self):
        with self.assertRaises(ValueError):Mount(-1)
    def test_invalid_pulse(self):
        with self.assertRaises(ValueError):Pulse(ramp=0)

class ResponseTests(unittest.TestCase):
    def test_free_exact(self):
        for p in [Pulse(),Pulse(split=1e-6,ramp=.001,hold=.14)]:
            delta=phase_and_bound(directed_actions(p,p,Mount(400)))['phase']
            assert_allclose(delta,free_phase(p,400),rtol=2e-9)
    def test_mass_scaling(self):
        p=Pulse();x=directed_actions(p,p,Mount(1,.42,.58))
        y=directed_actions(p,p,Mount(187,.42,.58))
        assert_allclose(np.array(x),np.array(y)*187,rtol=1e-13,atol=0)
    def test_exchange(self):
        a=Pulse();b=Pulse(delay=.5);m=Mount(187,.42,.58)
        x=directed_actions(a,b,m);y=directed_actions(b,a,m)
        assert_allclose(x,y[::-1],rtol=1e-10,atol=1e-48)
    def test_free_staggered_zero(self):
        v=phase_and_bound(directed_actions(Pulse(),Pulse(delay=.5),Mount(187)))
        self.assertLess(abs(v['phase']),1e-11)
        self.assertLess(abs(v['weyl']),1e-11)
    def test_damped_staggered_not_zero(self):
        v=phase_and_bound(directed_actions(Pulse(),Pulse(delay=.5),Mount(187,.42,.58)))
        self.assertGreater(abs(v['phase']),1e-4)
        self.assertGreater(abs(v['weyl']),1e-5)
    def test_causality(self):
        a=Pulse(ramp=.2,hold=0);b=Pulse(ramp=.2,hold=0,delay=1)
        ab,ba=directed_actions(a,b,Mount(1,1,.1))
        self.assertLess(abs(ab),1e-48)
        self.assertGreater(abs(ba),1e-42)
    def test_hinf_formula(self):
        for z in [.1,.58,1.,2.]:
            m=Mount(187,.42,z);w=np.linspace(0,10*m.omega,100001)
            v=np.max(np.abs(m.susceptibility(w)))
            assert_allclose(v,m.hinf(),rtol=3e-7)
    def test_hinf_action_bound(self):
        p=Pulse()
        for f in [.2,.42,2,10]:
            m=Mount(187,f,.58);a=directed_actions(p,p,m)
            self.assertLessEqual(max(map(abs,a)),m.hinf()*p.norms()['l2sq']*(1+1e-10))
    def test_ode_convergence(self):
        a=Pulse();b=Pulse(delay=.5);m=Mount(187,.42,.58)
        x=directed_actions(a,b,m);y=directed_actions(a,b,m,rtol=2e-13,atol=3e-15)
        assert_allclose(x,y,rtol=3e-8,atol=1e-46)
    def test_nulling(self):
        a=Pulse();b=Pulse(delay=.5);m=Mount(187,.42,.58)
        gs=[Pulse(ramp=.25,hold=0),Pulse(ramp=.5,hold=0,delay=3)]
        v=np.array(directed_actions(a,b,m))/HBAR
        mat=np.column_stack([np.array(directed_actions(a,g,m))/HBAR for g in gs])
        c=np.linalg.solve(mat,-v)
        w=Wave(((1.,b),(c[0],gs[0]),(c[1],gs[1])))
        out=phase_and_bound(directed_actions(a,w,m,rtol=2e-13,atol=3e-15))
        self.assertLess(out['beta'],1e-11)
        self.assertLess(np.linalg.cond(mat),1.4)
        for t in np.linspace(0,4,1001):
            self.assertGreaterEqual(w.d(t),-1e-15)
            self.assertLessEqual(w.d(t),a.split+1e-15)

class GravityTests(unittest.TestCase):
    def test_bose(self):
        p=Pulse();assert_allclose(pair_phase(p,p,450e-6),-.361183810595732,rtol=1e-12)
    def test_no_one_sided_phase(self):
        a=Pulse();b=Pulse(split=0)
        self.assertEqual(pair_phase(a,b,450e-6),0.)
    def test_parallel_sign(self):
        p=Pulse(split=29e-6,ramp=.25,hold=.5)
        self.assertGreater(pair_phase(p,p,83e-6,'parallel'),0)
    def test_screened_gap(self):
        gap,z=screened_gap()
        assert_allclose(gap(0),83e-6,rtol=1e-12)
        self.assertGreater(z,16e-6);self.assertLess(z,17e-6)
    def test_williams_hold_arithmetic(self):
        m,d,r,t=1e-14,1e-6,100e-6,.14
        delta=-2*G*m*m*t*d*d/(HBAR*r*(r*r-d*d))
        assert_allclose(delta,-1.7722746500675186e-5,rtol=1e-8)
    def test_stable_kernel(self):
        a=Pulse(split=1e-9,ramp=.1,hold=.1)
        exact=pair_phase(a,a,100e-6)
        integ,_=integrate_knots(lambda t:(a.d(t)/a.split)**2,a.knots)
        leading=-2*G*a.mass**2*a.split**2/(HBAR*(100e-6)**3)*integ
        assert_allclose(exact,leading,rtol=2e-10)

class ChannelTests(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(71)
    def physical(self):
        alpha=(self.rng.normal(size=(3,2))+1j*self.rng.normal(size=(3,2)))*.25
        V=np.real(alpha.conj().T@alpha)
        sigma=float(np.imag(np.sum(alpha[:,0]*alpha[:,1].conj())))
        delta=float(self.rng.normal())*.4
        return delta,sigma,V
    def test_channel_cp(self):
        for _ in range(40):
            d,s,v=self.physical();e=np.linalg.eigvalsh(multiplier(d,s,v))
            self.assertGreater(e.min(),-1e-12)
    def test_interpolation_cp(self):
        for _ in range(20):
            d,s,v=self.physical()
            for u in [.0,.2,.5,.8,1.]:
                self.assertGreater(np.linalg.eigvalsh(multiplier(u*d,u*s,v)).min(),-1e-12)
    def test_response_bound(self):
        for _ in range(50):
            d,s,v=self.physical();a=self.rng.normal(size=(4,4))+1j*self.rng.normal(size=(4,4))
            rho=a@a.conj().T;rho/=np.trace(rho)
            r=multiplier(d,s,v)*rho;r0=multiplier(0,0,v)*rho
            self.assertLessEqual(trace_distance(r,r0),min(1.,abs(d)/4+abs(s))+1e-12)
    def test_reference_system_bound(self):
        for _ in range(20):
            d,s,v=self.physical();a=self.rng.normal(size=(8,8))+1j*self.rng.normal(size=(8,8))
            rho=a@a.conj().T;rho/=np.trace(rho)
            mul=np.repeat(np.repeat(multiplier(d,s,v),2,axis=0),2,axis=1)
            mul0=np.repeat(np.repeat(multiplier(0,0,v),2,axis=0),2,axis=1)
            self.assertLessEqual(trace_distance(mul*rho,mul0*rho),min(1.,abs(d)/4+abs(s))+1e-12)
    def test_dephasing_separable(self):
        for _ in range(40):
            _,_,v=self.physical();vectors=[]
            for j in range(2):
                q=self.rng.normal(size=2)+1j*self.rng.normal(size=2);vectors.append(q/np.linalg.norm(q))
            a=np.kron(*vectors);rho=np.outer(a,a.conj())
            self.assertLess(negativity(multiplier(0,0,v)*rho),1e-12)
    def test_sharp_small_phase(self):
        rho=np.ones((4,4))/4;d=1e-4
        r=multiplier(d,0,np.zeros((2,2)))*rho
        assert_allclose(trace_distance(r,rho),math.sin(d/4),rtol=1e-8)
    def test_bound_identity(self):
        for _ in range(100):
            a,b=self.rng.normal(size=2)
            assert_allclose(abs(a+b)/2+abs(a-b)/2,max(abs(a),abs(b)),rtol=1e-13)
    def test_witness_spectrum(self):
        assert_allclose(np.linalg.eigvalsh(witness()),[-.5,.5,.5,.5],atol=1e-12)
    def test_witness_target(self):
        for d in [-2,-.3,.3,2]:
            r=multiplier(d,0,np.zeros((2,2)))/4
            val=-np.trace(witness(1 if d>0 else -1)@r).real
            assert_allclose(val,abs(math.sin(d/2))/2,rtol=1e-12)
    def test_weyl_matrix_comparison(self):
        # Independent finite oscillator implementation, hbar=M=omega=1.
        n=45;a=np.diag(np.sqrt(np.arange(1,n)),1);ad=a.T
        alphas=[.21+.13j,-.17+.11j]
        V=np.real(np.outer(np.conj(alphas),alphas))
        sigma=np.imag(alphas[0]*np.conj(alphas[1]));delta=.38
        states=[]
        for za,zb in SIGNS:
            alpha=za*alphas[0]+zb*alphas[1]
            state=expm(alpha*ad-np.conj(alpha)*a)[:,0]*np.exp(1j*delta*za*zb/4)
            states.append(state)
        gram=np.array([[np.vdot(y,x) for y in states] for x in states])
        assert_allclose(gram,multiplier(delta,sigma,V),atol=3e-14)
    def test_calibration_error(self):
        val=calibration_bound(-.1*HBAR,.2*HBAR,.01*HBAR,.02*HBAR)
        assert_allclose(val,.22)
    def test_two_directions_required(self):
        # Zero unitary invariant need not erase the Weyl part.
        d=0.;s=.05;v=np.eye(2)*.1
        self.assertGreater(np.max(abs(multiplier(d,s,v)-multiplier(0,0,v))),.01)

if __name__=='__main__':unittest.main(verbosity=2)
