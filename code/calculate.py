"""Generate the merged paper's numerical results and native-pgfplots data."""
from pathlib import Path
from dataclasses import asdict
import json, math
import numpy as np
from model import *
from design import FixedBar,design_row,pulse_extra,gravity_quadratic_interval
from elastic import Protocol,Rod,mode_action,force_transform,KB
from scipy.special import zeta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';DATA=ROOT/'data'
OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)

def main():
    bar=FixedBar()
    cases={'Bose':(Pulse(),450e-6,'collinear'),
           'Screened':(Pulse(split=29e-6,ramp=.25,hold=.5),83e-6,'parallel'),
           'Short_1ms':(Pulse(split=1e-6,ramp=.001,hold=.14),100e-6,'collinear'),
           'Short_10ms':(Pulse(split=1e-6,ramp=.01,hold=.14),100e-6,'collinear')}
    gap,final_z=screened_gap()
    results={'constants':{'G':G,'hbar':HBAR},'screened_final_z_m':final_z,'cases':{}}
    for name,(p,r,geo) in cases.items():
        g=pair_phase(p,p,r,geo,gap if name=='Screened' else None)
        w=abs(math.sin(g/2))/2
        results['cases'][name]={'pulse':asdict(p),'midpoint_separation':r,'geometry':geo,
            'gravity_pair_phase':g,'ideal_witness':w,'norms':p.norms(),
            'one_percent_cross_compliance':.01*w*HBAR/p.norms()['l2sq']}
    p=Pulse();results['Bose_capsule']={'mass_design':400.,'phase':free_phase(p,400.),
        'beta':abs(free_phase(p,400.))/4,'phase_ratio':abs(free_phase(p,400)/results['cases']['Bose']['gravity_pair_phase'])}
    gain=1e11;error=1e-10;eb=p.norms()['l1']*error/(gain*HBAR)
    results['replay_example']={'gain_design':gain,'displacement_error_design':error,
        'action_error_over_hbar':eb,'capsule_beta_plus_error':results['Bose_capsule']['beta']+eb,
        'peak_force':gain*p.norms()['force_peak']}
    # A numerical controller example, not a measured horizontal device response.
    mt=Mount(187,.42,.58);a=Pulse();b=Pulse(delay=.5)
    gs=[Pulse(ramp=.25,hold=0),Pulse(ramp=.5,hold=0,delay=3)]
    base=np.array(directed_actions(a,b,mt))/HBAR
    mat=np.column_stack([np.array(directed_actions(a,g,mt))/HBAR for g in gs])
    co=np.linalg.solve(mat,-base)
    corr=Wave(((1.,b),(co[0],gs[0]),(co[1],gs[1])))
    results['null']={'model':asdict(mt),'model_status':'numerical test response; no horizontal-device identification',
        'base':base.tolist(),'matrix':mat.tolist(),'coefficients':co.tolist(),
        'synchronous':phase_and_bound(directed_actions(a,a,mt)),
        'gravity_corrected':pair_phase(a,corr,450e-6),
        'gravity_fraction':abs(pair_phase(a,corr,450e-6)/results['cases']['Bose']['gravity_pair_phase']),
        'error_amplification':1+sum(abs(co))}
    np.savetxt(DATA/'null_pulses.csv',[[t,a.d(t)*1e6,b.d(t)*1e6,corr.d(t)*1e6]
        for t in np.linspace(0,4,801)],delimiter=',',header='t,A,Bstaggered,Bcorrected',comments='')
    results['bar']={**asdict(bar),'status':'specified fixed-end longitudinal model; all geometry and density are design choices',
        'speed':bar.speed,'omega1':bar.omega1,'cross_compliance':bar.cross,'self_compliance':bar.self_compliance}
    for label in ['Bose','Short_1ms','Short_10ms']:
        p,r,_=cases[label];row=design_row(p,r,bar)
        q=Protocol(mass=p.mass,separation=r,split=p.split,ramp=p.ramp,hold=p.hold)
        phase=0;noise=0;N=300
        for n in range(1,N+1):
            w=n*bar.omega1;c=bar.coupling(n)
            phase+=4*np.prod(c)*mode_action(q,w)/(bar.mass*HBAR)
            noise+=(c@c)*abs(force_transform(q,w))**2/(2*bar.mass*HBAR*w)/math.tanh(HBAR*w/(2*KB*293))
        e=pulse_extra(p);norms=p.norms()
        actiontail=2/(bar.mass*bar.omega1**2)*(norms['l2sq']+norms['l1']*e['d1l1'])*zeta(2,N+1)
        noiset=2*e['d2l1']**2/(bar.mass*HBAR)*(zeta(5,N+1)/bar.omega1**5+2*KB*293/HBAR*zeta(6,N+1)/bar.omega1**6)
        row.update(phase_300=phase,phase_tail_bound=4*actiontail/HBAR,thermal_trace_300_at_293K=noise,
                   thermal_trace_tail_bound=noiset,quadratic_gravity_interval=list(gravity_quadratic_interval(p,r)))
        results['bar'][label]=row
    rows=[]
    for t in np.geomspace(.0007,.03,91):
        r=design_row(Pulse(split=1e-6,ramp=t,hold=.14),100e-6,bar)
        err=r['action_error']/Pulse(split=1e-6,ramp=t,hold=.14).norms()['l2sq']
        rows.append([t*1000,r['one_percent_compliance'],bar.cross,max(bar.cross-err,1e-20),bar.cross+err,
                     r['ratio_static'],r['ratio_error']])
    np.savetxt(DATA/'fast_design.csv',rows,delimiter=',',header='tau_ms,target,cross,lower,upper,ratio,ratioerror',comments='')
    q=Protocol();masspi=abs(q.recoil_phase(1))/math.pi
    rod=Rod(mass=masspi,length=1.1,speed=5000,temperature=4,port_width=100e-6)
    bp,bs=rod.phase_bound(q),rod.noise_bound(q)
    results['free_elastic_example']={'mass':masspi,'length':1.1,'speed':5000,'temperature':4,
        'phase_bound':bp,'noise_trace_bound':bs,'negativity_lower':.5-bp/4-min(1,math.sqrt(bs),2*bs)}
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
