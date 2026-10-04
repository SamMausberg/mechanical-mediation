"""Outward interval arithmetic for the finite-gap design inequalities.

This checks arithmetic in a stipulated material model, not sample uncertainty.
"""
from pathlib import Path
import json
from mpmath import iv
iv.dps=65
r=iv.mpf
hbar=r('6.62607015e-34')/(2*iv.pi)
G=r('6.67430e-11');T=r(293)
E=(r('210.0593')+r('.1534883')*T-r('.001617390')*T**2+r('5.117060e-6')*T**3-r('6.154600e-9')*T**4)*10**9
ell=r('.1');A=r('.0001');b=r('.001');density=r(8000)
C=ell/(9*E*A);S=(2*ell/9-b/6)/(E*A);w2=iv.pi**2*E/(density*ell**2)
m=r('1e-14');d=r('1e-6');R=r('.0001');H=r('.14')
rows={}
for ts in ['.001','.01']:
 t=r(ts);Hef=H+r(1042)/1287*t
 dm=560*C*m*m*d*d/(11*hbar*t**3)
 err=4*S*m*m*d*d/(hbar*w2*t**5)*(560+r(91728)*iv.sqrt(5)/25)
 gmin=2*G*m*m*d*d*Hef/(hbar*R**3);gmax=gmin/(1-(d/R)**2)
 lower=(dm-err)/gmax;upper=(dm+err)/gmin
 rows[ts]={'phase_static':str(dm),'phase_error':str(err),'gravity_low':str(gmin),'gravity_high':str(gmax),
           'ratio_lower_bound':str(lower),'ratio_upper_bound':str(upper)}
 if ts=='.001':
  assert bool(lower>r('1.42')) and bool(upper<r('1.60'))
  assert bool(dm-err>r('2.538e-5')) and bool(dm+err<r('2.838e-5'))
 else:
  assert bool(upper<r('.00144'))
  assert bool((dm+err)/4 <r('.01')*iv.sin(gmin/2)/2)
root=Path(__file__).resolve().parents[1]
(root/'results/design_intervals.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2)); print('PASS: all outward-rounded manuscript design inequalities')
