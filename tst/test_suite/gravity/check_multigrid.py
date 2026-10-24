#!/usr/bin/env python3
"""Portable, short application checks for composite self-gravity.

Example (use a project-designated scratch path OUTSIDE the source checkout):
  python check_multigrid.py --binary /scratch/build/src/athena \
      --scratch /scratch/gravity-check --ranks 2

Requires an existing double-precision built_in_pgens executable. Does not build,
install packages, write in the source tree, or replace an existing result folder.
These smoke/convergence checks supplement the broader MPI/GPU/restart experiments;
they do not establish nonlinear production accuracy or performance scaling.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]


def reference(time_end):
    """Independent high-accuracy RK4 integration of the linear shearing-wave ODE."""
    def derivative(t, y):
        kx, ky = 2*math.pi*(1+1.5*t), 2*math.pi
        c = 1-50/(kx*kx+ky*ky)
        d, u, v, w = y
        return (-1j*(kx*u+ky*v), 2*v-1j*kx*c*d, -.5*u-1j*ky*c*d, 0j)
    y = [1e-5+0j, 0j, 0j, 0j]
    steps = max(1, math.ceil(time_end/1e-5))
    dt = time_end/steps
    for step in range(steps):
        t = step*dt
        a = derivative(t, y)
        b = derivative(t+dt/2, [v+dt*k/2 for v,k in zip(y,a)])
        c = derivative(t+dt/2, [v+dt*k/2 for v,k in zip(y,b)])
        d = derivative(t+dt, [v+dt*k for v,k in zip(y,c)])
        y = [v+dt*(aa+2*bb+2*cc+dd)/6 for v,aa,bb,cc,dd in zip(y,a,b,c,d)]
    return y


def run_case(args, name, filename, overrides=()):
    out = args.scratch/name
    out.mkdir()
    inp = out/'input.athinput'
    inp.write_bytes((ROOT/'inputs/hydro'/filename).read_bytes())
    command = [str(args.binary), '-i', str(inp), *overrides]
    if args.ranks > 1:
        command = shlex.split(args.mpiexec)+['-n', str(args.ranks)]+command
    (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    started = time.monotonic()
    with (out/'run.log').open('w') as stream:
        subprocess.run(command, cwd=out, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=args.timeout)
    lines = (out/'run.log').read_text().splitlines()
    rows = [json.loads(line[len('# SELF_GRAVITY_WAVE '):]) for line in lines
            if line.startswith('# SELF_GRAVITY_WAVE ')]
    assert len(rows) == 1, f'{name}: expected one rank-zero diagnostic, got {len(rows)}'
    result = rows[-1]
    result['wall_seconds'] = time.monotonic()-started
    assert math.isfinite(result['residual']) and result['residual'] < 2e-9, result
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def history(path):
    labels, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            # Native history labels are one-based and truncated to ten characters.
            labels.update({name:int(i)-1 for i,name in re.findall(r'\[(\d+)\]=(\S+)',line)})
        elif line.strip():
            rows.append([float(v) for v in line.split()])
    return labels, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=(1,2,4), default=1)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    args.binary, args.scratch = args.binary.resolve(), args.scratch.resolve()
    if not args.binary.is_file() or args.ranks < 1:
        parser.error('an existing executable and positive rank count are required')
    if args.scratch == ROOT or ROOT in args.scratch.parents:
        parser.error('scratch must be outside the source checkout')
    if args.scratch.exists() and any(args.scratch.iterdir()):
        parser.error('scratch must be new or empty; existing results are preserved')
    args.scratch.mkdir(parents=True, exist_ok=True)
    manifest = {'binary':str(args.binary), 'sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                'ranks':args.ranks, 'test_root':str(ROOT)}
    (args.scratch/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = {}
    for n in (16,32):
        results[f'periodic{n}'] = run_case(args, f'periodic{n}',
            'self_gravity_poisson.athinput', (f'mesh/nx1={n}',f'mesh/nx2={n}'))
    for n in (8,16):
        results[f'slab{n}'] = run_case(args, f'slab{n}', 'self_gravity_slab.athinput',
            (f'mesh/nx1={n}',f'mesh/nx2={n}',f'mesh/nx3={n}'))
    for family, coarse, fine, minimum in (('periodic',16,32,1.7),('slab',8,16,1.5)):
        a,b = results[f'{family}{coarse}'],results[f'{family}{fine}']
        for key in ('phi_l2','acceleration_l2'):
            order = math.log2(a[key]/b[key])
            assert order > minimum, (family,key,order)
        if family == 'slab':
            assert b['acceleration_relative_l2'] < .02  # retain mean-sheet gravity
    result = run_case(args, 'shear_amr', 'self_gravity_shear_amr.athinput')
    ref = reference(result['time'])
    measured = [complex(result['density_mode_cos'], -result['density_mode_sin'])]
    for d in (1,2,3):
        prefix = f'momentum{d}_over_rho0_mode_'
        measured.append(complex(result[prefix+'cos'], -result[prefix+'sin']))
    error = math.sqrt(sum(abs(x-y)**2 for x,y in zip(measured,ref)))/1e-5
    assert error < .02, ('shearing AMR ODE error',error)
    labels, rows = history(next((args.scratch/'shear_amr').glob('*.user.hst')))
    mass = [row[labels['mass']] for row in rows]
    blocks = [round(row[labels['blocks']]) for row in rows]
    mixed_name = 'oa_mixed_blocks' if 'oa_mixed_blocks' in labels else 'oa_mixed_b'
    mixed = [round(row[labels[mixed_name]]) for row in rows]
    assert max(abs(v/mass[0]-1) for v in mass) < 1e-12
    assert max(blocks) > blocks[0] and blocks[-1] == blocks[0]
    assert max(mixed) > 0, 'mixed orbital-advection faces were not exercised'
    results['shear_amr'] = {**result, 'ode_error_over_amp':error, 'max_mixed_blocks':max(mixed)}
    (args.scratch/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print('PASS: periodic/mixed fields, isolated mean slab, physical shearing AMR')


if __name__ == '__main__':
    main()
