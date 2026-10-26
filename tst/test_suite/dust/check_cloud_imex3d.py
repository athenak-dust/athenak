#!/usr/bin/env python3
"""Short no-gravity checks of 3D refinable-cloud IMEX2+ using built-in pgens.

Run with an existing MPI/Serial double-precision executable and an empty scratch
directory outside the checkout. Includes unequal species masses, stiff drag,
time refinement, actual coarse/fine leaves and conservative momentum exchange.
The broader shearing AMR/G/R, restart and gravity tests remain separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def run(args, name, input_name, overrides):
    folder = args.scratch/name
    folder.mkdir()
    inp = folder/'input.athinput'
    # AthenaK CLI overrides require existing keys. Build an explicit reviewable
    # fixture so opt-in keys and refined-region blocks need not exist upstream.
    data, block = {}, None
    for line in (ROOT/'inputs/dust'/input_name).read_text().splitlines():
        line = line.split('#')[0].strip()
        if line.startswith('<'):
            block = line.strip('<>')
            data[block] = {}
        elif '=' in line and block:
            key, value = line.split('=', 1)
            data[block][key.strip()] = value.strip()
    options = ['job/basename=test',
               'mesh/nghost=4', 'mesh/nx1=16', 'mesh/nx2=8', 'mesh/nx3=8',
               'meshblock/nx1=8', 'meshblock/nx2=8', 'meshblock/nx3=8',
               'hydro/reconstruct=wenoz', 'time/tlim=0.1', 'time/cfl_number=0.2',
               'time/ndiag=100', 'dust/coupling=imex', 'dust/experimental_3d=true',
               'dust/experimental_static_smr=true', 'dust/experimental_mpi=true',
               'dust/experimental_smr_transfer=refinable_cloud',
               'dust/cloud_gather_order=2', 'dust/cloud_transition=2',
               'dust/cloud_split_device=true', 'dust/cloud_p1_split=true',
               'dust/cloud_split_overlap=true', 'dust/cloud_merge=false',
               'dust/cloud_resample=false', *overrides]
    for option in options:
        path, value = option.split('=', 1)
        group, key = path.split('/', 1)
        data.setdefault(group, {})[key] = value
    inp.write_text(''.join(f'<{group}>\n'+''.join(f'{key} = {value}\n' for key, value
                       in values.items())+'\n' for group, values in data.items()))
    command = [str(args.binary), '-i', str(inp)]
    if args.ranks > 1:
        command = shlex.split(args.mpiexec)+['-n', str(args.ranks)]+command
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (folder/'run.log').open('w') as stream:
        subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT,
                       timeout=args.timeout, check=True)
    return folder


def damping_result(folder):
    lines = (folder/'test-errs.dat').read_text().splitlines()
    values = [float(x) for x in lines[-1].split()]
    # Nx1 Nx2 Nx3 Ncycle, gas error, three species errors, momentum error, positions.
    # The pgen intentionally writes NaN displacement columns on MPI: its initial
    # positions can only be reconstructed from the serial sequential PTAG scheme.
    assert all(math.isfinite(x) for x in values[:9]), values
    assert values[8] < 1e-12, ('total momentum error', values[8])
    return values[:9]  # only the finite velocity and momentum columns verified here


def history(path):
    labels, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            labels.update({label: int(n)-1 for n, label in
                           re.findall(r'\[(\d+)\]=(\S+)', line)})
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    return labels, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=(1, 2), default=1)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    args.binary, args.scratch = args.binary.resolve(), args.scratch.resolve()
    if not args.binary.is_file():
        parser.error('binary must exist')
    if args.scratch == ROOT or ROOT in args.scratch.parents:
        parser.error('scratch must be outside the source checkout')
    if args.scratch.exists() and any(args.scratch.iterdir()):
        parser.error('scratch must be new or empty; preserve existing evidence')
    args.scratch.mkdir(parents=True, exist_ok=True)
    manifest = dict(binary=str(args.binary), ranks=args.ranks,
                    sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    (args.scratch/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    mixture = ['particles/ppc=3', 'problem/eps_1=0.2', 'problem/eps_2=0.5',
               'problem/eps_3=0.3', 'dust/taus_1=0.1', 'dust/taus_2=0.03', 'dust/taus_3=1']
    result = {}
    for stiff in (False, True):
        opts = mixture+(['dust/taus_1=0.0001'] if stiff else [])
        key = 'stiff' if stiff else 'resolved'
        coarse = damping_result(run(args, key, 'dust_damping.athinput', opts))
        fine = damping_result(run(args, key+'_half', 'dust_damping.athinput',
                                  opts+['time/cfl_number=0.1']))
        # These CFLs are not the asymptotic time-order limit. Require a substantial
        # decrease against the independent damping ODE, without declaring order two.
        ratio = max(coarse[4:8])/max(fine[4:8])
        assert ratio > 2.5, (key, ratio)
        result[key] = dict(coarse=coarse, half=fine, max_error_ratio=ratio)
    # Root parents keep the representation consistent when only part of the grid refines.
    # dust_nsh's dust_test=wave provides the independent damped-wave eigenmode history.
    root = ['problem/cloud_root_random=true', 'problem/root_placement=lattice',
            'problem/dust_test=wave', 'problem/user_hist=true', 'problem/eps_1=1',
            'dust/nspecies=1', 'particles/ppc=1', 'dust/taus_1=0.1',
            'mesh/x2min=-0.25', 'mesh/x2max=0.25',
            'mesh/x3min=-0.25', 'mesh/x3max=0.25',
            'output1/dt=0.1', 'output1/data_format=%.16e']
    # Use a dedicated no-rotation fixture: a block named shearing_box would activate it
    # even if omega=0 and is intentionally absent here.
    fixture = ROOT/'inputs/dust/cloud_imex3d_wave.athinput'
    assert fixture.is_file()
    for mixed in (False, True):
        opts = root+(['mesh_refinement/refinement=static',
                      'mesh_refinement/num_levels=2', 'refined_region1/level=1',
                      'refined_region1/x1min=0', 'refined_region1/x1max=0.5',
                      'refined_region1/x2min=-0.25', 'refined_region1/x2max=0.25',
                      'refined_region1/x3min=-0.25', 'refined_region1/x3max=0.25']
                     if mixed else [])
        folder = run(args, 'wave_smr' if mixed else 'wave_uniform', fixture.name, opts)
        labels, rows = history(folder/'test.user.hst')
        last = rows[-1]
        errors = {name: last[labels[name]] for name in ('e_rhog', 'e_ux', 'e_rhod', 'dmass')}
        assert all(math.isfinite(x) for x in errors.values()), errors
        assert abs(errors['dmass']) < 1e-11, errors
        assert max(errors[name] for name in ('e_rhog', 'e_ux', 'e_rhod')) < .08, errors
        result[folder.name] = errors
    (args.scratch/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
