import MDAnalysis as mda
from pathlib import Path
import numpy as np
from tqdm import tqdm
import math

class PositionsReader:
    read_map = {
        'mono_com': 'get_mono_com',
        'sol_com': 'get_solvation_com',
        'chain_com': 'get_chain_com',
        'poly_O': 'get_polymer_oxygen',
        'poly_N': 'get_polymer_nitrogen',
        'sol_O': 'get_solvation_oxygen',
        'sol_H': 'get_solvation_hydrogen',
    }

    @classmethod
    def get_positions(cls, position_type: str, u: mda.Universe, read_interval=None, start=None, end=None):
        method_name = cls.read_map[position_type]
        read_interval, start, end = cls.time_input_to_frame_index(u, read_interval, start, end)
        return getattr(cls, method_name)(u, read_interval=read_interval, start=start, end=end)

    @staticmethod
    def time_input_to_frame_index(u: mda.Universe, read_interval=None, start=None, end=None):
        if read_interval is None:
            read_interval = 1
        else:
            read_interval = math.ceil(read_interval / u.trajectory.dt)

        if start is None:
            start = 0
        else:
            start = math.ceil(start / u.trajectory.dt)

        if end is None:
            end = int(1e10)
        else:
            end = math.ceil(end / u.trajectory.dt) + 1
        return read_interval, start, end

    @staticmethod
    def get_mono_com(u: mda.Universe, cache_file='mono_indices.npz', read_interval=None, start=None, end=None):

        cache_file = Path(cache_file)

        if cache_file.is_file():
            with np.load(cache_file, allow_pickle=False) as data:
                mono_list = [
                    u.atoms[data[f"arr_{i}"]]
                    for i in range(len(data.files))
                ]
        else:
            mono = u.select_atoms("resname M")
            heavy_mono_res = mono.select_atoms("not name H*").residues

            mono_list = []
            for r in heavy_mono_res:
                heavy = r.atoms.select_atoms("not name H*")
                unit = mono.select_atoms(
                    "group heavy or (name H* and bonded group heavy)",
                    heavy=heavy,
                )
                mono_list.append(unit)

            np.savez(cache_file, *[g.indices for g in mono_list])

        print(f"monomers : {len(mono_list)}")
        total_f = len(u.trajectory[start:end:read_interval])
        position = np.empty((total_f, len(mono_list), 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            for j, m in enumerate(mono_list):
                position[frame_idx, j, :] = m.center_of_mass()

        return position

    @staticmethod
    def get_chain_com(u: mda.Universe, read_interval=None, start=None, end=None):
        raise NotImplementedError

    @staticmethod
    def get_solvation_com(u: mda.Universe, read_interval=None, start=None, end=None):
        solvation = u.select_atoms('resname SOL')
        total_f = len(u.trajectory[start:end:read_interval])
        position = np.empty((total_f, len(solvation.residues), 3))
        for i, _ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            position[frame_idx, :, :] = solvation.center_of_mass(compound='residues')
        return position

    @staticmethod
    def get_solvation_oxygen(u: mda.Universe, read_interval=None, start=None, end=None):
        oxygen = u.select_atoms('resname SOL and name OW')
        total_f = len(u.trajectory[start:end:read_interval])
        position = np.empty((total_f, len(oxygen), 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            position[frame_idx, :, :] = oxygen.positions
        return position

    @staticmethod
    def get_solvation_hydrogen(u: mda.Universe, read_interval=None, start=None, end=None):
        hydro = u.select_atoms('resname SOL and name H*')
        total_f = len(u.trajectory[start:end:read_interval])
        position = np.empty((total_f, len(hydro), 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            position[frame_idx, :, :] = hydro.positions
        return position

    @staticmethod
    def get_polymer_oxygen(u: mda.Universe, read_interval=None, start=None, end=None):
        poly_oxy = u.select_atoms('resname M and name O')
        total_f = len(u.trajectory[start:end:read_interval])
        position = np.empty((total_f, len(poly_oxy), 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            position[frame_idx, :, :] = poly_oxy.positions
        return position

    @staticmethod
    def get_polymer_nitrogen(u: mda.Universe, read_interval=None, start=None, end=None):
        poly_nitro = u.select_atoms('resname M and name N')
        total_f = len(u.trajectory[start:end:read_interval])
        position = np.empty((total_f, len(poly_nitro), 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            position[frame_idx, :, :] = poly_nitro.positions
        return position

    @staticmethod
    def get_box(u: mda.Universe, read_interval=None, start=None, end=None):
        # 正交盒子
        total_f = len(u.trajectory[start:end:read_interval])
        box = np.empty((total_f, 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            frame_idx = i
            box[frame_idx, :] = u.dimensions[0:3]
        return box
