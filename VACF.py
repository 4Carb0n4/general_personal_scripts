import argparse
import numpy as np
import MDAnalysis as mda
import numba as nb
from pathlib import Path
from datetime import datetime
import sys

from tqdm import tqdm

from PositionsReader import PositionsReader


def get_position_data(universe, ref, sel, end, cache):
    print("read started")
    PositionsReader.CACHE = cache
    pos_a = PositionsReader.get_positions(ref, universe, end=end)
    pos_b = PositionsReader.get_positions(sel, universe, end=end)
    print("read done")

    return pos_a, pos_b


# -------------------------

class VelocityReader:
    read_map = {'sol_com': 'get_sol_com_velocity',
                'sol_O': 'get_OW_velocity'}

    @classmethod
    def get_velocities(cls, velocity_type: str, u: mda.Universe, read_interval=None, start=None, end=None):
        method_name = cls.read_map[velocity_type]
        read_interval, start, end = cls.time_input_to_frame_index(u, read_interval, start, end)
        return getattr(cls, method_name)(u, read_interval, start, end)

    @staticmethod
    def time_input_to_frame_index(u: mda.Universe, read_interval=None, start=None, end=None):
        if read_interval is None:
            read_interval = 1
        else:
            read_interval = np.ceil(read_interval / u.trajectory.dt).astype(int)

        if start is None:
            start = 0
        else:
            start = np.ceil(start / u.trajectory.dt).astype(int)

        if end is None:
            end = int(1e10)
        else:
            end = np.ceil(end / u.trajectory.dt).astype(int) + 1
        return read_interval, start, end

    @staticmethod
    def get_OW_velocity(u: mda.Universe, read_interval=None, start=None, end=None):
        OW = u.select_atoms('resname SOL and name OW')
        total_f = len(u.trajectory[start:end:read_interval])
        velo = np.empty((total_f, len(OW), 3))
        for i, ts in enumerate(tqdm(u.trajectory[start:end:read_interval])):
            velo[i, :, :] = OW.velocities
        return velo

    @staticmethod
    def get_sol_com_velocity(u: mda.Universe):
        water = u.select_atoms('resname SOL')
        velo = np.empty((u.trajectory.n_frames, len(water.residues), 3))
        mass_of_water = water.residues[0].mass
        for i, ts in enumerate(tqdm(u.trajectory)):
            velo_dot_mass = water.velocities * water.masses[:, None]
            for j in range(0, velo.shape[1]):
                velo[i, j, :] = (velo_dot_mass[j, 3] + velo_dot_mass[j + 1, 3] + velo_dot_mass[j + 2, 3]
                                 ) / mass_of_water
        return velo


@nb.jit
def sort_particles_by_dist_nb(atom_x: np.ndarray, atom_y: np.ndarray,
                              r_divide: tuple, frame: int):
    """
    sort water into different layers
    :param atom_x: [n_frames, n_atoms, 3]
    :param atom_y: [n_frames, n_atoms, 3]
    :param r_divide: str like "1.0,2.0,3.0"
    :param frame: int
    :return: [n_atoms,1]
    """
    n_particles = atom_y.shape[1]
    n_ref = atom_x.shape[1]

    n_layers = len(r_divide)
    layer = np.zeros((n_particles, 1))

    for sel_id in range(n_particles):
        min_dist = 1e10
        for ref_id in range(n_ref):
            dx = atom_x[frame, ref_id, 0] - atom_y[frame, sel_id, 0]
            dy = atom_x[frame, ref_id, 1] - atom_y[frame, sel_id, 1]
            dz = atom_x[frame, ref_id, 2] - atom_y[frame, sel_id, 2]

            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < min_dist * min_dist:
                min_dist = dist2 ** 0.5

        # 分类
        for layer_id, r in enumerate(r_divide):
            if min_dist < r:
                layer[sel_id, 0] = layer_id
                break
        else:
            layer[sel_id, 0] = n_layers
    return layer


def sort_particles_by_dist(atom_x: np.ndarray, atom_y: np.ndarray,
                           r_divide: tuple, frame: int):
    """
    sort water into different layers
    :param atom_x: [n_frames, n_atoms, 3]
    :param atom_y: [n_frames, n_atoms, 3]
    :param r_divide: str like "1.0,2.0,3.0"
    :param frame: int
    :return: list[list]
    """
    n_particles = atom_y.shape[1]
    n_ref = atom_x.shape[1]

    layers = [[] for i in range(len(r_divide) + 1)]

    for i in range(n_particles):
        min_dist = 1e10
        for j in range(n_ref):
            dx = atom_x[frame, j, 0] - atom_y[frame, i, 0]
            dy = atom_x[frame, j, 1] - atom_y[frame, i, 1]
            dz = atom_x[frame, j, 2] - atom_y[frame, i, 2]

            dist2 = dx * dx + dy * dy + dz * dz
            # dist2 = np.linalg.norm(atom_x[frame, j, :] - atom_y[frame, i, :])
            if dist2 < min_dist * min_dist:
                min_dist = dist2 ** 0.5

        # 分类
        for idx, r in enumerate(r_divide):
            if min_dist < r:
                layers[idx].append(i)
                break
        else:
            layers[-1].append(i)
    print('sort done')
    return layers  # 返回 list，每个元素是该层的索引


def ACF_self(velo: np.ndarray):
    """
    速度自相关
    :param velo: [n_frames, n_atoms, 3]
    :return: [n_frames,]
    """
    n_frames = velo.shape[0]
    fqd_c = np.fft.fft(velo, n=2 * n_frames, axis=0)
    fqd_m = np.fft.ifft(fqd_c * fqd_c.conj(), n=2 * n_frames, axis=0)
    AA_dot = fqd_m[:n_frames, :, :].real.sum(axis=-1)
    mean_atoms_frames = AA_dot.mean(axis=-1) / np.arange(n_frames, 0, -1).T

    return mean_atoms_frames


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument_group("输入输出")
    parser.add_argument('-p', '--topol', help='拓扑,.tpr/.top etc', required=True)
    parser.add_argument('-f', '--trajectory', help='轨迹,.xtc/.trr etc', required=True)
    parser.add_argument('-o', '--output', help='输出文件', default=None)
    parser.add_argument_group("计算：半径和组")
    parser.add_argument('-r', '--radius', help='', type=str, required=True)
    parser.add_argument('-ref', help='参考组',
                        choices=PositionsReader.read_map.keys(), required=True)
    parser.add_argument('-sel', help="对比组",
                        choices=PositionsReader.read_map.keys(), required=True)
    parser.add_argument('-cal', help="计算组",
                        choices=VelocityReader.read_map.keys(), required=True)
    parser.add_argument('--dump', action='store_true', default=False)
    parser.add_argument('--stop', action='store_true', default=False)
    parser.add_argument('--continue_run', action='store_true', help='从dump处恢复', default=False)
    parser.add_argument('-e', '--end', type=float, default=None)
    parser.add_argument('--cache', default='mono_indices.npz')

    args = parser.parse_args()
    return args


# ========================================
def main():
    args = get_args()

    top1 = Path(args.topol)
    traj1 = Path(args.trajectory)
    u = mda.Universe(str(top1), str(traj1))

    atom_x, atom_y = get_position_data(u, args.ref, args.sel, args.end, args.cache)
    total_frames = atom_x.shape[0]

    r_divide = tuple([float(i) for i in args.radius.split(',')])

    sort_result = sort_particles_by_dist(
        atom_x, atom_y, r_divide, 0
    )

    print(f"water total :{len(u.select_atoms("resname SOL").residues)}")
    water_number = []

    if not args.continue_run:
        atom_y_velo = VelocityReader.get_velocities(args.cal, u, end=args.end)
        print('read velo done')

    else:
        atom_y_velo = np.load('velocity_y.npy')
        print('load velocity_y.npy success')

    if args.dump:
        np.save('velocity_y.npy', atom_y_velo)

    if args.stop:
        sys.exit('停止输出文件')

    output_dir = args.output or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    data = np.arange(0, total_frames) * u.trajectory.dt

    header_comments = [
        f"# ref group {args.ref}",
        f"# sel group {args.sel}",
        f"# cal group {args.cal}",
        f"# layer devision {args.radius}"
    ]
    header_labels = ['@ xlabel time(ps)',
                     '@ ylabel VACF']
    for i, grp in enumerate(sort_result):
        print(f"Group {i} :{len(grp)}")
        water_number.append(len(grp))
        print(sum(water_number))

    for grp_id, atoms_indies in enumerate(sort_result):
        vacf = ACF_self(atom_y_velo[:, atoms_indies, :])
        data = np.column_stack((data, vacf))
        header_comments.append(f"# layer 0 {len(atoms_indies)}")
        header_labels.append(f'@ s{grp_id} legend  "water layer_{grp_id}"')

    header = "\n".join(header_comments + header_labels)
    np.savetxt(f"{output_dir.parent}/{output_dir.stem}.xvg",
               data,
               header=header,
               comments='')


# ========================================
if __name__ == '__main__':
    main()
