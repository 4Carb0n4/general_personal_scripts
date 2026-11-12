import sys
from datetime import datetime
from pathlib import Path
import MDAnalysis as mda
import math
import numpy as np
from numba import cuda

from PositionsReader import PositionsReader
from VACF import get_args


def get_position_data(universe, ref, sel, end):
    print("read started")
    pos_a = PositionsReader.get_positions(ref, universe, end=end)
    pos_b = PositionsReader.get_positions(sel, universe, end=end)
    print("read done")

    return pos_a, pos_b


# -------------------------
from VACF import VelocityReader


# noinspection PyArgumentList
@cuda.jit
def sort_particles_by_dist_cuda_kernal(atom_x, atom_y, r_divide, layer):
    sel_id = cuda.grid(1)

    if sel_id >= atom_y.shape[0]:
        return

    min_dist = 1e10

    for ref_id in range(atom_x.shape[0]):
        dx = atom_x[ref_id, 0] - atom_y[sel_id, 0]
        dy = atom_x[ref_id, 1] - atom_y[sel_id, 1]
        dz = atom_x[ref_id, 2] - atom_y[sel_id, 2]

        dist2 = dx * dx + dy * dy + dz * dz

        if dist2 < min_dist * min_dist:
            min_dist = math.sqrt(dist2)

    # 默认归入最外层
    n_layers = r_divide.shape[0]
    layer[sel_id] = n_layers

    for layer_id in range(n_layers):
        if min_dist < r_divide[layer_id]:
            layer[sel_id] = layer_id
            break


def sort_particles_by_dist_cuda(atom_x: np.ndarray, atom_y: np.ndarray,
                                r_divide: tuple, frame: int):
    """
    根据当前帧中到参考粒子的最近距离，将水粒子分层。

    :param atom_x: [n_frames, n_ref, 3]
    :param atom_y: [n_frames, n_particles, 3]
    :param r_divide: 距离阈值元组，例如 (1.0, 2.0, 3.0)
    :param frame: 帧索引
    :return: [n_particles]，NumPy float64 数组
    """
    n_particles = atom_y.shape[1]

    # 只传输当前帧，转换为连续数组
    x = np.ascontiguousarray(atom_x[frame])
    y = np.ascontiguousarray(atom_y[frame])
    radii = np.ascontiguousarray(
        np.asarray(r_divide, dtype=np.float64)
    )

    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_radii = cuda.to_device(radii)
    d_layer = cuda.device_array(
        n_particles, dtype=np.float64
    )

    threads_per_block = 128

    blocks_per_grid = math.ceil(n_particles / threads_per_block)

    sort_particles_by_dist_cuda_kernal[
        blocks_per_grid, threads_per_block
    ](d_x, d_y, d_radii, d_layer)

    return d_layer.copy_to_host()


from VACF import ACF_self

# ========================================

args = get_args()

top1 = Path(args.topol)
traj1 = Path(args.trajectory)
u = mda.Universe(str(top1), str(traj1))

atom_x, atom_y = get_position_data(u, args.ref, args.sel, end=args.end)
total_frames = atom_x.shape[0]

r_divide = tuple([float(i) for i in args.radius.split(',')])
total_layers = len(r_divide) + 1

if Path('sort_result.npy').is_file():
    sort_result = np.load('sort_result.npy')
else:
    sort_result = np.empty((total_frames, atom_y.shape[1]))
    for f in range(total_frames):
        sort_result[f, :] = sort_particles_by_dist_cuda(
            atom_x, atom_y, r_divide, f
        )

if args.dump:
    np.save('sort_result.npy', sort_result)

if args.stop:
    sys.exit('停止输出文件')

atom_y_velo = VelocityReader.get_velocities(args.cal, u, end=args.end)
print('read velo done')

output_dir = args.output or datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(output_dir)
output_dir.parent.mkdir(parents=True, exist_ok=True)

data = np.arange(0, total_frames) * u.trajectory.dt

header_commnents = [
    f"# stay atoms VACF",
    f"# ref group{args.ref}",
    f"# sel group{args.sel}",
    f"# cal group{args.cal}",
    f"# layer devition {args.radius}"]
header_labels = ["@ xlabel time(ps)",
                 "@ ylabel VACF"]

print(f"OW total :{atom_y_velo.shape[1]}")
water_number = []
for i in range(total_layers):
    layer_numbers=np.count_nonzero(sort_result[0] == i)
    print(f"Layer {i} :{layer_numbers}")
    water_number.append(layer_numbers)
print(f"Total: {sum(water_number)}")

for grp_id in range(total_layers):
    stay = sort_result == grp_id
    atoms_indies = np.all(stay, axis=0)  # bool index
    vacf = ACF_self(atom_y_velo[:, atoms_indies, :])
    data = np.column_stack((data, vacf))
    header_commnents.append(f"# layer {grp_id} stay atoms {np.count_nonzero(atoms_indies)}")
    header_labels.append(f'@ s{grp_id} legend  "layer_{grp_id}"')

header = "\n".join(header_commnents + header_labels)
np.savetxt(f"{output_dir.parent}/{output_dir.stem}.xvg",
           data,
           header=header,
           comments='', )

# ========================================
