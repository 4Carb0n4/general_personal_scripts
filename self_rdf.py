import numpy as np
from numba import cuda
import math
import argparse
import MDAnalysis as mda
from pathlib import Path



@cuda.jit
def distance_counts_cuda_kernal(pos_a: np.ndarray, pos_b: np.ndarray,
                                box_l: np.ndarray, counts, delta_bin):
    i, j = cuda.grid(2)
    if i >= pos_a.shape[1] or j >= pos_b.shape[1]:
        return
    # if i == j:
    #     return

    for f in range(pos_a.shape[0]):
        dx = pos_a[f, i, 0] - pos_b[f, j, 0]
        dx = dx - box_l[f, 0] * round(dx / box_l[f, 0])
        dy = pos_a[f, i, 1] - pos_b[f, j, 1]
        dy = dy - box_l[f, 1] * round(dy / box_l[f, 1])
        dz = pos_a[f, i, 2] - pos_b[f, j, 2]
        dz = dz - box_l[f, 2] * round(dz / box_l[f, 2])
        dr = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        bin_index = int(dr / delta_bin)  # int向0取整，dr为正数向下
        if bin_index < counts.shape[0]:
            cuda.atomic.add(counts, bin_index, 1)


class PositionsReader:
    def __init__(self, topol_file):
        pass

    @staticmethod
    def get_mono_com(u: mda.Universe):
        polymer_dp = -1
        output_data = np.zeros((polymer_dp, 3))
        for index in range(0, polymer_dp):
            # u.residues[0] 是第一个残基对象，.atoms 获取其原子组
            mono = u.residues[index].atoms + u.residues[index + polymer_dp].atoms

            output_data[index, :] = mono.center_of_mass(compound='group')
        return output_data

    @staticmethod
    def get_chain_com(u: mda.Universe):
        whole_chain = u.select_atoms("resname C")
        return whole_chain.center_of_mass(compound='group')

    @staticmethod
    def get_solvation_com(u: mda.Universe):
        solvation = u.select_atoms('resname SOL')
        position = np.empty(( int(u.trajectory.n_frames), len(solvation), 3))
        for ts in u.trajectory:
            position[ts, :, :] = solvation.center_of_mass(compound='residues')
        return position


class Orchestrator:
    def __init__(self):
        self.topol = None
        self.trajectory = None
        self.bins = None
        self.ref = None
        self.sel = None
        self.usage = None
        self.output = None

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--topol', help='拓扑,.tpr/.top etc', type=str, required=True)
        parser.add_argument('-f', '--trajectory', help='轨迹,.xtc/.trr etc', required=True)
        parser.add_argument('-b', '--bin', help='bins 球差分半径,default=1 Å', default=1, type=float)
        parser.add_argument('-o', '--output', help='输出文件', default=None)
        parser.add_argument('-u', '--usage', help='功能，计算rdf(-u rdf)或者数密度(-u pho)',
                            default="rdf")
        parser.add_argument('-ref', help='参考组,\n可选:mono_com,chain_com', required=True)
        parser.add_argument('-sel', help="计算组,\n可选:solvation_com,mono_com", required=True)
        args = parser.parse_args()

        self.topol = Path(args.topol)
        self.trajectory = Path(args.trajectory)
        self.bins = args.bin
        self.ref = args.ref
        self.sel = args.sel
        self.usage = args.usage
        self.output = Path(args.output)
        return None

    @staticmethod
    def distance_counts_shell(pos_a, pos_b, box_l, delta_bin, bin_numbers):
        """
        cuda计算的外层函数，负责划分block和数据交接
        """
        counts = np.zeros(bin_numbers)  # 用于数密度计数的数组，shape: [bins,]

        d_pos_a = cuda.to_device(pos_a)
        d_pos_b = cuda.to_device(pos_b)
        d_box_l = cuda.to_device(box_l)
        d_counts = cuda.to_device(counts)

        tpb = (32, 32)  # 性能和显卡相关性很强，32~256都可以，二维排布便于调用
        bpg = (math.ceil(pos_a.shape[1] / tpb[0]),
               math.ceil(pos_b.shape[1] / tpb[1]))
        distance_counts_cuda_kernal[bpg, tpb](
            d_pos_a, d_pos_b, d_box_l, d_counts, delta_bin)
        counts = d_counts.copy_to_host()
        return counts

    def cal_rdf(self, pos_a, pos_b, box_l, delta_bin):
        """
         计算RDF; 本函数 --> distance_counts_shell --> distance_counts_cuda_kernal
         :param pos_a: np.array[n_frames, n_atoms, 3] A原子坐标
         :param pos_b: np.array[n_frames, n_atoms, 3] B原子坐标
         :param box_l: np.array[n_frames,  3] 盒子边长
         :param delta_bin: float  划分球壳的宽度
         :return: np.array[bins,] 各个距离的计数器
         """
        # bin_cutoff = box_l.min() / 2 用了所有边长的最小值切断，兼容长方体盒子
        bin_numbers = int(box_l.min() / 2 / delta_bin)
        n_A_atoms = pos_a.shape[1]
        n_B_atoms = pos_b.shape[1]

        counts = self.distance_counts_shell(pos_a, pos_b, box_l, delta_bin, bin_numbers)

        # 归一化
        avg_volume = np.prod(box_l, axis=1).mean()  # [n_frames, 3] 计算每帧盒子体积再平均
        avg_number_density = n_B_atoms / avg_volume

        avg_counts = counts / n_A_atoms / pos_a.shape[0]  # 对参考A粒子;总帧数--做平均

        r_edges = np.arange(bin_numbers + 1) * delta_bin
        dV = (4.0 / 3.0) * np.pi * np.diff(r_edges ** 3) #每个球壳的体积

        rdf = avg_counts / dV / avg_number_density

        return rdf

    def cal_number_density(self, pos_a, pos_b, box_l, delta_bin):
        """
         计算数密度; 本函数 --> distance_counts_shell --> distance_counts_cuda_kernal
         :param pos_a: np.array[n_frames, n_atoms, 3] A原子坐标
         :param pos_b: np.array[n_frames, n_atoms, 3] B原子坐标
         :param box_l: np.array[n_frames,  3] 盒子边长
         :param delta_bin: float  划分球壳的宽度
         :return: np.array[bins,] 各个距离的计数器
         """
        # bin_cutoff = box_l.min() / 2 用了所有边长的最小值切断，利于长方体盒子
        bin_numbers = int(box_l.min() / 2 / delta_bin)
        n_A_atoms = pos_a.shape[1]

        counts = self.distance_counts_shell(pos_a, pos_b, box_l, delta_bin, bin_numbers)
        avg_counts = counts / n_A_atoms / pos_a.shape[0]  # 对参考A粒子;总帧数--做平均

        r_edges = np.arange(bin_numbers + 1) * delta_bin
        dV_shell = (4.0 / 3.0) * np.pi * np.diff(r_edges ** 3)

        num_density = avg_counts / dV_shell

        return num_density

    def get_standard_data(self):
        READER_MAP = {'mono_com': PositionsReader.get_mono_com,
                   'solvation_com': PositionsReader.get_solvation_com,
                   'chain_com': PositionsReader.get_chain_com
                   }

        u = mda.Universe(self.topol, self.trajectory)

        # 输入：frames, n_particles, 3
        box = np.zeros(( int(u.trajectory.n_frames) , 3))
        for index, ts in enumerate(u.trajectory):
            box[index, :] = [u.dimensions[0], u.dimensions[1], u.dimensions[2]]

        pos_a = READER_MAP[self.ref](u)
        pos_b = READER_MAP[self.sel](u) if self.sel != self.ref else pos_a
        print("read was done")

        return pos_a, pos_b, box

def main():
    orchestrator = Orchestrator()
    orchestrator.get_args()
    pos_a, pos_b, box = orchestrator.get_standard_data()

    if orchestrator.usage == 'rdf':
        data = orchestrator.cal_rdf(pos_a, pos_b, box, orchestrator.bins)
    elif orchestrator.usage == 'number_density':
        data = orchestrator.cal_number_density(pos_a, pos_b, box, orchestrator.bins)
    else:
        raise NotImplementedError(f"Unsupported usage:{orchestrator.usage}")

    out_name = orchestrator.output or f"num_den{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    np.savetxt(out_name, data)

if __name__ == "__main__":
    main()