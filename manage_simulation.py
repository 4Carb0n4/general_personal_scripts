#   按纵向：不同体系进行管理

import subprocess as sp
from pathlib import Path
import shutil

# ----------------------------------
SYSTEM_NAME = ["50wt", "60wt", "70wt", ]
STEP_NAME = ["vacf", ]
STRUCTURE_DIR = Path("/home1/ldk/works/hydrogels/aa_gmx/45wt/npt-c/npt9/npt9.gro")
GPU_ID = 2
TOPOL_DIR = Path("/home1/ldk/works/hydrogels/aa_gmx/45wt/topol.top")
MDP_DIR = Path("/home1/ldk/works/hydrogels/aa_gmx/mdp")
OUTPUT_MOTHER_DIR = Path("/home1/ldk/works/hydrogels/aa_gmx")

MAX_WARNINGS = 1


# ----------------------------------


class SimulationOrchestrator:
    def __init__(self):
        self.system_name = None
        self.gpu_id = None
        self.gro = None
        self.cpt = None
        self.top = None
        self.step_name = None
        self.all_mdp = None
        self.mdp = None
        self.system_root_dir = None
        self.output = None
        self.tpr = None

    def get_input_args(self):
        self.system_name = SYSTEM_NAME
        self.gpu_id = GPU_ID
        self.gro = STRUCTURE_DIR
        self.top = TOPOL_DIR
        self.step_name = STEP_NAME
        self.all_mdp = self.get_mdp_files(MDP_DIR, STEP_NAME)

    @staticmethod
    def get_mdp_files(mdp: Path, step_name) -> list[list[Path]]:
        mdp_file_list = []
        for name_per_step in step_name:
            step_mdp = []
            step_path = mdp.glob(f"{name_per_step}[0-9].mdp")
            step_mdp.extend(sorted(step_path))
            # 没有就无后缀
            if not step_mdp and (mdp / f"{name_per_step}.mdp").is_file():
                step_mdp.append(mdp / f"{name_per_step}.mdp")

            mdp_file_list.append(step_mdp)
            if not step_mdp:
                raise ValueError(f"MDP file {name_per_step} not found")

        return mdp_file_list

    def gmx_grompp(self, ):
        command_line = ['gmx', 'grompp',
                        '-f', str(self.mdp),
                        '-c', str(self.gro),
                        '-p', str(self.top),
                        '-o', str(self.output / "topol.tpr"),
                        '-maxwarn', str(MAX_WARNINGS)]
        if self.cpt is not None:
            command_line.extend(['-t', str(self.cpt)])

        sp.run(command_line, check=True, cwd=str(self.output))
        self.tpr = self.output / "topol.tpr"

    def gmx_mdrun(self, step_name=None):
        command_line = ['gmx', 'mdrun',
                        '-s', str(self.tpr),
                        '-deffnm', str(self.output.name),
                        '-ntmpi', '1', '-ntomp', '10',
                        '-pin', 'on']

        if self.gpu_id is not None:
            command_line.extend(['-gpu_id', str(self.gpu_id),
                                 '-nb', 'gpu'])
        # if step_name != "em":
        #     command_line.extend(['-pme', 'gpu'])

        sp.run(command_line, check=True, cwd=str(self.output))

    # 创建系统目录，复制top
    def copy_topol(self):
        target_topol = self.system_root_dir / "topol.top"
        if not target_topol.is_file():
            self.system_root_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.top, target_topol)
        self.top = target_topol

    def init_per_step(self, mdp, name_per_step, name_per_sys):
        self.mdp = mdp
        self.cpt = self.gro.with_suffix(".cpt") \
            if self.gro.with_suffix(".cpt").is_file() else None
        self.output = self.system_root_dir / name_per_step / mdp.stem
        self.output.mkdir(parents=True, exist_ok=True)

    def switch_to_next_step(self):
        self.gro = self.output / (self.output.name + ".gro")

    def init_per_sys_n(self, name_per_sys):
        # TOPOL_DIR 形如 .../gmx_solvate/45wt/topol.top
        template_sys = TOPOL_DIR.parent.name

        # 从固定的 STRUCTURE_DIR 推导初始结构，只替换体系目录名
        gro_parts = list(STRUCTURE_DIR.parts)
        gro_parts[gro_parts.index(template_sys)] = name_per_sys

        self.gro = Path(*gro_parts)
        self.top = TOPOL_DIR.parent.parent / name_per_sys / TOPOL_DIR.name
        self.system_root_dir = OUTPUT_MOTHER_DIR / name_per_sys
        self.cpt = None

        for path in (self.gro, self.top):
            if not path.is_file():
                raise FileNotFoundError(f"{name_per_sys} 缺少输入文件: {path}")

    def init_per_sys(self, name_per_sys, ref_gro, ref_top):
        ref_sys = ref_top.parent.name

        gro_parts = list(ref_gro.parts)
        gro_parts[gro_parts.index(ref_sys)] = name_per_sys

        self.gro = Path(*gro_parts)
        self.top = ref_top.parent.parent / name_per_sys / ref_top.name
        self.system_root_dir = OUTPUT_MOTHER_DIR / name_per_sys



def main():
    orch = SimulationOrchestrator()
    orch.get_input_args()

    # 外层循环体系，内层循环mdp
    temp = (orch.gro, orch.top)
    for idx, name_per_sys in enumerate(orch.system_name):
        orch.init_per_sys(name_per_sys, temp[0], temp[1])
        orch.copy_topol()

        for i, name_per_step in enumerate(STEP_NAME):
            for step_mdp in orch.all_mdp[i]:
                orch.init_per_step(step_mdp, name_per_step, name_per_sys)
                orch.gmx_grompp()
                orch.gmx_mdrun()
                orch.switch_to_next_step()


if __name__ == "__main__":
    main()
