#   按纵向：不同体系进行管理

import subprocess as sp
from pathlib import Path
import shutil

# ----------------------------------
SYSTEM_NAME = "60wt"
STEP_NAME = ["em", ]
STRUCTURE_DIR = Path("/home1/ldk/works/hydrogels/model_building/gmx_solvate/60wt/solvated.gro")
GPU_ID = 3
TOPOL_DIR = Path("/home1/ldk/works/hydrogels/model_building/gmx_solvate/60wt/topol.top")
MDP_DIR = Path("/home1/ldk/works/hydrogels/aa_gmx/mdp")
OUTPUT_SYSTEM_DIR = Path("/home1/ldk/works/hydrogels/aa_gmx")

MAX_WARNINGS = 9
# ----------------------------------


class SimulationOrchestrator:
    def __init__(self):
        self.gpu_id = GPU_ID
        self.gro = STRUCTURE_DIR
        self.cpt = None
        self.top = TOPOL_DIR
        self.step_name = STEP_NAME
        self.all_mdp = self.get_mdp_files(MDP_DIR, STEP_NAME)
        self.mdp = None
        self.system_root_dir = OUTPUT_SYSTEM_DIR / SYSTEM_NAME
        self.output = OUTPUT_SYSTEM_DIR / SYSTEM_NAME
        self.tpr = None


    @staticmethod
    def get_mdp_files( mdp: Path, step_name)->list[list]:
        mdp_file_list = []
        for name_per_step in step_name:
            step_mdp = []

            # 依次检查 0, 1, 无后缀 的mdp
            if (mdp/f"{name_per_step}0.mdp").is_file():
                step_mdp.append(mdp / f"{name_per_step}0.mdp")
            if (mdp/f"{name_per_step}1.mdp").is_file():
                step_mdp.append(mdp / f"{name_per_step}1.mdp")
            #没有就无后缀
            if not step_mdp and (mdp / f"{name_per_step}.mdp").is_file():
                step_mdp.append(mdp / f"{name_per_step}.mdp")

            mdp_file_list.append(step_mdp)
            if not step_mdp:
                raise ValueError(f"MDP file {name_per_step} not found")

        return mdp_file_list

    def gmx_grompp(self,):
        if self.cpt is None:
            command_line = ['gmx', 'grompp',
                            '-f', str(self.mdp),
                            '-c', str(self.gro),
                            '-p', str(self.top),
                            '-o', str(self.output / "topol.tpr"),
                            '-maxwarn', str(MAX_WARNINGS)]
        else:
            command_line = ['gmx', 'grompp',
                            '-f', str(self.mdp),
                            '-c', str(self.gro),
                            '-p', str(self.top),
                            '-o', str(self.output / "topol.tpr"),
                            '-t', str(self.cpt),
                            '-maxwarn', str(MAX_WARNINGS)]
        sp.run(command_line, check=True,cwd=str(self.output))
        self.tpr = self.output / "topol.tpr"


    def gmx_mdrun(self,):
        if self.gpu_id is None:
            command_line = ['gmx', 'mdrun',
                            '-s', str(self.tpr),
                            '-deffnm', str(self.output.name),
                            '-ntmpi', '1', '-ntomp', '10',
                            '-pin', 'on']
        else:
            command_line = ['gmx', 'mdrun',
                            '-s', str(self.tpr),
                            '-deffnm', str(self.output.name),
                            '-ntmpi', '1', '-ntomp', '10',
                            '-gpu_id', str(self.gpu_id),
                            '-pin', 'on',
                            '-nb', 'gpu']
        sp.run(command_line, check=True, cwd=str(self.output))

    # 创建系统目录，复制top
    def copy_topol(self):
        target_topol = self.system_root_dir / "topol.top"
        if not target_topol.is_file():
            self.system_root_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(TOPOL_DIR, target_topol)
        self.top = target_topol

    def init_per_step(self, mdp, name_per_step):
        self.mdp = mdp
        self.cpt = self.gro.with_suffix(".cpt") \
            if self.gro.with_suffix(".cpt").is_file() else None
        self.output = self.system_root_dir / name_per_step / mdp.stem
        self.output.mkdir(parents=True, exist_ok=True)

    def switch_to_next_mdp(self):
        self.cpt = self.output / (self.output.name + ".cpt")
        self.gro = self.output / (self.output.name + ".gro")

def main():
    orchestrator = SimulationOrchestrator()
    orchestrator.copy_topol()

    #外层循环体系，内层循环mdp
    for i, name_per_step in enumerate(STEP_NAME):
        for step_mdp in orchestrator.all_mdp[i]:
            orchestrator.init_per_step(step_mdp, name_per_step)
            orchestrator.gmx_grompp()
            orchestrator.gmx_mdrun()
            orchestrator.switch_to_next_mdp()

if __name__ == "__main__":
    main()
