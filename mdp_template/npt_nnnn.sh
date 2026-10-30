#!/bin/bash

# 设置基础路径
material_dir="/home/ldk/works/biye-2026-6/gmx_process/nvt"
prd_dir="/home/ldk/works/biye-2026-6/gmx_process/npt"

# 手动传入
sys_name=${1}
use_gpu=${2}
#if [[ "${2}" == "1" || -z "${2}" ]]; then
#  exit 1
#fi

#在下面使用gpuid

# 正确引用变量 ${sys_name}
target_dir="${material_dir}/${sys_name}"
output_dir="${prd_dir}/${sys_name}"

# 创建输出目录
mkdir -p "${output_dir}/npt-n"
cd "${output_dir}/npt-n" || exit 1

echo "正在处理: ${sys_name}"

# 注意：这里假设你的 nvt.mdp 文件在 target_dir/em 目录下，请根据实际位置修改 -f 路径
gmx grompp \
  -f "${prd_dir}/npt-n.mdp" \
  -c "${target_dir}/nvt.gro" \
  -p "${target_dir}/topol.top" \
  -o npt-n.tpr \
  -pp "${output_dir}/topol.top" && \
gmx mdrun -s npt-n.tpr -v -deffnm npt-n -ntmpi 1 -ntomp 12 -nb gpu -pin on -gpu_id "${use_gpu}"



# 正确引用变量 ${sys_name}
target_dir="${prd_dir}/${sys_name}"
output_dir="${prd_dir}/${sys_name}"

## 创建输出目录
#mkdir -p "${output_dir}/npt2"
#cd "${output_dir}/npt2" || exit
#
#echo "正在处理: ${sys_name}"
#
## 注意：这里假设你的 nvt.mdp 文件在 target_dir/em 目录下，请根据实际位置修改 -f 路径
#gmx grompp \
#  -f "${prd_dir}/npt2.mdp" \
#  -c "${target_dir}/npt-n/npt-n.gro" \
#  -p "${target_dir}/topol.top" \
#  -o npt2.tpr &&\
#gmx mdrun -v -deffnm npt2 -ntmpi 1 -ntomp 12 -nb gpu -pin on -s npt2.tpr -gpu_id "${use_gpu}"
#

echo "All missions completed"