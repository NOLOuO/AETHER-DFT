# Codex 任务说明：从周期 Al₂O₃ slab 构建 Gaussian F-center 团簇模型（含周期性处理）

## 0. 任务目标

用户要在 Gaussian 中研究 **800 ℃ 条件下 Al₂O₃ 表面 F-center / 氧空位与 H₂O 的作用**。  
当前阶段 **先不考虑羟基**，因此第一版模型采用：

```text
dehydroxylated / bare Al₂O₃(110)-D
```

这不是简单 `POSCAR -> gjf`。真正任务是：

```text
周期 VASP slab
→ 保留周期结构几何信息
→ 以目标表面 O 为中心构建 F-center
→ 用 PBC-aware 方法切有限 Gaussian 团簇
→ 用冻结外层 / ghost basis / 点电荷嵌入 / ECP cap 近似周期环境
→ 生成 Gaussian 输入文件
```

核心原则：

```text
F-center = 删除 O 后的氧空位
空位处不要放 O-ECP
空位处可选 ghost O basis
Gaussian 没有二维周期性，所以周期环境必须靠 embedded cluster 近似
```

---

## 1. 周期性到底怎么考虑

Gaussian 不能像 VASP 一样天然处理二维周期 slab。  
因此周期性要通过三层方式近似保留下来。

### 1.1 几何周期性：从周期 slab 切模型

不能凭空画团簇。必须从 VASP 的周期 slab 坐标出发。

操作：

```text
读取 POSCAR / CONTCAR
保留晶胞参数
Direct -> Cartesian
在 x/y 方向复制周期镜像
以中心 cell 中的目标 O 为 F-center 位点
按距离从复制后的周期结构中切团簇
```

这一步很关键。  
如果目标 F-center 靠近 cell 边界，直接在原始 cell 中切会漏掉跨周期边界的近邻原子。  
所以必须先生成扩胞，例如：

```text
3 × 3 × 1 slab replica
```

然后在中心 cell 选 vacancy O，从 replica 中按真实空间距离切 cluster。

### 1.2 局域结构周期性：冻结外层

Gaussian 团簇切出来后，没有无限晶格约束。  
因此要把外层原子冻结，保持它们接近周期 slab 里的位置。

推荐分区：

```text
active region: 0–5.5 Å
    F-center 周围 Al/O
    后续 H₂O 附近原子
    允许优化

frozen buffer: 5.5–7.5 Å
    外层 Al/O
    冻结，用来保持 slab 几何约束

outside region:
    第一版可丢弃
    正式版转成点电荷嵌入
```

如果边界污染明显，再扩大：

```text
active region: 0–6.0 Å
frozen buffer: 6.0–8.5 Å
```

### 1.3 静电周期性：点电荷嵌入

只切有限团簇会丢失无限 Al₂O₃ 晶格的长程 Madelung 静电场。  
正式版要在 Gaussian 团簇外加入点电荷近似外部周期环境。

推荐：

```text
QM cluster:
    F-center 周围真实 Al/O 原子

point charge shell:
    来自更大范围的周期 slab replica
    Al: +3
    O:  -2
    或使用 VASP Bader / CM5 / fitted charges

outer charge radius:
    第一版 12–15 Å
    收敛测试 20–30 Å
```

注意：

```text
点电荷不参与优化
点电荷位置来自原周期 slab
点电荷总电荷要检查，避免产生巨大非物理电场
外层点电荷可以按配位数缩放，减少边界偶极
```

### 1.4 防电子泄漏：正点电荷附近加 Al ECP cap

如果直接把外部 Al³⁺ 写成裸正点电荷，QM 电子可能非物理地塌到正点电荷附近。  
因此正式 embedded cluster 中，靠近 QM 区的外部 Al³⁺ 点电荷应使用 **Al ECP cap**。

推荐：

```text
外部 Al3+ 点电荷距离 QM cluster 边界 < 6–7 Å:
    加 Al ECP cap

外部更远的 Al3+ / O2-:
    只保留 point charge
```

这一步不是在 F-center 空位放 ECP。  
ECP cap 是放在 **边界/嵌入环境的阳离子位置**，用于防止电子泄漏。

---

## 2. 模型来源与高温假设

导师参考文献使用：

```text
Al₂O₃(110)-D surface
1 × 2 supercell
symmetrical nine-layer slab
15 Å vacuum
```

文献因为 423 K 下表面有羟基，所以构建了 1/3 monolayer hydroxylated Al₂O₃(110)-D。  
但用户当前条件是 **800 ℃**，并且要求 **先不考虑羟基**。因此第一版用：

```text
bare Al₂O₃(110)-D
```

如果能找到文献 SI 中的 bare Al₂O₃(110)-D POSCAR，优先使用。  
如果只有 Ru(Na) / Ru(H) 坐标，不要盲目删除“最后几个 O”。应通过 H 邻接、Ru 邻接、局部配位和人工检查来去掉羟基/水/Ru/Na 相关物种，恢复 bare Al₂O₃ 骨架。

---

## 3. Codex 需要实现的建模流程

### Step 1：读取周期结构并扩胞

脚本：`01_read_and_replicate_poscar.py`

功能：

```text
读取 POSCAR / CONTCAR
Direct -> Cartesian
输出原始 atom table
生成 3×3×1 replica
记录每个 replica atom 的：
    original_index
    image_shift
    element
    cartesian xyz
```

必须输出：

```text
parent_cartesian.xyz
parent_atoms.csv
replica_3x3x1.xyz
replica_atoms.csv
```

Python 脚本开头必须自动切换到脚本所在目录：

```python
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
```

---

### Step 2：恢复 bare Al₂O₃ 支撑体

脚本：`02_make_bare_support.py`

输入：

```text
parent_atoms.csv
```

功能：

```text
删除 Ru / Na / H
识别羟基 O / 水 O / Ru 相关额外 O
输出 bare support 候选结构
```

输出：

```text
bare_support_candidate.xyz
bare_support_candidate.csv
removed_atoms_report.csv
```

要求：

```text
不要只靠 O 的排序编号删除
必须输出删除原因：
    bonded_to_H
    bonded_to_Ru
    extra_O_manual
    low_coordination_surface_O
```

若无法自动判断，脚本应把所有可疑 O 输出到：

```text
suspect_O_for_manual_check.csv
```

---

### Step 3：选择 F-center 表面 O

脚本：`03_select_surface_oxygen.py`

功能：

```text
在 bare support 中筛选表面 O
计算 z 高度
计算 Al 邻居数，cutoff 建议 2.3 Å
输出候选 O
```

输出：

```text
candidate_surface_O.csv
```

候选表应包含：

```text
index
element
x y z
z_relative
n_Al_neighbors
Al_neighbor_indices
min_distance_to_boundary
recommendation_score
```

人工选择：

```text
vacancy_O_index = ?
vacancy_position = x y z
```

---

### Step 4：PBC-aware 切 Gaussian 团簇

脚本：`04_build_fcenter_cluster_pbc.py`

输入：

```text
bare support POSCAR / atom table
vacancy_O_index
active_radius
buffer_radius
replica_range = 3x3x1
ghost = true/false
```

核心算法：

```text
1. 在中心 cell 中定位 vacancy O
2. 删除该 O
3. 从 3×3×1 replica 中，以 vacancy_position 为中心按距离取原子
4. 只保留一个连续团簇
5. active_radius 内原子标记为 optimize
6. active_radius 到 buffer_radius 内原子标记为 frozen
7. 删除重复等价原子
8. 输出 Gaussian 坐标
```

输出：

```text
cluster_no_ghost.xyz
cluster_ghost.xyz
cluster_atoms.csv
cluster_regions.csv
```

Gaussian 冻结标记：

```text
active atom: 0
buffer atom: -1
```

---

## 4. F-center 在 Gaussian 里的表示

### 4.1 空位

```text
删除 O
空位处不写 O
空位处不写 O-ECP
```

### 4.2 ghost basis 对照

生成两套模型：

```text
A. no ghost:
   空位处什么都不放

B. ghost:
   在原 O 坐标处加 ghost O basis
```

优先尝试：

```text
O-Bq   x_vac   y_vac   z_vac
```

如果 Gaussian 版本不接受，则用：

```text
Bq   x_vac   y_vac   z_vac
```

并在 Gen basis 里给 Bq 指定 O-like basis。

---

## 5. 第一版 Gaussian 输入

第一版先不用点电荷和 ECP cap，只做：

```text
PBC-aware finite cluster
frozen buffer
no ghost / ghost contrast
spin-state contrast
```

生成：

```text
fcenter_no_ghost_0_1_sp.gjf
fcenter_no_ghost_0_3_sp.gjf
fcenter_no_ghost_p1_2_sp.gjf

fcenter_ghost_0_1_sp.gjf
fcenter_ghost_0_3_sp.gjf
fcenter_ghost_p1_2_sp.gjf
```

电荷/自旋：

```text
0 1   neutral F-center singlet
0 3   neutral F-center triplet
+1 2  F+ center doublet
```

关键词：

```gjf
%chk=fcenter_ghost_p1_2_sp.chk
%mem=64GB
%nprocshared=24
#p UB3LYP/Gen EmpiricalDispersion=GD3BJ
   Nosymm Guess=Mix SCF=(XQC,MaxCycle=500)
   Int=UltraFine Pop=Full Stable=Opt

Bare Al2O3 F-center cluster from PBC-aware cut

1 2
Al   0   x y z
O    0   x y z
Al  -1   x y z
O   -1   x y z
O-Bq    x_vac y_vac z_vac

Al O 0
def2SVP
****
```

如果 ghost 用 `Bq`，增加：

```gjf
Bq 0
def2SVP
****
```

---

## 6. 正式版 Gaussian：加入周期静电嵌入

如果第一版出现边界态污染，进入正式 embedded cluster。

脚本：`05_add_point_charge_embedding.py`

功能：

```text
从 3×3×1 或更大 replica 中选择不属于 QM cluster 的外部 Al/O
按距离构建 point charge shell
Al = +3
O = -2
必要时缩放最外层点电荷
靠近 QM 的外部 Al 点电荷标记为 ECP cap candidate
```

输出：

```text
embedding_charges.dat
ecp_cap_sites.xyz
fcenter_embedded_ghost_p1_2_sp.gjf
```

建议：

```text
charge shell radius: 15 Å 起步
convergence test: 20 Å, 25 Å, 30 Å
ECP cap distance: external Al within 6–7 Å from QM cluster edge
```

Gaussian 里如何写外部点电荷取决于使用的 Gaussian 版本和接口。Codex 需要给出两种写法：

```text
方案 A：Gaussian external point charges input
方案 B：把点电荷/嵌入交给 ChemShell 或其他 QM/MM wrapper
```

不要把这些点电荷写成普通原子参与优化。

---

## 7. 周期性验证标准

Gaussian 团簇近似周期 slab 后，必须做验证。

### 7.1 半径收敛

至少测试：

```text
active/buffer = 5.5/7.5 Å
active/buffer = 6.0/8.5 Å
```

比较：

```text
defect orbital 位置
spin density 位置
<S**2>
relative energy
```

### 7.2 ghost 收敛

比较：

```text
no ghost vs ghost O basis
```

如果 ghost 改变巨大，说明空位电子基函数不足，需保留 ghost 并进一步测试 basis。

### 7.3 点电荷半径收敛

正式版比较：

```text
no embedding
15 Å point charges
20 Å point charges
30 Å point charges
```

看缺陷态和自旋密度是否稳定。

### 7.4 与 VASP 周期结果对照

如果能跑 VASP slab 的氧空位模型，应对照：

```text
缺陷态是否在 gap 中
spin density 是否在氧空位附近
F+ center 是否给出一个未成对电子
相对稳定的 charge/spin state 是否一致
```

Yang & Wu 2024 特别提醒：边界 ECP 会显著影响前线轨道能级。因此如果要做正式 embedded cluster，需要用周期 slab 的能级或真空能级对齐结果作为参考，而不是只相信裸 cluster 的 HOMO/LUMO。

---

## 8. 后续加入 H₂O

裸 F-center 验证后再放 H₂O。

初始构型：

```text
O_water 指向 vacancy 周围缺配位 Al
H_water 指向邻近表面 O
O_water–Al 初始距离 1.9–2.3 Å
H_water–O_surface 初始距离 1.5–2.0 Å
```

生成：

```text
h2o_fcenter_ads_opt.gjf
h2o_fcenter_dissociation_scan.gjf
```

不要在裸 F-center 电子结构没验证前直接讨论 H₂O 解离。

---

## 9. Codex 需要最终输出的文件

```text
01_read_and_replicate_poscar.py
02_make_bare_support.py
03_select_surface_oxygen.py
04_build_fcenter_cluster_pbc.py
05_add_point_charge_embedding.py
06_generate_gaussian_inputs.py
07_check_gaussian_outputs.py

candidate_surface_O.csv
removed_atoms_report.csv
cluster_atoms.csv
cluster_regions.csv
embedding_charges.dat
README_run_order.md
```

`07_check_gaussian_outputs.py` 要提取：

```text
Normal termination
SCF energy
<S**2>
charge/multiplicity
HOMO/LUMO if available
```

---

## 10. 第一版不要做的事

```text
不要把 F-center 空位写成 O-ECP
不要把整块 POSCAR 直接当大分子丢进 Gaussian
不要直接优化全部原子
不要一开始放 H2O
不要一开始加 Ru / Na
不要把 423 K 羟基化模型作为 800 ℃ bare 支撑体
不要忽略 PBC，直接在原始 cell 边界切团簇
```

---

## 11. 参考文献

1. Ruixuan Qin, Lingyun Zhou, Pengxin Liu, Yue Gong, Kunlong Liu, Chaofa Xu, Yun Zhao, Lin Gu, Gang Fu, Nanfeng Zheng. **Alkali ions secure hydrides for catalytic hydrogenation.** *Nature Catalysis* 2020, 3, 703–709. DOI: 10.1038/s41929-020-0481-6.  
   - 用途：导师参考模型来源；Al₂O₃(110)-D slab、1×2 supercell、nine-layer slab、15 Å vacuum、羟基覆盖温度判断。

2. Benjamin X. Shi, Venkat Kapil, Andrea Zen, Ji Chen, Ali Alavi, Angelos Michaelides. **General embedded cluster protocol for accurate modeling of oxygen vacancies in metal-oxides.** *Journal of Chemical Physics* 2022, 156, 124704. DOI: 10.1063/5.0087031.  
   - 用途：氧空位 embedded cluster 方法核心依据；O vacancy-centered quantum cluster、point charge embedding、ECP cap、防电子泄漏。

3. Ming-Yu Yang, Xin-Ping Wu. **Level-Shifted Embedded Cluster Method for Modeling the Chemistry of Metal Oxides.** *Journal of Chemical Theory and Computation* 2024, 20, 1386–1397. DOI: 10.1021/acs.jctc.3c01123.  
   - 用途：说明边界 ECP 会影响前线轨道能级；需要用周期参考能级校准或验证 embedded cluster。

4. Ekaterina G. Ragoyja, Vitaly E. Matulis, Oleg A. Ivashkevich, Dmitry A. Lyakhov, Dominik Michels. **Computationally Effective Approach for Studies of Mechanism and Thermodynamics of Heterogeneous Catalytic Processes on Metal Oxides.** *International Journal of Quantum Chemistry* 2024, 124, e27470. DOI: 10.1002/qua.27470.  
   - 用途：Gaussian16 下金属氧化物多层团簇模型参考；γ-Al₂O₃(110)；显式层 + ECP soft charges + 点电荷层。

5. You Lu et al. **Multiscale QM/MM modelling of catalytic systems with ChemShell.** *Physical Chemistry Chemical Physics* 2023, 25, 21816–21835. DOI: 10.1039/D3CP00648D.  
   - 用途：QM/MM 与 solid-state embedding 的现代方法背景；活性中心 QM，环境用嵌入场/点电荷近似。

---

## 12. 最终一句话

Gaussian 不能直接保留 VASP 的二维周期性，因此必须：

```text
用周期 slab 作为母体
用 PBC-aware replica 切团簇
用冻结外层保持周期几何约束
必要时用点电荷 + Al ECP cap 恢复周期静电环境
用 VASP 周期氧空位模型验证 Gaussian 团簇的缺陷态和自旋密度
```

这才是把导师周期模型转换为 Gaussian F-center 计算的正确路线。
