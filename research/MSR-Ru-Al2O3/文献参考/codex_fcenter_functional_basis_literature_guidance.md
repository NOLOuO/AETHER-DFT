# Codex 指导文件：Al₂O₃ F-center Gaussian 计算的泛函与基组选择

## 0. 目标

本文件用于指导 Codex 在已有“周期 slab → Gaussian 团簇”的 workflow 基础上，进一步自动生成不同泛函/基组层级的 Gaussian 输入文件。

研究对象：

```text
体系：dehydroxylated / bare Al₂O₃(110)-D 表面 F-center / 氧空位
条件：800 ℃，第一阶段先不考虑羟基
软件：Gaussian16
目标：先验证裸 F-center 电子结构，再加入 H₂O
```

核心判断：

```text
F-center = 删除 O 后的氧空位
空位处不要放 O-ECP
空位处可选 ghost O basis
Gaussian 没有周期性，所以模型必须来自 PBC-aware 切团簇，并用冻结外层 / 点电荷 / ECP cap 补偿周期环境
泛函与基组必须做层级验证，不能只跑一个方法
```

---

## 1. 文献依据

### 1.1 Shi et al., JCP 2022

**General embedded cluster protocol for accurate modeling of oxygen vacancies in metal-oxides**  
J. Chem. Phys. 2022, 156, 124704. DOI: 10.1063/5.0087031.

要点：

```text
1. 氧空位 Ov 的形成能和电子结构对泛函非常敏感。
2. 标准 GGA 对氧空位通常不够可靠，容易导致缺陷电子离域。
3. embedded cluster 方法适合用于氧空位，因为可在有限 quantum cluster 上使用更高等级电子结构方法。
4. quantum cluster 应以氧空位为中心构建。
5. 外部环境可用 point charges 恢复长程静电势。
6. 靠近 QM 区的正点电荷需要 ECP cap 防电子泄漏。
7. 空位处可以放 oxygen basis functions；对本项目可理解为 ghost O basis 对照。
```

落实到本项目：

```text
- 不用纯 PBE/PW91 作为唯一结论。
- 第一轮用 UB3LYP-D3(BJ) 跑通。
- 精修用 UPBE0-D3(BJ) 做关键单点。
- 必须做 no ghost / ghost O basis 对照。
- 正式嵌入模型再考虑 point charges + Al ECP cap。
```

---

### 1.2 Yang & Wu, JCTC 2024

**Level-Shifted Embedded Cluster Method for Modeling the Chemistry of Metal Oxides**  
J. Chem. Theory Comput. 2024, 20, 1386–1397. DOI: 10.1021/acs.jctc.3c01123.

要点：

```text
1. 金属氧化物的 finite QM cluster 必须正确处理边界。
2. 未调节的 boundary ECP 可能使 HOMO/LUMO 能级明显偏离周期参考模型。
3. frontier orbital energy levels 对化学性质很关键。
4. ECP cap 不能乱加，必须检查边界是否引入假轨道。
```

落实到本项目：

```text
- 第一版先用 finite cluster + frozen buffer，不强行加 ECP cap。
- 如果边界态污染明显，再引入 point charges + Al ECP cap。
- 使用 ECP cap 后必须检查 defect orbital / SOMO 是否仍在 F-center 附近。
- 不允许把 O-ECP 放到空位处。
```

---

### 1.3 Ragoyja et al., IJQC 2024

**Computationally Effective Approach for Studies of Mechanism and Thermodynamics of Heterogeneous Catalytic Processes on Metal Oxides**  
Int. J. Quantum Chem. 2024, 124, e27470. DOI: 10.1002/qua.27470.

要点：

```text
1. 该文献直接使用 Gaussian16。
2. 对 γ-Al₂O₃(110) 构建 cluster / multilayer cluster。
3. 主要方法为 TPSSh/6-311G*。
4. Al soft charges 使用 SDD pseudopotential。
5. 三层模型：显式 QM 原子 + ECP soft charges + point charge array。
6. 用 HOMO-LUMO gap、吸附能、几何、频率和热力学性质验证模型。
```

落实到本项目：

```text
- TPSSh/6-311G* 作为 literature-consistent check。
- 如果做三层 embedded cluster，可以参考：QM 层 TPSSh/6-311G*，Al soft charge 层 SDD，外层 point charges。
- 对 F-center 电子局域性，仍需 B3LYP/PBE0 作为主验证。
```

---

### 1.4 Qin et al., Nature Catalysis 2020

**Alkali ions secure hydrides for catalytic hydrogenation**  
Nature Catalysis 2020, 3, 703–709. DOI: 10.1038/s41929-020-0481-6.

要点：

```text
1. 导师参考模型来自 Al₂O₃(110)-D slab。
2. 周期模型为 1×2 supercell、nine-layer slab、15 Å vacuum。
3. 文献在 423 K 下采用 1/3 ML hydroxylated surface。
4. 用户当前为 800 ℃，第一阶段可以采用 dehydroxylated / bare Al₂O₃(110)-D。
5. 文献中 g ≈ 2.002 EPR 信号被归因于 Al₂O₃ 氧空位中的 F+ center。
```

落实到本项目：

```text
- 第一阶段不放 Ru / Na / OH / H₂O。
- 先建立 bare Al₂O₃(110)-D F-center。
- 必须计算 +1 2，即 F+ center doublet。
```

---

## 2. 总体方法层级

Codex 需要生成 4 个 level 的 Gaussian 输入。

```text
Level 1: 快速筛选
Level 2: 局部优化
Level 3: 精修单点验证
Level 4: 文献一致性对照
```

---

## 3. Level 1：快速筛选

### 3.1 目的

```text
检查 SCF 是否能收敛
比较 charge/multiplicity
比较 no ghost / ghost O basis
查看 defect orbital / spin density 是否在 F-center 附近
```

### 3.2 方法

```text
Functional: UB3LYP-D3(BJ)
Basis: def2SVP
Route:
#p UB3LYP/Gen EmpiricalDispersion=GD3BJ Nosymm Guess=Mix SCF=(XQC,MaxCycle=500) Int=UltraFine Pop=Full Stable=Opt
```

### 3.3 需要生成的输入

```text
fcenter_no_ghost_0_1_sp_b3lyp_def2svp.gjf
fcenter_no_ghost_0_3_sp_b3lyp_def2svp.gjf
fcenter_no_ghost_p1_2_sp_b3lyp_def2svp.gjf

fcenter_ghost_0_1_sp_b3lyp_def2svp.gjf
fcenter_ghost_0_3_sp_b3lyp_def2svp.gjf
fcenter_ghost_p1_2_sp_b3lyp_def2svp.gjf
```

电荷/自旋：

```text
0 1   neutral F-center singlet
0 3   neutral F-center triplet
1 2   F+ center doublet
```

---

## 4. Level 2：局部优化

### 4.1 选择标准

只对 Level 1 中电子结构合理的态做优化。

合理标准：

```text
Normal termination
<S**2> 合理
HOMO/SOMO/defect orbital 在空位附近
spin density 不跑到边界
ghost/no ghost 结果不出现非物理巨大差异
```

### 4.2 方法

```text
Functional: UB3LYP-D3(BJ)
Basis: def2SVP
Route:
#p UB3LYP/Gen EmpiricalDispersion=GD3BJ Opt=(CalcFC,MaxCycle=200) Nosymm Guess=Mix SCF=(XQC,MaxCycle=500) Int=UltraFine Pop=Full Stable=Opt
```

### 4.3 冻结规则

坐标中：

```text
active region: 0
frozen buffer: -1
ghost center: 通常不加冻结标记；若 Gaussian 报错，改用 Bq 并单独给基组
```

示例：

```gjf
1 2
Al   0   x y z
O    0   x y z
Al  -1   x y z
O   -1   x y z
O-Bq    x_vac y_vac z_vac
```

---

## 5. Level 3：精修单点验证

### 5.1 目的

```text
验证 F-center 电子局域性是否不依赖 B3LYP
验证 F+ doublet 是否稳定
验证 ghost O basis 是否必要
```

### 5.2 方法

```text
Functional: UPBE0-D3(BJ)
Basis:
  active region + ghost O: def2TZVP
  frozen buffer: def2SVP
Route:
#p UPBE1PBE/Gen EmpiricalDispersion=GD3BJ Nosymm Guess=Read SCF=(XQC,MaxCycle=500) Int=UltraFine Pop=Full Stable=Opt
```

说明：

```text
Gaussian 中 PBE0 常用关键词为 PBE1PBE；开壳层用 UPBE1PBE。
如果 Guess=Read 或 Geom=AllCheck 与 Gen basis 段冲突，Codex 应生成显式坐标版本。
```

---

## 6. Level 4：文献一致性对照

### 6.1 目的

与 Ragoyja 2024 的 Gaussian16 γ-Al₂O₃ cluster 文献保持可比。

### 6.2 方法

```text
Functional: TPSSh
Basis: 6-311G*
Route:
#p UTPSSh/Gen Nosymm Guess=Mix SCF=(XQC,MaxCycle=500) Int=UltraFine Pop=Full Stable=Opt
```

说明：

```text
Gaussian 中 TPSSh 关键词可能因版本不同写作 TPSSh 或 TPSSTPSS。
Codex 需要在 README 中提醒用户根据 Gaussian 版本确认。
此级别只对最终筛选出的合理态做单点，不必全矩阵跑。
```

---

## 7. 基组分层规则

Codex 生成 Gen basis 段时，优先按原子编号分组，不要只按元素统一分组。

原因：

```text
同样是 Al/O，active region 和 frozen buffer 可能用不同基组。
ghost O basis 需要单独指定。
```

### 7.1 Level 1 / 2

统一基组：

```gjf
Al O 0
def2SVP
****
```

若有 H₂O：

```gjf
Al O H 0
def2SVP
****
```

### 7.2 Level 3

分层基组：

```gjf
<active_atom_indices> 0
def2TZVP
****

<buffer_atom_indices> 0
def2SVP
****
```

如果使用 `Bq` ghost：

```gjf
Bq 0
def2TZVP
****
```

如果使用 `O-Bq`，优先让其使用 O 的 def2TZVP；如果 Gaussian 报错，切换到 `Bq-with-basis` 模式。

---

## 8. Ghost O basis 模式

Codex 需要支持三种模式：

```text
ghost_mode = none
ghost_mode = O-Bq
ghost_mode = Bq-with-basis
```

优先：

```gjf
O-Bq   x_vac y_vac z_vac
```

备用：

```gjf
Bq   x_vac y_vac z_vac
```

并添加：

```gjf
Bq 0
def2SVP
****
```

或精修：

```gjf
Bq 0
def2TZVP
****
```

---

## 9. 何时用 GenECP

### 9.1 第一版不用 GenECP

裸 Al₂O₃ F-center 第一版：

```text
Al/O/H 全电子基组
不加 ECP
Route 用 /Gen
```

### 9.2 以下情况才用 GenECP

```text
加入 Al ECP cap
加入 soft charge layer
加入 Ru / Ag / 其他重元素
```

Route：

```gjf
#p UB3LYP/GenECP EmpiricalDispersion=GD3BJ
```

注意：

```text
ECP 段只给 ECP cap、soft charges 或重元素。
不要给 F-center 空位加 O-ECP。
不要给所有 Al/O 统一加 ECP。
```

---

## 10. 点电荷与 ECP cap

如果有限团簇出现边界态污染，进入 embedded cluster 正式版。

### 10.1 点电荷

```text
Al point charge: +3
O point charge: -2
```

可选高级方案：

```text
Bader / CM5 / fitted charges
```

### 10.2 ECP cap

```text
位置：靠近 QM 区的外部 Al3+ 点电荷位置
目的：防止电子泄漏到正点电荷
不是：F-center 空位
```

### 10.3 验证

加入 ECP cap 后必须检查：

```text
HOMO/SOMO 是否仍在 F-center 附近
边界 ECP 附近是否出现假轨道
与 no ECP cap / VASP 周期模型是否趋势一致
```

---

## 11. H₂O 吸附阶段

只有裸 F-center 通过验证后再加 H₂O。

### 11.1 优化

```text
UB3LYP-D3(BJ)/def2SVP
```

### 11.2 关键单点

```text
UPBE0-D3(BJ)
H₂O + F-center active region + ghost O: def2TZVP
buffer: def2SVP
```

### 11.3 不建议

```text
不要没验证裸 F-center 就直接讨论 H₂O 解离
不要直接用纯 PBE 下结论
不要在 H₂O 计算中随意改变裸 F-center 的 charge/multiplicity
```

---

## 12. Codex 需要新增脚本

在原有周期 workflow 基础上新增：

```text
08_generate_basis_groups.py
09_generate_gaussian_levels.py
10_compare_functional_basis_results.py
```

### 12.1 `08_generate_basis_groups.py`

输入：

```text
cluster_atoms.csv
cluster_regions.csv
ghost_mode
```

输出：

```text
basis_groups_level1.json
basis_groups_level3.json
```

内容示例：

```json
{
  "level1": {
    "all_real_atoms": "def2SVP",
    "ghost": "def2SVP"
  },
  "level3": {
    "active_atoms": "def2TZVP",
    "buffer_atoms": "def2SVP",
    "ghost": "def2TZVP"
  }
}
```

### 12.2 `09_generate_gaussian_levels.py`

功能：

```text
生成 Level 1 / Level 2 / Level 3 / Level 4 gjf
自动命名
自动写 chk
自动写 charge/multiplicity
自动写 Gen 或 GenECP
支持 ghost_mode
支持 active/buffer 分层基组
```

输出目录：

```text
inputs_level1/
inputs_level2/
inputs_level3/
inputs_level4/
```

### 12.3 `10_compare_functional_basis_results.py`

提取：

```text
Normal termination
SCF energy
<S**2>
charge/multiplicity
method
basis level
ghost mode
HOMO-LUMO gap if available
warnings
```

输出：

```text
functional_basis_summary.csv
```

---

## 13. 默认参数

```yaml
functional_first_pass: UB3LYP
dispersion: GD3BJ
basis_first_pass: def2SVP

functional_refined_sp: UPBE1PBE
basis_active_refined: def2TZVP
basis_buffer_refined: def2SVP

functional_literature_check: TPSSh
basis_literature_check: 6-311G*

ghost_modes:
  - none
  - O-Bq

charge_multiplicity:
  - [0, 1]
  - [0, 3]
  - [1, 2]

active_radius: 5.5
buffer_radius: 7.5
replica: [3, 3, 1]
```

---

## 14. 命名规范

```text
fcenter_{ghostmode}_{charge}_{mult}_{job}_{functional}_{basis}.gjf
```

示例：

```text
fcenter_noghost_0_1_sp_b3lyp_def2svp.gjf
fcenter_ghost_p1_2_opt_b3lyp_def2svp.gjf
fcenter_ghost_p1_2_sp_pbe0_activeTZVP_bufferSVP.gjf
fcenter_ghost_p1_2_sp_tpssh_6311gstar.gjf
```

---

## 15. 最小可行计算矩阵

```text
1. UB3LYP-D3(BJ)/def2SVP
   no ghost vs ghost
   0 1 / 0 3 / +1 2
   single point

2. 对合理态：
   UB3LYP-D3(BJ)/def2SVP
   local optimization

3. 对优化后合理态：
   UPBE0-D3(BJ)
   active def2TZVP / buffer def2SVP
   single point

4. 对最终态：
   TPSSh/6-311G*
   literature-consistent single point
```

若 `+1 2` 态给出空位附近单电子缺陷态，且 spin density 不跑到边界，则优先作为 F+ center 模型进入 H₂O 吸附阶段。

---

## 16. 禁止事项

```text
不要把 F-center 空位写成 O-ECP
不要把纯 GGA 当唯一结论
不要全体系一开始就 def2TZVP
不要随便给所有 Al/O 加 ECP
不要使用边界 ECP 后不检查 frontier orbitals
不要忽略 no ghost / ghost O basis 对照
不要没验证裸 F-center 就直接放 H₂O
```
