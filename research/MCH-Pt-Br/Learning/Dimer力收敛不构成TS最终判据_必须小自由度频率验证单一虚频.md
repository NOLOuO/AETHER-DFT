# Dimer力收敛不构成TS最终判据_必须小自由度频率验证单一虚频

- Captured: 2026-06-15 16:09:17
- Tags: #TS #Dimer #频率验证 #收敛判据 #noBr #MKM #可复用规则

# Dimer力收敛不构成TS最终判据——必须小自由度频率验证单一虚频

## 来源
2026-06-15 noBr TS6 Dimer 收敛复测。远端路径: `/home/szhang/research/MCH-Pt-Br/MKM_actual_data_package_20260605/noBr/structures/16_TS6`

## 核心判断
Dimer 力收敛（max_force < EDIFFG）是 TS 搜索的必要条件，但不是充分条件。收敛后的 CONTCAR 不能直接归档为"可用 TS"，必须在小自由度条件下做 DFT 频率计算，验证是否只剩单一主反应虚频。

## 证据（TS6 复测）
- Dimer max_force = 0.025 eV/Å ✅
- 能量稳定（最后4步 ±0.5 meV）✅
- SCF 正常（5-6次电子迭代/步）✅
- Dimer 投影方向正确（N*F0 < 0）✅
- **频率验证：未做 ❌ ——这是关键缺口**

## 为什么必须做频率
1. Dimer 力收敛只保证结构沿 dimer 方向走到 stationary point，不回答是否有多个虚频。
2. TS 的数学定义：Hessian 有且仅有一个负本征值——只有频率计算（Hessian 对角化）能确认。
3. 历史上多次遇到 Dimer 力收敛但频率算出 2-3 个大额外虚频的案例（如 TS3/TS5 rescue），证明力收敛不等于 saddle point。

## 频率计算的具体口径
- 复用已跑通的 FREQ_DFT 模板（PREC=Normal, EDIFF=1E-8, IBRION=5, POTIM=0.015, NSW=1）
- **只放开吸附物 + 最近 6 个 Pt**，其余 Pt 全固定（小自由度，省成本且已验证足够）
- 软小虚频（几十 cm⁻¹）暂不作为失败标准；只盯 >100 cm⁻¹ 的大额外虚频
- 验证通过标准：单一大虚频（通常 >200i cm⁻¹）+ 无 >100 cm⁻¹ 的额外虚频

## 适用边界
- 本判断适用于 MCH-Pt-Br 项目所有 MACE NEB → VASP Dimer 流程产生的 TS 候选
- 不跳过频率直接归档，不论 Dimer 跑得多"漂亮"
- 若频率有大额外虚频 → 回到 MACE 虚频方向做 ±0.20 Å 位移重新 Dimer，不反复烧同一输入
