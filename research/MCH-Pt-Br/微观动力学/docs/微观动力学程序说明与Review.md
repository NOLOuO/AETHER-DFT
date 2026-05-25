# MCH/Pt 微观动力学程序说明与 Review

> 路径: `/home/szhang/research/MCH-Pt-Br/微观动力学/`  
> 日期: 2026-05-22  
> 当前结论: 已把 review 中发现的主要问题直接修进程序; 当前版本可作为 **DFT 自由能表进入 MKM 的工作骨架**。示例能量仍是占位数, 不能解释物理趋势。

---

## 1. 程序现在做什么

主程序:

```text
mch_microkinetics.py
```

输入:

```text
inputs/species_data.csv      # 物种、占位数、气相分子质量、是否属于 C7 pool
inputs/example_network.csv   # elementary steps、能垒、反应自由能、吸附参数
```

输出:

```text
*.coverages.csv        # 稳态覆盖度
*.rates.csv            # 每步 kf/kr/正向/反向/净速率
*.drc.csv              # DRC
*.reaction_orders.csv  # reaction order
*.tof_vs_T.csv         # 温度扫描 + apparent Ea
```

定位:

```text
DFT IS/TS/FS 自由能
→ CSV 反应网络
→ mean-field ODE 稳态
→ 覆盖度 / 速率 / DRC / reaction order / apparent Ea
```

---

## 2. 反应网络

当前 `example_network.csv` 是最小顺序脱氢网络:

```text
C7H14_g + 4* <-> C7H14*
C7H14* + *  <-> C7H13* + H*
C7H13* + *  <-> C7H12* + H*
C7H12* + *  <-> C7H11* + H*
C7H11* + *  <-> C7H10* + H*
C7H10* + *  <-> C7H9*  + H*
C7H9*  + *  <-> C7H8*  + H*
H2_g + 2*   <-> 2H*
C7H8_g + 4* <-> C7H8*
```

当前能量仍是 placeholder。真实使用时要替换:

```text
Gaf_eV / Gar_eV / dG_eV
```

---

## 3. 速率常数

### 3.1 表面反应

表面反应用 TST:

```text
k = kBT/h * exp(-G_act / kBT)
```

如果给了 `Gar_eV`, 反向直接用反向自由能垒。

如果没有 `Gar_eV` 但给了 `dG_eV`, 用热力学一致性:

```text
kr = kf * exp(dG / kBT)
```

### 3.2 吸附/脱附

吸附用 kinetic gas theory:

```text
k_ads = S * p * A_site / sqrt(2*pi*m*kB*T)
```

默认:

```text
A_site = 10 Å²
sticking = 1
```

吸附自由能解释方式:

```bash
--adsorption-dg-state standard   # 默认, dG_eV 按 1 bar 标准态理解
--adsorption-dg-state current    # dG_eV 已经包含当前压力化学势
```

---

## 4. 覆盖度和 C7 pool cap

现在已经修正为:

```text
site_size 与 carbon_pool_cap 分开
```

`species_data.csv` 中 C7* 设置为:

```text
site_size = 4
carbon_pool = true
```

默认 C7 pool cap:

```bash
--carbon-pool-cap 0.11
```

空位:

```text
theta_* = 1 - sum(site_size_i * theta_i)
```

C7 pool:

```text
theta_C7_pool = sum(theta_i for carbon_pool species)
```

重要修正:

```text
cap 只限制“增加 C7 pool 覆盖度”的方向;
不再限制 C7* 内部脱氢/氢化;
不再限制 C7* 脱附/减少 C7 pool 的方向。
```

也就是按每步反应的:

```text
ΔN_C7_pool = C7_pool(products) - C7_pool(reactants)
```

判断:

```text
ΔN_C7_pool > 0: forward 乘 carbon availability factor
ΔN_C7_pool < 0: reverse 乘 carbon availability factor
ΔN_C7_pool = 0: 两个方向都不乘
```

这修掉了之前 review 中的 HIGH 问题。

---

## 5. DRC / reaction order / apparent Ea

### 5.1 DRC

命令:

```bash
--drc --drc-target net_dehydrogenation
```

默认 perturbation:

```text
10%
```

保持 equilibrium constant 不变: 同时缩放 `kf` 和 `kr`。

### 5.2 Reaction order

命令:

```bash
--reaction-orders
```

默认压力扰动:

```text
1.12 倍
```

会分别扰动:

```text
C7H14_g
H2_g
C7H8_g
```

### 5.3 Apparent Ea

温度扫描时自动输出:

```text
Eapp_<target>_eV
```

已修正: 如果目标速率在相邻温度点变号, 对应点 `Eapp = nan`, 避免用 `abs(rate)` 掩盖反应方向反转。

---

## 6. Schema validation

现在程序启动时会检查:

- step type 是否合法
- adsorption step 是否有 gas species
- adsorption step 是否有 `dG_eV` 或 `Gar_eV`
- adsorption step 是否能找到气相分子质量
- surface step 是否有 `Gaf_eV`
- surface step 不允许气相物种直接参与
- surface reversible step 应提供 `Gar_eV` 或 `dG_eV`
- species 是否在 `species_data.csv` 声明
- `site_size` 是否非负

这样后面填真实 DFT 表时, 不容易静默跑错。

---

## 7. 推荐运行命令

```bash
cd /home/szhang/research/MCH-Pt-Br/微观动力学

python ./mch_microkinetics.py \
  --network inputs/example_network.csv \
  --species inputs/species_data.csv \
  --T 623 \
  --p-mch 1.0 \
  --p-h2 0.15 \
  --p-tol 0.10 \
  --out-prefix outputs/smoke_placeholder_623K \
  --drc \
  --reaction-orders \
  --T-range 573:673:50
```

---

## 8. 已验证结果

执行过:

```bash
python -m py_compile mch_microkinetics.py
```

通过。

执行过完整 smoke test:

```bash
python ./mch_microkinetics.py \
  --network inputs/example_network.csv \
  --species inputs/species_data.csv \
  --T 623 \
  --p-mch 1.0 \
  --p-h2 0.15 \
  --p-tol 0.10 \
  --out-prefix outputs/smoke_placeholder_623K \
  --drc \
  --reaction-orders \
  --T-range 573:673:50
```

关键输出:

```text
steady_converged = True
max_abs_dtheta_dt = 1.943e-09 ML/s
theta_carbon_pool = 1.890261805337e-01 ML
carbon_pool_factor = 7.579020368019e-04
net_dehydrogenation = -2.315590417355e+01 s^-1 site^-1
deh_flux_span = 1.847411112976e-13 s^-1 site^-1
```

DRC 文件已正常输出, 不再全是 nan:

```text
R_ads_TOL  drc ~ 0.837
R_ads_MCH  drc ~ 0.078
R3/R4      drc ~ 0.019
```

注意: 这些数值来自 placeholder 能量, 只能说明程序链路通了, 不能当物理结论。

---

# 9. Review after fix

## 9.1 总体结论

```text
RECOMMENDATION: COMMENT / usable as skeleton
```

现在程序可以作为后续填 DFT 数据的工作骨架。之前的核心 HIGH 问题已经修掉:

```text
C7 pool cap 不再冻结所有 C7* 内部反应和脱附。
```

---

## 9.2 已修复的问题

### Fixed HIGH 1 — C7 cap 错误作用到所有 C7 相关反应

原问题:

```text
只要 step 涉及 C7*, 正反向都乘 cap factor。
```

已改为:

```text
只限制增加 C7 pool 覆盖度的方向。
```

### Fixed HIGH 2 — 默认 cap 下 DRC / reaction order 全 nan

原问题来自速率被错误 cap 压到接近 0。

现在 smoke test 中 DRC 正常输出。

### Fixed MEDIUM — apparent Ea 变号处理

现在如果 rate sign change, `Eapp` 输出 `nan`, 并有:

```text
rate_sign_change_<target>
```

### Fixed MEDIUM — 增加 schema validation

现在输入格式错误会直接报错, 不会悄悄跑。

---

## 9.3 仍需注意的问题

### MEDIUM 1 — 当前 C7 cap 用 logistic availability, 不是文献 sigmoid adsorption-energy correction

现在的 cap 是数值稳定的 availability gate:

```text
1 / (1 + exp(10 * (theta_C7_pool/cap - 1)))
```

它能防止无限吸附, 但还不是 Pt 文献里的 coverage-dependent adsorption energy sigmoid。

如果要更贴文献, 后续应该加:

```text
coverage-dependent adsorption energy correction
```

尤其是 toluene / cyclic carbon adsorption energy 随覆盖度变化。

### MEDIUM 2 — `net_dehydrogenation` 仍是 deH flux 平均

现在加了:

```text
deh_flux_min
deh_flux_max
deh_flux_span
```

如果 `deh_flux_span` 很小, 平均值可用作链路 flux。正式写文章时建议优先报告:

```text
toluene_desorption
MCH_consumption
```

具体取哪个, 取决于网络是否包含副反应和出口定义。

### MEDIUM 3 — 当前示例能量是假数据

所有输出数值只验证程序, 不代表 Pt 或 Br-Pt 的真实动力学。

### LOW — 还没有 pytest 自动测试

目前验证靠 smoke test。后续可以加一个最小测试脚本, 固定检查输出文件和列名。

---

## 10. 下一步

现在应该做的不是再改框架, 而是:

1. 整理 TS1–TS6 / IS / FS 的 DFT 自由能。
2. 替换 `example_network.csv` 的 placeholder 能量。
3. 先跑 pure Pt:

```bash
--T-range 573:923:25 --drc --reaction-orders
```

4. 检查:

```text
steady_converged
deh_flux_span
DRC 是否合理
reaction order 是否离谱
```

5. pure Pt 合理后复制一份网络给 Br-Pt。

---

## 11. 当前 verdict

```text
Files reviewed: 3
- mch_microkinetics.py
- species_data.csv
- example_network.csv

CRITICAL: 0
HIGH: 0
MEDIUM: 3
LOW: 1
Architectural status: WATCH
Final recommendation: COMMENT
```

一句话:

> 现在程序骨架可以用了; 真正限制下一步的是 DFT 自由能输入表, 不是 MKM 框架本身。

---

## 12. 二次 Review 更新: 还剩的问题和已补内容

### 12.1 本轮已补

本轮又补了两个实用项:

1. `*.summary.csv`
   - 汇总温度、压力、是否收敛、最大残差、空位、C7 pool、cap violation、主要 observable。
   - 目的: 后面批量跑 Pt / Br-Pt 时不用逐个打开 coverages/rates。

2. `carbon_pool_violation`
   - 单点 summary 和 T-scan 都会输出。
   - 目的: 当前 cap 是数值稳定的 soft cap, 不是硬约束; 如果超过 0.11 ML, 必须显式看到。

本轮验证命令:

```bash
python ./mch_microkinetics.py \
  --network inputs/example_network.csv \
  --species inputs/species_data.csv \
  --T 623 \
  --p-mch 1.0 \
  --p-h2 0.15 \
  --p-tol 0.10 \
  --out-prefix review6_full \
  --drc \
  --reaction-orders \
  --T-range 573:673:50
```

输出新增:

```text
outputs/smoke_placeholder_623K.summary.csv
```

### 12.2 当前还剩的真实问题

#### WATCH 1 — C7 cap 现在是 soft cap, 不是严格硬 cap

当前为了避免 ODE solver 在硬截断点失稳, 使用 smooth approximation。结果是示例中:

```text
theta_carbon_pool = 0.1244 ML
carbon_pool_cap   = 0.1100 ML
violation         = 0.0144 ML
```

这不是程序崩了, 但说明 cap 不是严格数学约束。正式数据里要盯 `carbon_pool_violation`。如果真实 DFT 输入后 violation 仍明显, 需要进一步改成更物理的 coverage-dependent adsorption free energy, 而不是继续调 cap 函数。

#### WATCH 2 — 当前 TOF target 仍需要按物理方向选择

默认 target 是:

```text
net_dehydrogenation
```

它适合检查链路 flux, 但正式报告应优先看:

```text
toluene_desorption
mch_consumption
```

尤其当反应方向反转或有副反应时, 不应只看 `net_dehydrogenation`。

#### WATCH 3 — T-scan 个别点可能未达默认严格残差

示例中 573 K / 673 K 的残差约 `1.5–1.7e-8`, 略高于默认 `1e-8`, 所以 `converged=0`。这不影响发现程序链路, 但正式跑时应检查:

```text
converged
max_abs_dtheta_dt
deh_flux_span
```

必要时用:

```bash
--steady-tol 1e-7
```

或延长积分。

### 12.3 现在不建议马上加的内容

暂时不加这些, 因为会把输入数据需求放大:

- hydrodemethylation side path
- 115-step 全网络
- BEP 自动补 barrier
- CSTR 10% conversion
- BEEF ensemble uncertainty

### 12.4 下一步真正应该加的内容

真正下一步不是继续堆功能, 而是加真实输入表:

```text
pure_Pt_network.csv
Br_Pt_network.csv
```

把 TS1–TS6 的 DFT 自由能填进 `Gaf_eV/Gar_eV/dG_eV`。等真实数据进入后, 再判断是否必须加 coverage-dependent adsorption energy / lateral interaction。
