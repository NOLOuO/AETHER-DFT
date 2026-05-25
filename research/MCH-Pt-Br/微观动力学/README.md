# MCH-Pt-Br 微观动力学目录

## 目录结构

```text
微观动力学/
├── mch_microkinetics.py          # 主程序
├── inputs/
│   ├── example_network.csv       # placeholder 反应网络模板
│   └── species_data.csv          # 物种/占位/气相质量模板
├── outputs/
│   └── smoke_placeholder_623K.*  # 一套可复现实例输出, 只用于验证程序链路
└── docs/
    └── 微观动力学程序说明与Review.md
```

## 当前状态

- 程序链路已跑通: 稳态覆盖度、rates、DRC、reaction order、T-scan、summary。
- `inputs/example_network.csv` 里的能量是 placeholder, **不能作为物理结论**。
- 做 Br / no-Br 的 TOF 对比时, 程序本身不需要单独一列“每个中间体自由能”,但输入到网络里的每一步 `Gaf_eV / Gar_eV / dG_eV` 仍然通常要由这些中间体的 DFT 自由能拼出来。
- 下一步应复制模板生成真实输入:

```text
inputs/pure_Pt_network.csv
inputs/Br_Pt_network.csv
```

然后把 DFT 的 `Gaf_eV / Gar_eV / dG_eV` 填进去。

## 推荐 smoke test

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

## 关键检查项

正式跑真实数据时先看:

```text
*.summary.csv
*.tof_vs_T.csv
```

重点检查:

```text
converged
max_abs_dtheta_dt
deh_flux_span
carbon_pool_violation
net_dehydrogenation / toluene_desorption / mch_consumption
```
