# ETCE Phase 2B 本地运行

把两个 Python 文件放到项目的 `scripts/` 目录：

```text
ETCE/
├─ scripts/
│  ├─ 06_run_phase2a.py
│  ├─ 07_validate_phase2a.py
│  ├─ 08_run_phase2b.py
│  └─ 09_validate_phase2b.py
├─ src/
│  └─ etce/
│     ├─ phase2/
│     └─ utils/
├─ results/
│  └─ phase2/
│     └─ configs/
│        └─ best_configs.json
└─ data/
```

Phase 2B 会严格复用 Phase 2A 已选中的超参数，不重新调参。

正式随机种子：

```text
13, 21, 42, 73, 101
```

## 运行

先运行正式多 seed：

```bash
python scripts/08_run_phase2b.py
```

完成后运行独立验证：

```bash
python scripts/09_validate_phase2b.py
```

## 主要输出

```text
results/phase2b/
├─ PHASE2B_REPORT.md
├─ metrics/
│  ├─ formal_metrics_per_seed.csv
│  └─ formal_metrics_summary.csv
├─ predictions/
│  ├─ seed_13/
│  ├─ seed_21/
│  ├─ seed_42/
│  ├─ seed_73/
│  └─ seed_101/
├─ checkpoints/
│  ├─ seed_13/
│  ├─ seed_21/
│  ├─ seed_42/
│  ├─ seed_73/
│  └─ seed_101/
├─ configs/
│  └─ frozen_phase2a_configs_used.json
└─ logs/
   ├─ training_log_phase2b.json
   ├─ test_access_log_phase2b.json
   └─ validation_phase2b.json
```

## 注意

Phase 2B 会执行：

- 5 seeds × 4 models × 4 prefixes = 80 次正式训练；
- 每次都保存 checkpoint；
- test 对每个 `(seed, model, prefix)` 只评估一次；
- 生成 test/validation 的逐样本预测；
- 自动汇总 mean ± std；
- 自动计算 95% Student-t CI。

不要删除已经通过验证的 `results/phase2/`，Phase 2B 会读取其中的 `best_configs.json`。
