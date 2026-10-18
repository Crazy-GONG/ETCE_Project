# ETCE Phase 4B：服务器运行指南

Phase 4B 使用冻结的 Phase 4A.1 协议，在 validation split 上建立扩大后的 calibration rule bank。它不选择 lambda、不执行 BPA/D-S fusion，也不读取 test case。

## 服务器兼容约定

从脚本 22/23 开始，运行脚本遵循以下约定：

- 项目根目录默认由脚本位置自动确定，不依赖 `D:\ETCE`。
- 也可以通过环境变量 `ETCE_ROOT` 指定项目根目录。
- Linux 服务器使用系统虚拟环境；不要上传 Windows 的 `.deps/` 或 `.venv/`。即使服务器目录意外存在 `.deps/`，脚本也不会在 Linux 中加载它。
- 所有正式输入和输出都使用项目根目录下的相对路径。
- 模型推理协议固定为 CPU，迁移服务器不会改成 GPU 推理。
- target cache 使用原子写入，作业被调度器终止后可以安全续跑。
- 已存在的 cache 只有在 JSON 完整且 target 身份一致时才会复用。
- 运行环境和依赖版本写入 `runtime_phase4b.json`。

## 必须上传的内容

推荐上传完整工程，但至少必须包含：

```text
ETCE/
├── data/raw/BPI Challenge 2017.xes.gz
├── requirements-server.txt
├── src/
├── scripts/
│   ├── 22_phase4b_validation_rule_bank.py
│   ├── 23_validate_phase4b.py
│   └── run_phase4b_server.sh
└── results/
    ├── phase1_2/
    ├── phase2b/
    ├── phase3/
    └── phase4/
        ├── phase4a/
        └── phase4a1/
```

脚本启动时会检查 Phase 2B、Phase 3B、Phase 4A.1 的 validation 状态。缺少输入或上游状态不是 `passed` 时会立即停止。

## 创建服务器环境

建议使用 Python 3.12：

```bash
cd /path/to/ETCE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

依赖版本冻结为：

```text
joblib        1.5.3
numpy         2.5.2
pandas        3.0.5
scipy         1.18.0
scikit-learn  1.9.0
torch         2.8.0
xgboost       3.4.1
```

## 运行

推荐使用统一入口，它会顺序执行构建和验证并保存日志：

```bash
cd /path/to/ETCE
source .venv/bin/activate
bash scripts/run_phase4b_server.sh
```

后台运行：

```bash
nohup bash scripts/run_phase4b_server.sh > phase4b_server_console.log 2>&1 &
echo $! > phase4b_server.pid
```

查看进度：

```bash
tail -f results/phase4/server_logs/phase4b_build.log
```

也可以手动分两步运行：

```bash
export ETCE_ROOT=/path/to/ETCE
python scripts/22_phase4b_validation_rule_bank.py
python scripts/23_validate_phase4b.py
```

必须等待 22 正常结束后再启动 23。不要在多台机器上同时写同一个 `phase4b_validation_bank` 目录。

## 断点续跑

脚本 22 按 target 保存：

```text
results/phase4/phase4b_validation_bank/target_cache/
```

作业中断后重新执行同一命令即可。完整 cache 会跳过，损坏或不完整 cache 会重新计算。不要手工拼接来自不同代码版本或不同上游 checkpoint 集合的 cache。

## 输出

```text
results/phase4/phase4b_validation_bank/
├── PHASE4B_REPORT.md
├── calibration_case_selection.json
├── phase4b_run_config.json
├── runtime_phase4b.json
├── neighborhood_manifest.csv
├── surrogate_metrics_calibration.csv
├── surrogate_aggregate_calibration.csv
├── canonical_rules_calibration.jsonl
├── rule_support_summary.json
├── validation_phase4b.json
└── target_cache/
```

只有 `validation_phase4b.json` 的 `status` 为 `passed` 后，才能把本阶段交给后续 reliability/BPA calibration。

## 下载回本地

验证通过后，应整体下载：

```text
results/phase4/phase4b_validation_bank/
```

不要只下载报告、CSV 或部分 cache。后续阶段必须使用同一套已验证输出。
