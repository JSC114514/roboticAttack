# Spatial DINO 信息图谱

实现范围与标签限制见 [PLAN.md](PLAN.md)。旧实验、原图和旧标签保持只读。没有训练或部署补丁。

本目录是 v1 作业 141878/141879 失败后的采样时序修复。v1 的源代码、提交凭证和失败成本原样保留，修复来源见 reports/FIX_PROVENANCE.json。读出目标、数据划分、层/表示、对照和资源不变。

标签在**原相机 Observable 的 sensor 返回 RGB 时**采集；临时包装仅用于本进程，不强制更新观测、不推进物理、不重新设置初态或修改共享库。原缓存 RGB、原动作后的状态、本体状态仍按原标准检查。采样时刻和控制步结束时刻分别记录，不再假定二者一致。

仅复用此前运行库的只读模型加载与 EGL 生命周期代码（原件哈希见 COPIED_SOURCES.json）；旧补丁、表示干预和 256 像素环境例程未带入本流程。

## 手动入口

在项目目录执行：

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_information_atlas_v2/run.sh --preflight
bash research_runs/wrist_dino_information_atlas_v2/run.sh --dry-run
bash research_runs/wrist_dino_information_atlas_v2/run.sh --submit
bash research_runs/wrist_dino_information_atlas_v2/run.sh --status --slurm
```

只有明确执行 `--submit` 才调用 Slurm。入口创建日志目录、冻结代码/输入/模型身份、拒绝重复活动链；一项 GPU 作业后接一项 CPU 作业。存在用户既有 GPU 作业时用 afterany 依赖串行等待，不取消它们。CPU 同样用 afterany，若 GPU 失败则快速报缺片，避免依赖永久悬挂。提交响应不明确时不自动重试。

GPU：4090×1、4 CPU、12G、2h；CPU：无 GPU、2 CPU、4G、4h。v1 失败尝试实测进程 RSS HWM 5.71 GiB、CUDA allocated 峰值 14.64 GiB，未发生 OOM，因此不增加资源。v2 全流程耗时尚未实测。CPU 全量对照可能较慢，时间超限后可手动重复相同提交命令，已封存的任务/读出面板会复用。不会自动扩展资源。

同一 GPU 作业内先完成 240 轨迹、1,200 状态的标签对齐，全部通过后才加载 VLA 提取特征。失败时无需先加载大模型；标签按任务独立封存，后续特征失败可复用标签。特征目录中的标签使用同文件系统硬链接，不额外复制大数组，两个阶段都校验哈希。

`--status` 不访问调度器；`--status --slurm` 额外只读查询已记录 job 的 sacct。reports/PROGRESS.json 是最新进度，reports/COSTS 保存每次尝试的进程 RSS、CUDA 峰值和耗时，reports/FAILURES 与 data/attempts 保留失败成本。Slurm 采样 MaxRSS 不是进程 RSS HWM 或显存。

## 部件核对

完成 capture 后，CPU 阶段自动生成可独立打开的 `parts/review.html`（嵌入预选 80 图）。逐图画各类多边形，或明确标记不可见/无法辨认，勾选已核对，再下载 JSON。页面会在浏览器本地保存草稿。

```bash
bash research_runs/wrist_dino_information_atlas_v2/run.sh --import-parts /absolute/path/parts_annotations.json
bash research_runs/wrist_dino_information_atlas_v2/run.sh --submit
```

第二次仅补 CPU 部件读出和报告，不再次提取特征。未经核对时状态为 `CORE_COMPLETE_PARTS_PENDING`，不是全计划完成。空标签不当背景，不虚构部件分数。

## 产物

- `reports/PREFLIGHT.json`：数据/动作/初态哈希、240 轨迹、80 图选择、代码和模型身份；仅 CPU 文件核验不代表 GPU 重放成功。
- `data/task_XX/`：每任务 120 状态的 RECORDS、SCHEMA、分片特征和新标签；哈希封存后原子发布。先检查 10 个任务齐全再汇总。
- `data/labels/task_XX/`：先于模型加载封存的标签及采样时刻记录；`camera_sample_timing` 包含采样时间、控制步结束时间、缓存年龄、相机与本体状态一致性。
- `readouts/global|local|parts/`：逐面板预测、五折超参、留一任务和轨迹置乱控制、完成凭证。
- `results/`：INVENTORY.json、INFORMATION_ATLAS.csv、VIEW_CONTRASTS.csv、LOCAL_READOUTS.csv、PART_READOUTS.csv、RESULTS.md、REPORT.html、figures。

信息读出用整轨迹五折；重复帧与像素不独立采样做置信区间。主读出只接收特征；独立元数据基线不会被拼接进去。对齐失败保留整个尝试及错误信息，不剔除柜体任务凑结果。

局部类别、深度、人工部件也提供独立坐标/多数类、留一任务及 5 次整轨迹置乱控制。固定 MLP 对消费输出用 3 种子检查真实标签；置乱控制用第一个固定种子配合同样 5 次置乱。没有根据验证分数挑层、追加容量或筛掉难样本。

## CPU 检查

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m pytest -q -p no:cacheprovider research_runs/wrist_dino_information_atlas_v2/tests
```

测试不会申请 GPU、调用真实 Slurm 或修改旧数据。GPU/EGL/原生 hook/逐动作画面对齐必须在用户提交后验证。

时序回归测试直接使用已安装 robosuite 的 Observable 调度实现，复现“控制步末画面已经不同、缓存仍是较早画面”的情况；物理和图像为 CPU 替身。这验证调度修复逻辑，不冒充真实 MuJoCo/GPU 验证。
