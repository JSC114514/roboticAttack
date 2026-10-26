# Spatial DINO 几何机制与 RGB 可实现性小试

独立运行目录；旧 atlas、补丁、状态、轨迹和 checkpoint 均只读。这里只测干净状态的受控渲染反事实、内部干预和真实 PNG 的动作响应，不产生闭环 CASR 结论。

## 手动提交与状态

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
bash research_runs/wrist_dino_geometry_pilot_v1/run.sh --preflight
bash research_runs/wrist_dino_geometry_pilot_v1/run.sh --dry-run
bash research_runs/wrist_dino_geometry_pilot_v1/run.sh --submit
bash research_runs/wrist_dino_geometry_pilot_v1/run.sh --status --slurm
```

只有用户明确运行 `--submit` 才调用 sbatch；准备、导入、测试和 dry-run 不查询或提交调度器。一次手动提交串联准备→机制→短训练/PNG核验→统计；同一命令可续跑，拒绝重复活动任务和身份不一致的缓存。不会取消任务、重置 GPU 或自动增加候选。

GPU 阶段串行；首个 GPU 阶段等待提交时已有的 GPU 任务结束。阶段用 afterany 串联，后继 worker 在加载模型前检查前置产物；前置失败会快速退出并留成本记录，最后 CPU 阶段生成未完成说明，避免永远等待失败依赖。sbatch 返回不确定时保留 AMBIGUOUS_SBATCH_RESPONSE，禁止自动重试；先检查回执和 Slurm。

## 固定设计

- 复用 atlas v2 已对齐的 1,200 状态，选 40 条独立轨迹的一帧：每任务/每划分字典序前两条轨迹，腕部可见、无接触、未抓持，选物体距末端最近帧，平局取较早 step。Discovery20 用于补丁训练，Validation20 只做固定最终核验；这些验证状态曾用于旧研究，属于探索性验证。
- 方位：在世界桌面 XY 平面绕末端顺/逆时针 15°、30°，保留高度和三维距离。距离：水平径向靠近/远离 25%、50%，保留方位和高度。位移上限 30mm；角度通过弦长反算截断，记录实际变化，不让截断破坏距离不变条件。
- 在原生腕部相机 `_sensor` 的实际采样时刻渲染；在 `mjv_updateScene` 之后、`mjr_render` 之前临时移动目标碗的 scene geom，finally 恢复位置与 Python 钩子。不修改 qpos、模型、相机、接触或采样时钟，也不调用 forward/step 来生成反事实。柜体使用原来已实现的同一模型。
- 检查零位移 RGB（归一化 RMS≤1e-4）、原状态/本体/画面重放、采样前后状态与时钟一致；保留渲染零变化的原生动作响应作为伪影对照。所有无目标、不可见和失败记录保留。遮挡只是中心在视锥但无目标像素的代理，未检查反事实物体间的物理碰撞，不能当真实轨迹。
- block16 与原生消费 block22：原生 CF RGB、DINO-only、SigLIP-only、both、逐 token 范数匹配随机 DINO 和 identity。内部替换只改空间 token；block16 保留原 CLS/register。每帧验证 block22 双分支替换重现原生 CF；不把 block16 的空间-only 干预写成完整 RGB 等价。
- 几何读出：复用完整 Discovery600 的 4×4 空间缓存，五折按轨迹选 ridge alpha，序列化权重。只监测相对 xyz 与距离，不参与补丁损失；不重新提取全部层。
- 16 候选＝方位正/负、距离近/远 × block16/22 × seed42/43；每候选一张跨 Discovery20 共享的 50×50 腕部 RGB 补丁，固定顺序循环 50 次，共 1,000 次更新。全部候选运行，不以机制效果筛掉难目标。
- Adam lr0.002，warmup20次，然后 cosine 到1000次；目标固定为 large 条件。损失按 clean→CF token 变化能量加权匹配 DINO 空间输出，并归一化为干净输入距离1。全零目标保留为 no_target 次数；没有动作损失、读出器损失或接受/回退门槛。
- EOT 的源文件逐字复制，训练随机位置0..174，旋转±30°、剪切±0.2、20%单位变换、不缩放。补丁先按 PNG 量化，再通过显式 STE 保留梯度，随后使用原 EOT；完整图像经真实 uint8/PIL/原生 processor 前向。这样训练前向对应保存 PNG 后再做相同变换，反向才使用已验证的代理。
- 保存0/500/1000次 PNG；只核验最终1000次 PNG。Validation20 上固定位置(26,137)加四组配对 EOT，完整原生 DINO+SigLIP+动作头。共享 clean、同种子初始 PNG 和既有 Spatial UADA500 B1 对照；固定位置另分别恢复干净 DINO/SigLIP。
- 主指标为下一条待执行平移动作在目标变化方向上的投影；完整8×7输出保留。命令不是实际位移；未执行后续槽位不当成真实控制结果。区间按任务→轨迹重采样，EOT、种子和重复图像不增加独立轨迹数。

## 资源、耗时与续跑

|阶段|每worker资源|预计时间|Slurm时间上限|
|---|---|---|---|
|prepare|2 CPU / 4G / 无GPU|15–30分钟|45分钟|
|mechanism：重放渲染、特征、原生干预|4 CPU / 12G / 1×4090|0.5–1.5小时|2小时|
|pilot：16候选串行训练及最终PNG核验|4 CPU / 16G / 1×4090|1–3小时|4小时|
|analyze|2 CPU / 4G / 无GPU|15–30分钟|45分钟|

不同时申请16张GPU。12G/16G依据同类已有峰值加余量，新流程尚无GPU实测；时间是预算，不是结果。第一批状态/首候选更新吞吐和剩余时间。过程记录 RSS HWM、CUDA allocated/reserved、CPU/elapsed 和失败成本；Slurm sampled MaxRSS 单独查看。

逐状态提交渲染、特征、机制和评测分片；每100次更新保存包含 Adam 状态的 LAST.pt，中断最多重做99次更新。失败尝试日志保留。源代码、外部依赖、使用的上游文件、模型小文件哈希及大权重 size/mtime 进入身份；已提交/已产出后代码变化应建立独立版本目录。无法捕获 SIGKILL 的退出成本以 STARTED 回执、最近进度和 sacct 为准，不伪造最终峰值。

## 输出和检查

- `reports/PREFLIGHT.json`：40帧、16候选、输入来源及身份；不表示 GPU 检查通过。
- `reports/GPU_RGB_INTERFACE.json`：提交后的原生像素一致性、消费层、梯度检查。
- `data/renders` / `data/features` / `data/mechanism`：同状态关联的分片与完整性回执。
- `training/<candidate>/step_1000.png`：最终部署产物；LAST.pt 仅用于本地续跑。
- `results/RESULTS.md`、`MECHANISM_SUMMARY.csv`、`MECHANISM_CONTRASTS.csv`、`PNG_SUMMARY.csv`：解释、分层结果、配对区间、失败案例/直观图。
- `reports/COSTS`、`reports/FAILURES`、`results/COST_SUMMARY.json`：失败也计成本。

本地检查命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m pytest \
  -q research_runs/wrist_dino_geometry_pilot_v1/tests
```

本地测试使用合成数据和模拟调度器，不加载VLA、不创建EGL、不提交任务。真实相机、原生checkpoint、CUDA梯度和吞吐由手动提交后的前置检查核实；这些尚未运行前，不声称补丁或机制有效。
