# 统一视频交付

今后的评测报告同时交付 `research_videos/<run-or-date>/`，不再让用户查找各回合的 attempt 目录。

当前两轮实验的集中导出命令（CPU，未提交 Slurm）：

```bash
cd /mnt/beegfs/home/caijiasheng25/robopatch_defense/roboticAttack-main
/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B research_tools/export_video_gallery.py \
  --run 动作恢复诊断=research_runs/wrist_spatial_relation_rescue_v1 \
  --run 原生补丁对比=research_runs/wrist_spatial_failure_audit_v1 \
  --output research_videos/2026-10-20_spatial_relation --compilations
```

入口为 `INDEX.html`；根目录连续合集可直接播放。单段视频按类别和起点命名；比较曲线与视频放在一起。

工具仅读取显式指定的运行目录，只从最终 `COMPLETE.json` 与每起点成功收据收集文件，检查来源修订和 SHA256，不收集失败或旧 attempt。源实验保持只读，输出使用独立副本。重复运行跳过相同副本和合集；发现已有文件内容不同立即报错，不覆盖。

合集使用 imageio-ffmpeg 的无损流拼接，再用最多两解码线程验证，不用 GPU、不重新仿真。不要把格式不同的比较片段强行合并；拼接失败保留错误并先提供单段视频。`MANIFEST.json` 保存源路径、目标路径、来源修订、校验值及导出成本。

未来新运行需保持 `comparisons/<start>/COMPLETE.json`、`episodes/<start>/<condition>/COMPLETE.json`（或 `replays`）结构，收据中明确列出最终视频；新条件会自动单独分类。将导出作为评测报告的最后一步，不能把对已封存运行的展示修改混入科学代码。
