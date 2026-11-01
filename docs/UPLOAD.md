# 独立分支上传

发布目录：`/mnt/beegfs/home/caijiasheng25/roboticAttack_publish_20261023`。
分支：`research/wrist-dino-study`。远端：`https://github.com/JSC114514/roboticAttack.git`。

原main保持原样。本次只上传此分支，不合并main，不修改默认分支。原始实验工作区不执行git add。

```bash
cd /mnt/beegfs/home/caijiasheng25/roboticAttack_publish_20261023

git branch --show-current
/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B scripts/release/check_release.py
/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m unittest discover -s tests_release -v

git add --all -- .
git diff --cached --stat
git commit -m "Publish wrist DINO study, ablations and cross-model results"
git push -u origin research/wrist-dino-study
```

待上传文件清单见[UPDATE_FILES.tsv](UPDATE_FILES.tsv)，新增研究内容见[导出统计](../studies/wrist_dino/provenance/EXPORT_SUMMARY.json)。文件清单和发布总哈希排除自己的递归引用；原始与注释副本身份见SOURCE_MANIFEST.json。

本次不代替用户执行git add、commit、push。研究代码与结果已准备、检查后再上传。无视频、无模型权重、无优化器、无候选缓存、无本次集群排队脚本。
