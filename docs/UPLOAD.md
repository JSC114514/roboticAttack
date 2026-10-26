# 手动检查与上传

发布目录：`/mnt/beegfs/home/caijiasheng25/roboticAttack_publish_20261023`。
远端：`https://github.com/JSC114514/roboticAttack.git`。不要在原实验工作区执行下面的git add；该工作区使用不同远端。

本次改动的精确路径、大小和哈希在[UPDATE_FILES.tsv](UPDATE_FILES.tsv)，覆盖本次新增／修改文件。全部发布文件与哈希在根目录release_checksums.json；源码原始来源见SOURCE_MANIFEST.json。

```bash
cd /mnt/beegfs/home/caijiasheng25/roboticAttack_publish_20261023
/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B scripts/release/check_release.py
/home/caijiasheng25/.conda/envs/robopatch_vla/bin/python -B -m unittest discover -s tests_release -v

git status --short
git diff --stat
git add --all -- .
git diff --cached --stat
git commit -m "Archive patch experiments and document wrist visual reliability research"
git push origin HEAD
```

本次整理不代替用户执行git add、commit或push。若远端已有他人新提交，先用git fetch origin和git log比较，不使用强制推送。

文件清单不包含自身与release_checksums.json的哈希，避免自引用；这两份文件也需随提交上传。完整发布哈希索引包含UPDATE_FILES.tsv。
