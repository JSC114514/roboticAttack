# 当前实验：腕部DINO ILPD四组对照

从这里找文件：[实验方案](PLAN.md)。所有新结果集中在本目录，旧outputs不搬动。

- 训练补丁：`training/<method>/outer_0200.png`及同名`.pt`（追加200轮，累计700轮）。
- 起点补丁：`training/<method>/outer_0000.png`（冻结UADA500）。
- 训练状态、记录与断点：`training/<method>/{status.json,training.parquet,resume.pt}`。
- GPU启动对齐：`training/<method>/RUNTIME_ALIGNMENT.json`；来源：`RUN_MANIFEST.json`和`source_snapshot.zip`。
- 日志：`logs/ilpd_train_<主作业ID>_<数组ID>.out`与`.err`。
- 预检：`reports/`；后续四suite测试：`evaluation/`，尚未提交测试。

数组0=`uada_bounded`，1=`dino_gradscale_bounded`，2=`dino_ilpd_bounded`，3=`siglip_ilpd_bounded`。

项目根目录手动运行：`bash submit_exploring_oft_dino_ilpd_refine200.sh`。仅预检：`CHECK_ONLY=1 bash submit_exploring_oft_dino_ilpd_refine200.sh`。不会自动接测试；完成后先看四张PNG。

历史位置：[总导航](../README.md)、[旧outputs导航](../../outputs/README_RESEARCH_INDEX.md)。
