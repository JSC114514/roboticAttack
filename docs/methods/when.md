# WHEN / UPA-RFAS

核心代码为VLAAttacker/white_patch/feature_space.py；VLAAttacker/when/包含特征/对比损失、注意力、语义、变换和资产清单工具。训练入口scripts/when/train_sim_patch.py，源模型评测eval_source_openvla.py，OFT迁移评测eval_oft.py。

训练组合feature deviation、repulsive contrastive、PAD、PSM及RAUP背景扰动；只更新patch及被当前内层优化的扰动，VLA保持冻结。patch与mask使用同一几何变换。

```bash
python scripts/when/train_sim_patch.py --help
python scripts/when/eval_oft.py --help
python scripts/when/check_requirements.py --json
```

configs/when/train/保留原配置。当前默认UPA_RFAS_REFERENCE指向仓库内VLAAttacker；official_aligned配置显式指向原外部官方clean副本。复现时选择其中一条并核对来源，不能不加说明地混用。

原始OpenVLA训练和OFT上的PNG评测是不同模型阶段；WHEN迁移性能不代表在OFT上重新训练得到的性能。视频采用前视/腕部独立文件，导出器在research_tools/export_singleview_videos.py。
