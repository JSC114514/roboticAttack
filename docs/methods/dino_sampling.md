# DINO训练观测采样与受扰轨迹回放

## 基础采样

VLAAttacker/white_patch/UADA_OFT_DINO_SAMPLING.py及对应wrapper；scripts/plan_dino_guided_placement.py负责历史离线选点。DINO局部原型给训练画面打分，同任务内加权采样；完整模型仍按UADA式动作目标训练RGB补丁，不增加DINO特征损失。

## 回放

scripts/collect_exploring_oft_dino_replay.py采集补丁作用下的轨迹，build_exploring_oft_dino_replay_bank.py构建bank，dino_replay_common.py定义采样。训练器UADA_OFT_DINO_REPLAY.py，入口UADA_OFT_DINO_REPLAY_wrapper.py。

对同一观测比较clean动作、补丁动作及恢复干净腕部DINO后的动作，按恢复撤销多少错误动作推动来评分。比较clean-only、uniform replay、DINO replay和shuffled replay，分开识别状态覆盖与DINO评分的作用。

## 动态采样版本

UADA_OFT_DINO_SAMPLING_CLEAR.py、oft_dino_sampling_clear_objectives.py、scripts/dino_sampling_clear_common.py及对应wrapper。四组dynamic_soft/dynamic_hard/stable_hard/shuffled_hard使用各自评分与外观约束；原配置object_dino_sampling_clear2000_v1.json。训练从随机初始化开始，不能与旧补丁继续训练版混为一组预算。

```bash
python VLAAttacker/UADA_OFT_DINO_SAMPLING_wrapper.py --help
python VLAAttacker/UADA_OFT_DINO_REPLAY_wrapper.py --help
python VLAAttacker/UADA_OFT_DINO_SAMPLING_CLEAR_wrapper.py --help
```

以上为参数入口。正式运行要求对应bank、模型与frozen_inputs匹配，部分runner要求用户已提交的Slurm环境。不要把历史恢复脚本当从零训练入口。动态回放与初版采样的时间关系及失败尝试见[研究脉络](../RESEARCH_STORY.md)。

本发布未找到并核验PPT中“动态回放组合1000/2000”表格的完整统计资产，故没有把截图数字重新抄成发布结果。方法源码与已核验的其他结果分别归档。
