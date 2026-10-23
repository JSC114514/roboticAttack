# Exploring基线与OFT适配

原始实现：VLAAttacker/white_patch/UADA.py、TMA.py、UPA.py；对应wrapper在VLAAttacker/。共享RGB变换在appply_random_transform.py，RLDS加载在openvla_dataloader.py与RLDSBatchTransform.py。

UADA按参考动作构造远端点目标，TMA指定动作目标，UPA针对位置相关动作。原始模型使用action token。OFT适配通过连续动作头预测8×7命令，需分别注明参考来自专家还是clean策略，不能把clean参考适配称为原始专家参考UADA。

OFT实现入口：VLAAttacker/UADA_OFT_DINO_wrapper.py；基类：white_patch/UADA_OFT_DINO.py；原始优化原语副本：UADA_OFT_SOURCE_COPY.py。方法exploring_oft_uada关闭额外DINO目标。双视角接口位于VLAAttacker/multiview/。

先查看参数，再传入已准备的配置：

```bash
python VLAAttacker/UADA_wrapper.py --help
python VLAAttacker/UADA_OFT_DINO_wrapper.py --help
```

脚本scripts/run_UADA.sh等保留为原入口，里面的原机器路径需按本机配置。标准原流程是准备RLDS数据→训练PNG→固定checkpoint与同初态闭环评测→统计clean-success条件CASR。该发布没有执行任何训练。
