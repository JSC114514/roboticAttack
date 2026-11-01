# EOT与完整梯度预检

PASS。四套件原生动作误差≤1e-5；Exploring→OFT源EOT像素一致；腕部DINO、SigLIP和RGB梯度有限且非零。语言重计算与直接梯度已配对核对。

原scheduler从lr0开始，第一outer的50次更新不会改变像素；本轮保留此源码行为，不改学习率语义。详细日志见reports/P03_VALIDATION.json。
