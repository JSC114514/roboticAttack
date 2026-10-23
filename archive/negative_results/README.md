# 失败路线与部分成功路线

这里保留方法为何转向的依据。失败按“表示/离线改善但闭环未增强”“源域有效但迁移弱”“工程刺激未完成”分别记录。

| 路线 | 做法 | 观察与决定 |
|---|---|---|
| DINO-Norm/MGWP | 增大表征变化，加入range/lateral几何方向 | 部分闭环有效，未形成稳定机制额外优势 |
| RAMP | 多时刻累计动作漂移 | 离线漂移提高；相对Continue闭环净增小 |
| Continue/Critical | 阶段条件方向与继续训练 | Object有效，Spatial/Goal迁移弱 |
| CID-CF | 跨上下文一致DINO因果方向 | 代理一致性改善，闭环无独立优势且源域退化 |
| DINO-CF/restore/WorstChunk | 限定DINO路径或要求恢复 | 同源UADA比较未稳定增强 |
| Object-Decoy | 局部特征向目标物体原型靠拢 | Object两档24%/18%，同场UADA32.1% |
| 局部关系与图像位移 | 用定位读数和物体移动检查几何 | 动态读数不稳定；部分刺激未进模型测试 |
| Jacobian | 敏感token沿动作梯度方向变化 | Object7.1%/16.0%，同场UADA32.1% |
| 对应/ILA/交替/ILPD | 调整表示方向与训练梯度 | 主要套件无稳定增益，转向受扰回放 |
| 完整几何特征匹配/教师/XYZ读出 | 将明确几何目标压进共享PNG | 多数未稳定打通表示→动作→闭环 |
| 几何目标联合选择 | 策略阻碍程度×补丁可实现性 | 三种匹配Goal0/39，动作基线35/39 |

早期核心片段在early_sources，完整历史叙述见[研究脉络](../../docs/RESEARCH_STORY.md)与[原始主方案](../research_records/OFT_DINO_MECHANISM_CLOSED_LOOP_MASTER_PLAN.md)。后续源码在VLAAttacker/white_patch及精选research_runs。

初版采样与Jacobian等在时间上交错，不能把所有负结果写成采样之前已经获知。归档不附失败日志、优化器或重复工程版本的输出。
