# 方法来源与协议适配审计

本轮为联合OFT统一协议移植评估，所有新结果标ADAPTED_RE_EVALUATION，不声称原论文token／专家参考的精确复现。包内说明作为待审核协议材料；本地源码和用户直接指令决定实际执行。

| ID | 源码／数学目标 | 审核结果 |
|---|---|---|
| B01–B04 | UADA_OFT_DINO.py：MSE(a,opposite(clean))，仅改view或mask | 统一实现；原transform与SOURCE_COPY优化器直接复用 |
| B05 | b12_diag500_v1/config.json tma_normalized_target=[0,0,0]及objectives.py | XYZ固定零目标，与B03仅target不同 |
| O01 | UADA_OFT_DINO.py full_worst_chunk：8槽误差max | 仅mean→max，去掉旧手工梯度组合，标移植评估 |
| O02 | oft_ramp_recovery_aware/scripts/train_ramp.py | SKIPPED_PROTOCOL：原目标需6时刻序列、累计方向和梯度投影；扁平1920池单观测每outer下无法仅改时间聚合保持原问题 |
| O03 | UADA_OFT_FEATURE.py feature_disruption_loss=mean cosine−1 | 单scalar Laction+λLaux；λ用训练池24固定状态的初始loss中位尺度比预先冻结，不用梯度比例或验证反馈 |
| O04 | 几何辅助目标需要同状态物体位置／反事实方向 | SKIPPED_PROTOCOL：本轮RLDS未含物体位姿／sim_state，旧缓存属于不同轨迹；不能把旧Goal/Spatial方向冒作联合专家池标签 |
| O05 | b12_diag500_v1/code/new_objectives.py affinity_objective | 移植原函数topk8，无语义标签；用同源EOT的zero/one补丁观测支持区域、原crop后16×16池化；恢复RNG后正式前向，不改变变换。仅新增scalar辅助项，λ同训练池固定loss尺度校准 |
| O06 | UADA_OFT_OBJECT_DECOY.py object_decoy_loss | SKIPPED_PROTOCOL：需要target_object和matched_background掩码；新专家池无对应掩码，旧Object池300观测不能混入 |
| O07/O08 | sr_math.retreat_target / matched_targets旋转配对 | SKIPPED_PROTOCOL：需要当前交互实体真实位置／方向与命令范围。新RLDS的state是8维本体，不是物体几何或仿真完整状态；旧237/217观测不能替代新1920池 |
| S01 | B01 | 完全相同配置时ALIAS，不重复训练 |
| S02 | UADA_OFT_DINO_SAMPLING.py dino_sampling_scores、plan_dino_guided_placement.prepare/fit_prototypes | SKIPPED_PROTOCOL：旧静态分数明确依赖逐状态目标／背景掩码和任务原型。新1920观测无配套标签；不另发明无标签分数冒作原方法 |
| S03–S06 | collect_exploring_oft_dino_replay.py score_record、dino_replay_common.py | 可在独立诊断中用DINO恢复计算score=max(0,Lrestored−Lpatched)；全部训练仍走完整自然梯度。冻结B01收集一次公共池；S04/S05/S06只变mix/权重／对应 |
| S07 | sampling_clear动态重评分实现 | 沿用sampling_clear2000配置refresh_after_outer=250/500/1000/1500；本轮500outer内只在250后重评分一次，不缩放时点。沿用S05的同一2/1权重生成，仅刷新当前PNG的源恢复分数，记录额外前向成本 |
| M01/M02/M03 | 同PNG跨视角／同状态单分支恢复／逐决策与执行控制记录 | 独立诊断，不计作补丁训练或原生PNG成绩 |

## 已冻结公共协议

联合模型revision 638918f3d1c2e43a39a8a20772bdb8b91835e4b7，300k L1头及本体投影器；各套件官方归一化。raw RGB50×50；源RandomPatchTransform geometry=True、resize_patch=False，不新增TV/HF或亮度变化；源transformers.AdamW，lr.002、wd0，warmup20outer，500×50actualupdates，scheduler每outer一次。第一outer学习率为0，保留源语义。

动作端点统一采用草案允许的标准化有效区间：XYZ/旋转[-1,1]，夹爪[0,1]，clean≥中点取下界。不同于旧Object预测分位数；所有方法使用同一规则，不按套件效果改端点。

训练任务24个，每任务10互异专家轨迹×8时序状态。开发初态0–9；全40任务最终0–9；确认8任务追加10–19，运行前核验可用性。相同起点／种子配对，未按clean成功与否筛选名单。

## 阻塞与失败口径

方法缺标签或数学定义不符合单变量时标SKIPPED_PROTOCOL，不花GPU训练一个替代问题。工程错误保留原异常／Slurm成本，定位后修复重提；科学负结果不通过改参数挽救。本轮不生成视频。
