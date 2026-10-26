# 腕部视角适配：模型、数据、方法与诊断

## 模型与实验目的

使用官方联合模型`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`，revision `638918f3d1c2e43a39a8a20772bdb8b91835e4b7`。使用其300000步动作头／本体投影器和各套件归一化统计；输入前视、腕部、语言与本体状态，输出8×7动作。LIBERO Long在代码中记作`libero_10`。

这条路线训练的是小型视觉适配参数，不是对抗补丁。目的是减少无关视角变化造成的错误控制。

## 数据与优化

详细参数完整保留在[原配置](../../research_runs/wrist_geometry_response_adapt_v1/config.json)与[逐版本说明](../../research_runs/wrist_geometry_response_adapt_v1/EXPERIMENT_DETAILS.md)。模型覆盖四套件，专家数据和相机配对均为混合池；不是拿Goal专用模型处理所有套件。

pilot专家数据512对训练、320对验证；相机训练／验证各80状态。相机对在同一物理状态采样，只有腕部位置不同。低秩适配DINO block18—22，rank8、alpha8，训练245,760参数；全部配置1,000更新、seed42。学习率及具体采样规则以配置和wr_train/wr_data为准。

令a为归一化原生动作，每维按跨度(2,2,2,2,2,2,1)缩放：

- B：两时刻预测与对应专家动作的均方误差。
- C：同状态原相机与偏移相机预测动作的均方差。
- R：两时刻预测动作差与专家动作差的误差。

普通一致性优化归一化B+λC。约束方法最小化C，并用非负乘子约束B和R相对校准基线的增幅。R用时序帧计算，不是单独移动物体的物理干预；不能用R解释所有空间响应。

## 真实目标与相机配对诊断

[诊断配置](../../research_runs/wrist_camera_task_response_diag_v1/config.json)固定12条候选轨迹、3模型和3×3干预，先以物理标签选状态再调用策略。真实目标free-joint平移与两路画面同步；相机偏移只改变腕部画面，不改变物理状态。初始画面使用各传感器原生采样快照，避免重算派生姿态造成零干预不一致。

执行8步后计算P=初始目标距离−最终目标距离。目标响应为P(t,c)−P(0,c)，模型相对原模型的交互差再减去原模型同一响应。单位米；机器人基座坐标与相机局部坐标分开记录。完整动作、末端轨迹、接触和抓持同时记录，不能只凭动作更稳定判断控制更好。

## 源码阅读顺序

1. wr_common/wr_data/wr_camera：配置、专家数据、原生时钟和相机采样。
2. wr_engine/wr_math/wr_train：低秩注入、B/C/R定义、训练和断点。
3. wr_evaluate/wr_report：完整闭环与分母、配对统计。
4. diag_prepare/diag_geometry：状态筛选、真实物理位移、可见性与穿插检查。
5. diag_evaluate/diag_report：完整8步执行、实际运动、分层统计和独立视频。

[结果与当前决定](../CURRENT_DIRECTION.md)。这次发布不自动训练或评测。
