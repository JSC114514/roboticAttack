# Adapter同状态分支恢复

两张预先指定的seed42 PNG，每套件20个起点，每张三种恢复；共480回合。分支供体来自当前实际状态贴PNG之前的观测。此表用于定位传播路径。

| 套件 | 补丁 | 恢复分支 | 任务成功/20 | 救回原新增失败 | 相对原补丁新增失败 |
|---|---|---|---:|---:|---:|
| Spatial | LR004-B01-seed42 | DINO | 15/20 | 1/1 | 1 |
| Spatial | LR004-B01-seed42 | SigLIP | 17/20 | 1/1 | 0 |
| Spatial | LR004-B01-seed42 | 两分支 | 16/20 | 1/1 | 0 |
| Spatial | LR004-O03-seed42 | DINO | 10/20 | 0/6 | 0 |
| Spatial | LR004-O03-seed42 | SigLIP | 16/20 | 6/6 | 0 |
| Spatial | LR004-O03-seed42 | 两分支 | 16/20 | 6/6 | 0 |
| Object | LR004-B01-seed42 | DINO | 20/20 | 2/2 | 0 |
| Object | LR004-B01-seed42 | SigLIP | 20/20 | 2/2 | 0 |
| Object | LR004-B01-seed42 | 两分支 | 20/20 | 2/2 | 0 |
| Object | LR004-O03-seed42 | DINO | 20/20 | 0/0 | 0 |
| Object | LR004-O03-seed42 | SigLIP | 20/20 | 0/0 | 0 |
| Object | LR004-O03-seed42 | 两分支 | 20/20 | 0/0 | 0 |
| Goal | LR004-B01-seed42 | DINO | 20/20 | 1/1 | 0 |
| Goal | LR004-B01-seed42 | SigLIP | 20/20 | 1/1 | 0 |
| Goal | LR004-B01-seed42 | 两分支 | 19/20 | 1/1 | 1 |
| Goal | LR004-O03-seed42 | DINO | 20/20 | 6/6 | 0 |
| Goal | LR004-O03-seed42 | SigLIP | 20/20 | 6/6 | 0 |
| Goal | LR004-O03-seed42 | 两分支 | 19/20 | 6/6 | 1 |
| Long | LR004-B01-seed42 | DINO | 15/20 | 6/8 | 1 |
| Long | LR004-B01-seed42 | SigLIP | 15/20 | 5/8 | 0 |
| Long | LR004-B01-seed42 | 两分支 | 17/20 | 8/8 | 1 |
| Long | LR004-O03-seed42 | DINO | 15/20 | 3/3 | 3 |
| Long | LR004-O03-seed42 | SigLIP | 15/20 | 2/3 | 3 |
| Long | LR004-O03-seed42 | 两分支 | 17/20 | 3/3 | 1 |

原新增失败的分母：同一目标模型、同一起点clean成功且该PNG失败。新增失败：原PNG成功、恢复条件失败，完整保留。两分支恢复每次决策均核验与同状态clean动作最大误差≤1e-5；闭环成功率另行按实际结果统计。

LR004-O03 seed42合计15个原新增失败：DINO恢复救回9个，SigLIP恢复救回14个，两分支恢复救回15个。Spatial的6个由SigLIP恢复全部救回而DINO恢复未救回；Goal的6个两种单分支恢复均能救回；Long的3个DINO救回3个、SigLIP救回2个。
