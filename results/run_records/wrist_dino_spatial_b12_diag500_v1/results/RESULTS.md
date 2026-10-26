# 16组500轮：逐套件CASR与同batch基线

数据来自本轮配对1360回合；不跨套件平均。
Spatial是来源内，其余三个套件是补丁迁移；40/41为预先固定的开发初态；仅作初步比较。9-B1是退化对照。

|Suite|Clean成功/20|
|---|---:|
|libero_object|20| 
|libero_spatial|19| 
|libero_goal|19| 
|libero_10|18| 

## Batch size=1

|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|
|---|---|---:|---:|---:|---:|---:|
|1|uada_original|20.00|30.00|50.00|90.00|1.76|
|2|dino_key|5.00|30.00|50.00|75.00|0.32|
|3|dino_center|0.00|20.00|35.00|70.00|0.32|
|4|advencoder_cos|0.00|10.00|5.00|70.00|0.37|
|6|dino_multilayer|0.00|10.00|30.00|60.00|0.33|
|7|dino_affinity|0.00|25.00|40.00|65.00|0.32|
|9|advencoder_nce（退化对照）|0.00|10.00|10.00|35.00|0.59|
|12|tma_xyz|0.00|30.00|40.00|90.00|1.77|

同batch相对两种基线的差值（百分点）：

|ID|Suite|对UADA|对TMA|
|---|---|---:|---:|
|2|libero_object|-15.00|+5.00|
|2|libero_spatial|+0.00|+0.00|
|2|libero_goal|+0.00|+10.00|
|2|libero_10|-15.00|-15.00|
|3|libero_object|-20.00|+0.00|
|3|libero_spatial|-10.00|-10.00|
|3|libero_goal|-15.00|-5.00|
|3|libero_10|-20.00|-20.00|
|4|libero_object|-20.00|+0.00|
|4|libero_spatial|-20.00|-20.00|
|4|libero_goal|-45.00|-35.00|
|4|libero_10|-20.00|-20.00|
|6|libero_object|-20.00|+0.00|
|6|libero_spatial|-20.00|-20.00|
|6|libero_goal|-20.00|-10.00|
|6|libero_10|-30.00|-30.00|
|7|libero_object|-20.00|+0.00|
|7|libero_spatial|-5.00|-5.00|
|7|libero_goal|-10.00|+0.00|
|7|libero_10|-25.00|-25.00|
|9|libero_object|-20.00|+0.00|
|9|libero_spatial|-20.00|-20.00|
|9|libero_goal|-40.00|-30.00|
|9|libero_10|-55.00|-55.00|

## Batch size=2

|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|
|---|---|---:|---:|---:|---:|---:|
|1|uada_original|10.00|25.00|60.00|80.00|3.28|
|2|dino_key|5.00|20.00|45.00|75.00|0.60|
|3|dino_center|0.00|15.00|25.00|70.00|0.60|
|4|advencoder_cos|0.00|15.00|20.00|75.00|1.11|
|6|dino_multilayer|10.00|10.00|35.00|60.00|0.68|
|7|dino_affinity|0.00|20.00|10.00|60.00|0.62|
|9|advencoder_nce|5.00|20.00|15.00|80.00|1.10|
|12|tma_xyz|0.00|30.00|50.00|90.00|3.23|

同batch相对两种基线的差值（百分点）：

|ID|Suite|对UADA|对TMA|
|---|---|---:|---:|
|2|libero_object|-5.00|+5.00|
|2|libero_spatial|-5.00|-10.00|
|2|libero_goal|-15.00|-5.00|
|2|libero_10|-5.00|-15.00|
|3|libero_object|-10.00|+0.00|
|3|libero_spatial|-10.00|-15.00|
|3|libero_goal|-35.00|-25.00|
|3|libero_10|-10.00|-20.00|
|4|libero_object|-10.00|+0.00|
|4|libero_spatial|-10.00|-15.00|
|4|libero_goal|-40.00|-30.00|
|4|libero_10|-5.00|-15.00|
|6|libero_object|+0.00|+10.00|
|6|libero_spatial|-15.00|-20.00|
|6|libero_goal|-25.00|-15.00|
|6|libero_10|-20.00|-30.00|
|7|libero_object|-10.00|+0.00|
|7|libero_spatial|-5.00|-10.00|
|7|libero_goal|-50.00|-40.00|
|7|libero_10|-20.00|-30.00|
|9|libero_object|-5.00|+5.00|
|9|libero_spatial|-5.00|-10.00|
|9|libero_goal|-45.00|-35.00|
|9|libero_10|+0.00|-10.00|

所有组同25000次更新，但B2处理样本次数更多。NCE的B变化也改变负样本数。
每任务结果、同方法B对比、NCE对cos、区间和成本见同目录CSV；不自动扩轮或跨模型。
