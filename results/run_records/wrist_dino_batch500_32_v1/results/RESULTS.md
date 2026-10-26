# 32组500轮：逐套件CASR与同batch基线

数据来自本轮配对13200回合；不跨套件平均。
Long是来源内，其余三个套件是补丁迁移；40–49为已有开发初态。9-B1是退化对照。

|Suite|Clean成功/100|
|---|---:|
|libero_object|99| 
|libero_spatial|91| 
|libero_goal|96| 
|libero_10|95| 

## Batch size=1

|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|
|---|---|---:|---:|---:|---:|---:|
|1|uada_original|5.00|20.67|4.00|78.64|1.66|
|2|dino_key|3.00|15.22|31.83|88.67|0.31|
|3|dino_center|9.00|19.44|31.11|77.44|0.31|
|4|advencoder_cos|3.00|17.11|28.42|89.00|0.35|
|6|dino_multilayer|3.00|15.33|22.36|69.92|0.32|
|7|dino_affinity|1.00|18.44|23.11|79.75|0.29|
|9|advencoder_nce（退化对照）|6.00|9.67|3.00|29.25|0.58|
|12|tma_xyz|9.11|23.78|35.72|97.89|1.67|

同batch相对两种基线的差值（百分点）：

|ID|Suite|对UADA|对TMA|
|---|---|---:|---:|
|2|libero_object|-2.00|-6.11|
|2|libero_spatial|-5.44|-8.56|
|2|libero_goal|+27.83|-3.89|
|2|libero_10|+10.03|-9.22|
|3|libero_object|+4.00|-0.11|
|3|libero_spatial|-1.22|-4.33|
|3|libero_goal|+27.11|-4.61|
|3|libero_10|-1.19|-20.44|
|4|libero_object|-2.00|-6.11|
|4|libero_spatial|-3.56|-6.67|
|4|libero_goal|+24.42|-7.31|
|4|libero_10|+10.36|-8.89|
|6|libero_object|-2.00|-6.11|
|6|libero_spatial|-5.33|-8.44|
|6|libero_goal|+18.36|-13.36|
|6|libero_10|-8.72|-27.97|
|7|libero_object|-4.00|-8.11|
|7|libero_spatial|-2.22|-5.33|
|7|libero_goal|+19.11|-12.61|
|7|libero_10|+1.11|-18.14|
|9|libero_object|+1.00|-3.11|
|9|libero_spatial|-11.00|-14.11|
|9|libero_goal|-1.00|-32.72|
|9|libero_10|-49.39|-68.64|

## Batch size=2

|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|
|---|---|---:|---:|---:|---:|---:|
|1|uada_original|13.22|27.00|55.75|97.89|3.32|
|2|dino_key|4.00|17.56|36.42|87.67|0.56|
|3|dino_center|6.00|16.44|27.50|79.44|0.55|
|4|advencoder_cos|6.11|25.56|36.67|86.67|1.01|
|6|dino_multilayer|5.00|18.33|25.33|70.83|0.60|
|7|dino_affinity|7.00|20.56|22.81|82.64|0.55|
|9|advencoder_nce|7.22|18.22|18.94|87.78|1.00|
|12|tma_xyz|9.11|21.33|17.75|89.00|3.30|

同batch相对两种基线的差值（百分点）：

|ID|Suite|对UADA|对TMA|
|---|---|---:|---:|
|2|libero_object|-9.22|-5.11|
|2|libero_spatial|-9.44|-3.78|
|2|libero_goal|-19.33|+18.67|
|2|libero_10|-10.22|-1.33|
|3|libero_object|-7.22|-3.11|
|3|libero_spatial|-10.56|-4.89|
|3|libero_goal|-28.25|+9.75|
|3|libero_10|-18.44|-9.56|
|4|libero_object|-7.11|-3.00|
|4|libero_spatial|-1.44|+4.22|
|4|libero_goal|-19.08|+18.92|
|4|libero_10|-11.22|-2.33|
|6|libero_object|-8.22|-4.11|
|6|libero_spatial|-8.67|-3.00|
|6|libero_goal|-30.42|+7.58|
|6|libero_10|-27.06|-18.17|
|7|libero_object|-6.22|-2.11|
|7|libero_spatial|-6.44|-0.78|
|7|libero_goal|-32.94|+5.06|
|7|libero_10|-15.25|-6.36|
|9|libero_object|-6.00|-1.89|
|9|libero_spatial|-8.78|-3.11|
|9|libero_goal|-36.81|+1.19|
|9|libero_10|-10.11|-1.22|

## Batch size=4

|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|
|---|---|---:|---:|---:|---:|---:|
|1|uada_original|9.00|25.89|46.22|99.00|6.42|
|2|dino_key|12.00|20.56|33.11|82.44|1.04|
|3|dino_center|8.11|19.44|35.22|78.22|1.08|
|4|advencoder_cos|7.22|19.11|26.44|90.89|1.73|
|6|dino_multilayer|8.00|16.44|25.11|74.53|1.18|
|7|dino_affinity|3.00|14.22|19.22|76.39|1.08|
|9|advencoder_nce|12.44|16.00|18.97|93.89|1.92|
|12|tma_xyz|9.00|27.00|40.11|100.00|6.60|

同batch相对两种基线的差值（百分点）：

|ID|Suite|对UADA|对TMA|
|---|---|---:|---:|
|2|libero_object|+3.00|+3.00|
|2|libero_spatial|-5.33|-6.44|
|2|libero_goal|-13.11|-7.00|
|2|libero_10|-16.56|-17.56|
|3|libero_object|-0.89|-0.89|
|3|libero_spatial|-6.44|-7.56|
|3|libero_goal|-11.00|-4.89|
|3|libero_10|-20.78|-21.78|
|4|libero_object|-1.78|-1.78|
|4|libero_spatial|-6.78|-7.89|
|4|libero_goal|-19.78|-13.67|
|4|libero_10|-8.11|-9.11|
|6|libero_object|-1.00|-1.00|
|6|libero_spatial|-9.44|-10.56|
|6|libero_goal|-21.11|-15.00|
|6|libero_10|-24.47|-25.47|
|7|libero_object|-6.00|-6.00|
|7|libero_spatial|-11.67|-12.78|
|7|libero_goal|-27.00|-20.89|
|7|libero_10|-22.61|-23.61|
|9|libero_object|+3.44|+3.44|
|9|libero_spatial|-9.89|-11.00|
|9|libero_goal|-27.25|-21.14|
|9|libero_10|-5.11|-6.11|

## Batch size=8

|ID|方法|Object%|Spatial%|Goal%|Long%|训练小时|
|---|---|---:|---:|---:|---:|---:|
|1|uada_original|14.11|25.89|55.00|98.00|13.02|
|2|dino_key|7.00|21.56|34.31|80.33|2.11|
|3|dino_center|11.00|20.56|31.22|80.56|2.01|
|4|advencoder_cos|13.22|18.22|34.67|96.00|3.86|
|6|dino_multilayer|9.00|19.44|20.00|74.56|2.32|
|7|dino_affinity|5.00|19.22|23.33|86.25|2.11|
|9|advencoder_nce|4.00|22.89|16.61|82.44|3.84|
|12|tma_xyz|9.11|31.11|43.94|98.89|12.82|

同batch相对两种基线的差值（百分点）：

|ID|Suite|对UADA|对TMA|
|---|---|---:|---:|
|2|libero_object|-7.11|-2.11|
|2|libero_spatial|-4.33|-9.56|
|2|libero_goal|-20.69|-9.64|
|2|libero_10|-17.67|-18.56|
|3|libero_object|-3.11|+1.89|
|3|libero_spatial|-5.33|-10.56|
|3|libero_goal|-23.78|-12.72|
|3|libero_10|-17.44|-18.33|
|4|libero_object|-0.89|+4.11|
|4|libero_spatial|-7.67|-12.89|
|4|libero_goal|-20.33|-9.28|
|4|libero_10|-2.00|-2.89|
|6|libero_object|-5.11|-0.11|
|6|libero_spatial|-6.44|-11.67|
|6|libero_goal|-35.00|-23.94|
|6|libero_10|-23.44|-24.33|
|7|libero_object|-9.11|-4.11|
|7|libero_spatial|-6.67|-11.89|
|7|libero_goal|-31.67|-20.61|
|7|libero_10|-11.75|-12.64|
|9|libero_object|-10.11|-5.11|
|9|libero_spatial|-3.00|-8.22|
|9|libero_goal|-38.39|-27.33|
|9|libero_10|-15.56|-16.44|

所有组同25000次更新，但B8处理样本次数更多。NCE的B变化也改变负样本数。
每任务结果、同方法B对比、NCE对cos、区间和成本见同目录CSV；不自动扩轮或跨模型。
