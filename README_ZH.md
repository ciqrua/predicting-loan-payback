# 基于 Stacking 集成的贷款还款预测

本项目针对一个二分类问题：**预测一笔贷款是否能够被成功还清**。  
项目基于借款人的财务信息与人口统计特征，构建了一个 Kaggle 风格的机器学习流程，采用**稳健的交叉验证策略**与 **Stacking 集成学习方法** 来提升模型的泛化能力与稳定性。

---

## 目录
- [问题定义](#问题定义)
- [数据集说明](#数据集说明)
- [整体流程](#整体流程)
- [特征处理](#特征处理)
- [建模策略](#建模策略)
- [模型评估](#模型评估)
- [实验结果](#实验结果)
- [项目结构](#项目结构)
- [复现方式](#复现方式)
- [未来改进方向](#未来改进方向)

---

## 问题定义
给定历史贷款记录数据，目标是预测一笔贷款是否会被成功还清。  
该任务被建模为一个 **二分类问题**。由于目标变量存在一定程度的不平衡，项目选用 **ROC-AUC** 作为主要评估指标，以更全面地衡量模型的整体判别能力。

---

## 数据集说明
- 数据来源：Kaggle Playground Series（Season 5, Episode 11）
- 目标变量：`loan_paid_back`
- 特征类型包括：
  - **数值型特征**：收入、负债比、贷款金额、利率等
  - **类别型特征**：就业状态、教育背景、住房情况等

出于数据许可与仓库体积的考虑，原始数据文件未直接包含在本仓库中。  
数据获取方式及文件说明详见 `data/README.md`。

---

## 整体流程
项目整体流程如下：

1. 加载并合并训练集与测试集（保证预处理一致性）
2. 缺失值处理
3. 类别特征编码
4. Stratified K-Fold 交叉验证
5. 训练多个梯度提升模型
6. 生成 Out-of-Fold（OOF）预测
7. 构建 Stacking 集成模型
8. 使用 ROC-AUC 进行模型性能评估

---

## 特征处理
- **数值型特征**
  - 使用中位数（median）填补缺失值
- **类别型特征**
  - 缺失值统一填充为 `"missing"`
  - 使用 `LabelEncoder` 进行编码
- 在建模前移除不具备预测意义的标识列（`id`）

所有特征处理步骤在训练集与测试集上保持完全一致，以避免数据泄漏问题。

---

## 建模策略

### 基模型
采用 **Stratified K-Fold（5 折）交叉验证**，并结合多个随机种子，对以下树模型进行训练：

- LightGBM  
- XGBoost  
- CatBoost  

对于每一个基模型，都会生成对应的 **Out-of-Fold（OOF）预测结果**，从而获得对模型泛化性能更加可靠的估计。

### Stacking 集成
将各基模型的 OOF 预测结果作为元特征（meta-features），训练一个 **Logistic Regression** 作为元模型。  
该 Stacking 方法能够融合不同模型的优势，从而提升整体预测性能与稳定性。

---

## 模型评估
- **评估指标**：ROC-AUC  
- **验证方式**：Out-of-Fold（OOF）预测  
- **可视化方式**：基于 OOF 概率绘制 ROC 曲线  

该评估策略可以有效避免信息泄漏，更真实地反映模型在未见数据上的表现。

---

## 实验结果

| 模型 | OOF ROC-AUC |
|------|-------------:|
| LightGBM | 0.9227 |
| XGBoost | 0.9219 |
| CatBoost | 0.9232 |
| Stacking 集成模型 | **0.9234** |

实验结果表明，Stacking 集成模型取得了最佳性能，验证了多模型融合在该任务中的有效性。

**Kaggle 榜单成绩**

- Public Score：0.92343  
- Private Score：0.92425  

榜单成绩与基于 Out-of-Fold（OOF）预测得到的 ROC-AUC 基本一致，
表明模型具有良好的泛化能力，验证策略较为可靠。


---

## 项目结构
```text
.
├── predicting-loan-payback.ipynb
├── README.md
├── requirements.txt
├── submission.csv
├── data/
│   └── README.md
└── images/
    ├─ ROC Curve.png
    └─ Target Distribution.png
```

---

## 复现方式
1. 安装项目依赖：
```bash
pip install -r requirements.txt
```
2. 从 Kaggle 下载数据集，并将 train.csv 与 test.csv 放入 data/ 目录。
3. 运行 Notebook：
```bash
jupyter notebook predicting-loan-payback.ipynb
```

---

## 未来改进方向

- 使用 Target Encoding 等方法替代 Label Encoding，以更好地表达类别特征

- 增加特征重要性分析与误差分析

- 尝试不同的元模型（Meta-Model）以进一步优化 Stacking 效果

---
