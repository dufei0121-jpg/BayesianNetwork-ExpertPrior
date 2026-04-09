# 融合专家确信程度的贝叶斯网络结构学习（改进版）

## 📋 项目概述

本项目是基于 [Howardhuang98/Bayesian_network_learning](https://github.com/Howardhuang98/Bayesian_network_learning) 的改进版本，针对贝叶斯网络结构学习中专家先验评分函数进行了重要改进。原方法在处理专家对多数边不确定的场景时存在局限性，本改进版重新设计了评分机制，使专家知识仅在专家确定时发挥显著作用，在专家不确定时趋于中性。

## ✨ 改进亮点

### 🎯 问题识别
原方法将专家对每条边的置信度（三个值：指向、无、反向）连乘，再通过分段线性变换映射到极值。这种方法存在两个主要问题：
1. **乘积效应过小**：当专家对大多数边不确定（置信度≈0.333）时，连乘结果极小，失去指导意义
2. **噪声放大**：线性变换会放大专家判断中的噪声，影响学习效果

### 🚀 创新方案
本改进版针对地质工程等实际应用场景中"专家对多数边模棱两可、仅对少数边确信"的特点，重新设计了专家评分函数：

- **确信程度量化**：基于专家确信程度偏离中性点（0.5）的距离进行平方惩罚
- **中性点设计**：当专家完全不确定时（置信度=0.5），评分为0，不影响结构学习
- **自适应权重**：专家分数乘以 `k * log(sample_size)`，使先验强度随样本量对数增长，避免小样本下先验过强

## 🔬 算法原理

### 专家评分函数
对于每个候选父节点 `node`，取专家认为该节点作为父节点的置信度 `thinks[1]`（范围 0~1）：

- **若 `node` 在父节点集合中**：
  ```
  贡献 = abs(thinks[1] - 0.5) × (thinks[1] - 0.5)
  ```
- **若 `node` 不在父节点集合中**：
  ```
  贡献 = abs((1 - thinks[1]) - 0.5) × ((1 - thinks[1]) - 0.5)
  ```

### 设计特性
1. **单调性**：专家越确信（接近0或1），评分贡献越大
2. **对称性**：对肯定和否定判断给予对称的权重
3. **鲁棒性**：专家越不确定（接近0.5），贡献越小甚至忽略
4. **可扩展性**：通过超参数 `k` 调节专家知识的整体权重

## 🛠️ 安装与使用

### 环境要求
```bash
pip install pandas networkx numpy matplotlib tqdm
```

### 快速开始
```python
from LearningWithExpertKnowledge.estimator import *
from LearningWithExpertKnowledge.expert import *
import pandas as pd

if __name__ == '__main__':
    # 1. 加载数据
    asian_data = pd.read_csv(r"./data/asian.csv", index_col=0)
    expert_data = pd.read_csv(r"./data/asian_expert.csv", index_col=0)
    
    # 2. 实例化专家
    huang = ExpertKnowledge(data=expert_data)
    
    # 3. 创建估计器
    est = Estimator(data=asian_data, expert=huang, k=100000)
    
    # 4. 运行学习算法
    est.run()
    
    # 5. 保存和查看结果
    est.DAG.save_to_png(weight=False)
    print("学习到的边：", est.DAG.edges)
    
    # 6. 导出到Excel（可选）
    est.DAG.to_excel("dag_result.xlsx")
```

## 📁 文件结构

```
├── README.md              # 本文件（改进说明）
├── README_original.md     # 原项目的说明文档
├── estimator.py           # 算法主体（修改了expert_score函数）
├── expert.py              # 专家模块（提供专家知识查询）
├── graph.py               # 图结构（原DAG.py，未改动）
├── log.txt                # 运行日志
└── data/                  # 数据集目录
    ├── asian.csv
    ├── asian.png
    ├── asian_expert.csv
    ├── asian_expert.xlsx
    ├── data.xlsx
    ├── standard_network.csv
    └── standard_network.xlsx
```

## 📊 数据格式

### 专家知识矩阵
专家知识以CSV或Excel格式存储，表示专家对变量间因果关系的置信度：
- **行指向列**：`matrix[i][j]` 表示专家认为 `i → j` 的置信度
- **取值范围**：0~1，且 `matrix[i][j] + matrix[j][i] ≤ 1`
- **对角线**：始终为0（变量不能指向自身）

示例：
```
    A    B    C    D
A  0    0.1  0.5  0.3
B  0.8  0    0.2  0.2
C  0.7  0.3  0    0.1
D  0.3  0.9  0.1  0
```

### 观测数据
观测数据为标准的数据框格式，每列为一个变量，每行为一个观测样本。

## 🎯 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `k` | float | 10000 | 专家知识的整体权重系数 |
| `data` | pd.DataFrame | - | 观测数据 |
| `expert` | ExpertKnowledge | - | 专家知识实例 |

## 📈 结果可视化

学习到的贝叶斯网络结构可以通过多种方式可视化：

1. **PNG图像**：`est.DAG.save_to_png(weight=False)` 生成网络图
2. **Excel格式**：`est.DAG.to_excel("filename.xlsx")` 导出边列表
3. **Cytoscape兼容**：导出的Excel文件可直接导入 [Cytoscape](https://cytoscape.org/) 进行高级可视化

## 🔍 调试与日志

算法运行过程中会生成详细的日志文件 `log.txt`，包含：
- 数据预览信息
- 专家知识预览
- 每次迭代的评分变化
- 最终学习到的网络结构

## 📚 参考文献

1. 高晓光, 叶思懋, 邸若海, 等. 基于融合先验方法的贝叶斯网络结构学习[J]. 系统工程与电子技术, 2018, 40(4): 790-796.
2. Howard Huang. Bayesian Network Learning with Expert Knowledge. https://github.com/Howardhuang98/Bayesian_network_learning

## 🤝 贡献指南

欢迎对本项目进行改进和扩展：

1. **专家评分函数**：可在 `estimator.py` 中进一步优化 `expert_score` 方法
2. **学习算法**：可在 `estimator.py` 中改进搜索策略或评分函数
3. **可视化**：可在 `graph.py` 中增强网络可视化功能

## 📄 许可证

本项目基于原项目 [Howardhuang98/Bayesian_network_learning](https://github.com/Howardhuang98/Bayesian_network_learning) 进行改进，遵循原项目的许可证条款。