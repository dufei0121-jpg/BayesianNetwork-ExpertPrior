#import sys
#import os
# 使用原始字符串避免转义问题
#sys.path.append(r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge")
import pandas as pd
import numpy as np
import networkx as nx
import math
from LearningWithExpertKnowledge.estimator import * 
from LearningWithExpertKnowledge.expert import *
from LearningWithExpertKnowledge.graph import DAG


def calculate_hamming_distance(dag1, dag2):
    """
    计算两个有向无环图之间的汉明距离
    汉明距离 = 两个图中边的差集大小
    """
    edges1 = set(dag1.edges)
    edges2 = set(dag2.edges)
    
    # 汉明距离等于两个边集的对称差的大小
    return len(edges1.symmetric_difference(edges2))


def calculate_kl_divergence(dag1, dag2, data):
    """
    计算两个贝叶斯网络之间的KL散度
    这里使用简化版本，基于边的存在概率
    """
    # 获取所有可能的边组合
    nodes = list(set(dag1.nodes).union(set(dag2.nodes)))
    all_possible_edges = set((u, v) for u in nodes for v in nodes if u != v)
    
    kl_divergence = 0
    
    # 计算每个可能的边在两个网络中的存在情况
    for edge in all_possible_edges:
        # 边存在的概率（这里简化为0或1）
        p = 1 if edge in dag1.edges else 0
        q = 1 if edge in dag2.edges else 0
        
        # KL散度计算：p * log(p/q)，需要避免log(0)的情况
        if p > 0 and q > 0:
            kl_divergence += p * math.log(p / q)
        elif p > 0 and q == 0:
            # 当p>0且q=0时，KL散度为无穷大，这里用一个大值代替
            kl_divergence += 100
    
    return kl_divergence


def calculate_correct_edge_ratio(learned_dag, standard_dag):
    """
    计算正确边比率
    
    Args:
        learned_dag: 学习得到的DAG
        standard_dag: 标准DAG
        
    Returns:
        float: 正确边比率（0-1之间）
    """
    learned_edges = set(learned_dag.edges)
    standard_edges = set(standard_dag.edges)
    
    # 正确识别的边（两个网络都有的边）
    correct_edges = learned_edges.intersection(standard_edges)
    
    # 错误识别的边（学习网络有但标准网络没有的边）
    false_edges = learned_edges - standard_edges
    
    # 遗漏的边（标准网络有但学习网络没有的边）
    missed_edges = standard_edges - learned_edges
    
    # 正确边比率（基于标准网络）
    if len(standard_edges) > 0:
        correct_ratio = len(correct_edges) / len(standard_edges)
    else:
        correct_ratio = 0.0
    
    return correct_ratio


def calculate_bic_score(dag, data):
    """
    计算贝叶斯网络的BIC评分
    BIC = log_likelihood - (d * log(n)) / 2
    其中：d是模型参数数量，n是样本数量
    """
    n = len(data)
    log_likelihood = 0
    d = 0  # 模型参数数量
    
    # 对每个节点计算其条件对数似然和参数数量
    for node in dag.nodes:
        parents = dag.get_parents(node)
        
        # 计算节点的状态数和父节点的状态组合数
        node_states = len(data[node].unique())
        parent_states = 1
        
        # 计算父节点的状态组合数
        for parent in parents:
            parent_states *= len(data[parent].unique())
        
        # 参数数量：(节点状态数 - 1) * 父节点状态组合数
        d += (node_states - 1) * parent_states
        
        # 简化的对数似然计算
        # 按父节点状态分组，计算每个分组内节点状态的频率
        if parents:
            grouped = data.groupby(parents)
            for _, group in grouped:
                # 计算每个节点状态的频率
                counts = group[node].value_counts()
                total = len(group)
                
                # 添加拉普拉斯平滑以避免log(0)
                for count in counts:
                    log_likelihood += count * math.log((count + 1) / (total + node_states))
        else:
            # 无父节点的情况
            counts = data[node].value_counts()
            total = len(data)
            
            for count in counts:
                log_likelihood += count * math.log((count + 1) / (total + node_states))
    
    # 计算BIC评分
    bic = log_likelihood - (d * math.log(n)) / 2
    return bic


def load_standard_network(file_path):
    """
    从文件加载标准网络
    这里假设文件是CSV格式，包含source和target列
    """
    standard_data = pd.read_csv(file_path)
    standard_dag = DAG()
    
    # 添加节点和边
    nodes = set(standard_data['source']).union(set(standard_data['target']))
    standard_dag.add_nodes_from(nodes)
    
    for _, row in standard_data.iterrows():
        standard_dag.add_edge(row['source'], row['target'])
    
    return standard_dag


if __name__ == '__main__':          
     # 配置参数
     config = {
         'sample_fraction': 1,  # 样本比例 (0.1-1.0)
         'k_value': 0,  # 专家知识权重
         'random_state': 42,  # 随机种子
         'standard_network_path': r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge\data\标准网络.csv"  # 标准网络文件路径，None表示不进行比较
     }
     
     # 加载数据
     asian_data = pd.read_csv(r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge\data\asian.csv", index_col=0)     
     expert_data = pd.read_csv(r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge\data\asian_expert.csv", index_col=0)     
     
     # 根据比例采样数据
     if config['sample_fraction'] < 1.0:
         sampled_data = asian_data.sample(frac=config['sample_fraction'], random_state=config['random_state'])
         print(f"使用{len(sampled_data)}个样本（占总样本的{config['sample_fraction']*100:.1f}%）")
     else:
         sampled_data = asian_data
         print(f"使用全部{len(sampled_data)}个样本")
     
     # 创建专家知识对象
     huang = ExpertKnowledge(data=expert_data)      
     
     # 实例化估计器，传入采样后的数据
     est = Estimator(data=sampled_data, expert=huang, k=config['k_value'])      
     
     # 运行估计器
     print("开始学习网络结构...")
     est.run()      
     
     # 保存和显示结果
     est.DAG.save_to_png(weight=False)      
     print("\n学习得到的网络边：")
     print(est.DAG.edges)
     print(f"总边数：{len(est.DAG.edges)}")
     
     # 计算BIC评分
     bic_score = calculate_bic_score(est.DAG, sampled_data)
     print(f"\nBIC评分：{bic_score:.4f}")
     
     # 计算与标准网络的比较指标
     if config['standard_network_path']:
         try:
             standard_dag = load_standard_network(config['standard_network_path'])
             hamming_dist = calculate_hamming_distance(est.DAG, standard_dag)
             kl_div = calculate_kl_divergence(est.DAG, standard_dag, sampled_data)
             print(f"\n与标准网络的汉明距离：{hamming_dist}")
             print(f"与标准网络的KL散度：{kl_div:.4f}")
         except Exception as e:
             print(f"\n无法加载标准网络或计算距离：{e}")
     else:
         print("\n未指定标准网络，跳过比较")
         # 如果需要，可以手动创建一个标准网络进行比较
         # 例如，从现有的亚洲网络数据创建一个标准DAG
         # standard_dag = DAG()
         # standard_dag.add_nodes_from(asian_data.columns)
         # # 添加预期的边
         # expected_edges = [('A', 'S'), ('T', 'L'), ('S', 'L'), ('S', 'B'), ('L', 'E'), ('B', 'E'), ('E', 'X'), ('X', 'D')]
         # standard_dag.add_edges_from(expected_edges)
         # # 计算距离
         # hamming_dist = calculate_hamming_distance(est.DAG, standard_dag)
         # kl_div = calculate_kl_divergence(est.DAG, standard_dag, sampled_data)
         # print(f"\n与预期网络的汉明距离：{hamming_dist}")
         # print(f"与预期网络的KL散度：{kl_div:.4f}")
     
     # 运行不同样本量的对比实验（可选）
     run_sample_comparison = False  # 设置为True运行对比实验
     
     if run_sample_comparison:
         sample_sizes = [0.2, 0.5, 0.8, 1.0]
         results = []
         
         print("\n开始不同样本量的对比实验...")
         for size in sample_sizes:
             sampled_data = asian_data.sample(frac=size, random_state=config['random_state'])
             print(f"\n使用{len(sampled_data)}个样本（占总样本的{size*100:.1f}%）")
             est = Estimator(data=sampled_data, expert=huang, k=config['k_value'])
             est.run()
             bic = calculate_bic_score(est.DAG, sampled_data)
             results.append((size, len(est.DAG.edges), bic))
             print(f"样本比例: {size}, 边数: {len(est.DAG.edges)}, BIC: {bic:.4f}")
         
         # 将结果保存到文件
         results_df = pd.DataFrame(results, columns=['样本比例', '边数', 'BIC评分'])
         results_df.to_csv('不同样本量的结果.csv', index=False)
         print("\n对比实验结果已保存到'不同样本量的结果.csv'")