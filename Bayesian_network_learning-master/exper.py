import pandas as pd
import numpy as np
import os
import sys
import time

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从bnsl.py导入必要的函数和类
from bnsl import (
    load_standard_network,
    calculate_hamming_distance,
    calculate_kl_divergence,
    calculate_correct_edge_ratio,
    calculate_bic_score
)
from LearningWithExpertKnowledge.estimator import Estimator
from LearningWithExpertKnowledge.expert import ExpertKnowledge


def run_single_bnsl(random_state):
    """
    运行一次贝叶斯网络学习
    
    Args:
        random_state: 随机种子，确保每次运行的样本不同
    
    Returns:
        tuple: (汉明距离, KL散度, 正确边比率, BIC评分, 样本量)
    """
    # 配置参数
    config = {
        'sample_fraction': 0.1,  # 样本比例
        'k_value': 1,  # 专家知识权重
        'random_state': random_state,
        'standard_network_path': r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge\data\标准网络.csv"
    }
    
    try:
        # 加载数据
        asian_data = pd.read_csv(
            r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge\data\asian.csv", 
            index_col=0
        )
        expert_data = pd.read_csv(
            r"E:\BNSL\Bayesian_network_learning-master\LearningWithExpertKnowledge\data\asian_expert.csv", 
            index_col=0
        )
        
        # 根据比例采样数据
        if config['sample_fraction'] < 1.0:
            sampled_data = asian_data.sample(
                frac=config['sample_fraction'], 
                random_state=config['random_state']
            )
        else:
            sampled_data = asian_data
            
        # 获取样本量
        sample_size = len(sampled_data)
        
        # 创建专家知识对象
        huang = ExpertKnowledge(data=expert_data)
        
        # 实例化估计器
        est = Estimator(data=sampled_data, expert=huang, k=config['k_value'])
        
        # 运行估计器
        est.run()
        
        # 计算BIC评分
        bic_score = calculate_bic_score(est.DAG, sampled_data)
        
        # 计算与标准网络的比较指标
        hamming_dist = None
        kl_div = None
        correct_edge_ratio = None
        
        if config['standard_network_path']:
            standard_dag = load_standard_network(config['standard_network_path'])
            hamming_dist = calculate_hamming_distance(est.DAG, standard_dag)
            kl_div = calculate_kl_divergence(est.DAG, standard_dag, sampled_data)
            correct_edge_ratio = calculate_correct_edge_ratio(est.DAG, standard_dag)
        
        return hamming_dist, kl_div, correct_edge_ratio, bic_score, sample_size
    
    except Exception as e:
        print(f"运行出错 (随机种子 {random_state}): {e}")
        return None, None, None, None, None


def run_multiple_experiments(num_runs=100):
    """
    运行多次贝叶斯网络学习并计算平均结果
    
    Args:
        num_runs: 运行次数
    """
    print(f"开始运行 {num_runs} 次贝叶斯网络学习实验...")
    
    hamming_distances = []
    kl_divergences = []
    correct_edge_ratios = []
    bic_scores = []
    sample_sizes = []
    
    start_time = time.time()
    
    for i in range(num_runs):
        # 使用不同的随机种子
        random_state = 100 + i
        
        print(f"\n运行 {i+1}/{num_runs} (随机种子: {random_state})...")
        run_start_time = time.time()
        
        # 运行一次实验
        hamming, kl, correct_ratio, bic, sample_size = run_single_bnsl(random_state)
        
        if hamming is not None and kl is not None and correct_ratio is not None and bic is not None and sample_size is not None:
            hamming_distances.append(hamming)
            kl_divergences.append(kl)
            correct_edge_ratios.append(correct_ratio)
            bic_scores.append(bic)
            sample_sizes.append(sample_size)
            
            print(f"  完成！样本量: {sample_size}, 汉明距离: {hamming}, KL散度: {kl:.4f}, 正确边比率: {correct_ratio:.4f}, BIC评分: {bic:.4f}")
        else:
            print("  失败，跳过此次结果")
        
        run_time = time.time() - run_start_time
        print(f"  运行时间: {run_time:.2f} 秒")
    
    total_time = time.time() - start_time
    
    # 计算平均值
    avg_hamming = np.mean(hamming_distances) if hamming_distances else 0
    avg_kl = np.mean(kl_divergences) if kl_divergences else 0
    avg_correct_edge_ratio = np.mean(correct_edge_ratios) if correct_edge_ratios else 0
    avg_bic = np.mean(bic_scores) if bic_scores else 0
    
    # 计算标准差
    std_hamming = np.std(hamming_distances) if hamming_distances else 0
    std_kl = np.std(kl_divergences) if kl_divergences else 0
    std_correct_edge_ratio = np.std(correct_edge_ratios) if correct_edge_ratios else 0
    std_bic = np.std(bic_scores) if bic_scores else 0
    
    # 计算平均样本量
    avg_sample_size = np.mean(sample_sizes) if sample_sizes else 0
    
    # 打印结果
    print("\n" + "="*75)
    print(f"实验总结 ({len(hamming_distances)}/{num_runs} 次成功)")
    print("="*75)
    print(f"样本量: {avg_sample_size:.0f}")
    print(f"平均汉明距离: {avg_hamming:.4f} (±{std_hamming:.4f})")
    print(f"平均KL散度: {avg_kl:.4f} (±{std_kl:.4f})")
    print(f"平均正确边比率: {avg_correct_edge_ratio:.4f} (±{std_correct_edge_ratio:.4f})")
    print(f"平均BIC评分: {avg_bic:.4f} (±{std_bic:.4f})")
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"平均每次运行时间: {total_time/num_runs:.2f} 秒")
    print("="*75)
    
    # 只在控制台输出结果，不再保存到文件
    print("\n实验结果详情:")
    print("="*75)
    print(f"{'运行次数':<10}{'样本量':<10}{'汉明距离':<12}{'KL散度':<15}{'正确边比率':<15}{'BIC评分':<15}")
    print("-"*90)
    for i in range(len(hamming_distances)):
        print(f"{i+1:<10}{sample_sizes[i]:<10}{hamming_distances[i]:<12.2f}{kl_divergences[i]:<15.4f}{correct_edge_ratios[i]:<15.4f}{bic_scores[i]:<15.4f}")
    print("="*75)


if __name__ == '__main__':
    # 运行10次实验
    run_multiple_experiments(num_runs=10)