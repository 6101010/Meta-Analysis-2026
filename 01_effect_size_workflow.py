#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import re
import warnings
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import scipy.stats as stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

class MetaAnalysisWorkflow:
    """元分析Effect Size计算自动化工作流"""
    
    def __init__(self):
        """初始化工作流"""
        # 阶段0：环境配置与用户意图解析
        self.setup_configuration()
        self.setup_logging()
        
    def setup_configuration(self):
        """设置核心配置参数"""
        # --- 核心文件与设置 ---
        self.FILE_PATH = r"meta_analysis_results_english.csv"
        self.ENCODING = "utf-8"
        self.RANDOM_SEED = 42
        
        # --- 输出文件命名 ---
        self.OUTPUT_DATA_FILENAME = "meta_analysis_prepared_data_v1.4.csv"
        self.OUTPUT_AUDIT_LOG_FILENAME = "data_audit_log_v1.4.txt"
        self.OUTPUT_VISUALIZATION_FILENAME = "es_distribution_histogram_v1.4.png"
        self.OUTPUT_REPORT_FILENAME = "meta_analysis_comprehensive_report_v1.4.md"
        
        # --- 层级1: 强制覆盖 (Highest Priority) ---
        self.FORCED_EFFECT_SIZE_PATTERN = None  # 可设为 'A', 'B', 或 'C'
        self.FORCED_CLUSTER_VARIABLE = None     # 可设为具体的列名
        
        # --- 层级2: 引导发现 (Medium Priority) ---
        self.CUSTOM_COLUMN_ALIASES = None       # 自定义列名别名
        self.CUSTOM_CLUSTER_CANDIDATES = None   # 自定义聚类候选变量
        
        # 设置随机种子
        np.random.seed(self.RANDOM_SEED)
        
        # 初始化审计日志
        self.audit_log = []
        
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('meta_analysis_workflow.log', encoding='utf-8-sig'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def define_effect_size_patterns(self):
        """定义Effect Size模式的精确匹配别名"""
        self.EXACT_MATCH_ALIASES = {
            # 模式A: 连续变量 (Cohen's d)
            'mean_e': ['mean_experimental', 'mean_exp', 'mean_treatment', 'mean_treat', 
                      'mean_post_experimental', 'mean_post_exp', 'experimental_mean',
                      'treatment_mean', 'exp_mean', 'treat_mean'],
            'sd_e': ['sd_experimental', 'sd_exp', 'sd_treatment', 'sd_treat',
                    'sd_post_experimental', 'sd_post_exp', 'experimental_sd',
                    'treatment_sd', 'exp_sd', 'treat_sd'],
            'n_e': ['n_experimental', 'n_exp', 'n_treatment', 'n_treat',
                   'sample_size_experimental', 'experimental_n', 'treatment_n',
                   'exp_n', 'treat_n'],
            'mean_c': ['mean_control', 'mean_ctrl', 'mean_post_control', 
                      'mean_post_ctrl', 'control_mean', 'ctrl_mean'],
            'sd_c': ['sd_control', 'sd_ctrl', 'sd_post_control', 'sd_post_ctrl',
                    'control_sd', 'ctrl_sd'],
            'n_c': ['n_control', 'n_ctrl', 'sample_size_control', 'control_n',
                   'ctrl_n'],
            
            # 模式B: 二元变量 (Odds Ratio, Risk Ratio)
            'events_e': ['events_experimental', 'events_exp', 'events_treatment',
                        'events_treat', 'experimental_events', 'treatment_events'],
            'total_e': ['total_experimental', 'total_exp', 'total_treatment',
                       'total_treat', 'experimental_total', 'treatment_total'],
            'events_c': ['events_control', 'events_ctrl', 'control_events',
                        'ctrl_events'],
            'total_c': ['total_control', 'total_ctrl', 'control_total',
                       'ctrl_total'],
            
            # 模式C: 相关系数
            'r': ['correlation', 'corr', 'pearson_r', 'r_value'],
            'n': ['sample_size', 'total_n', 'sample_size_total', 'n_total']
        }
        
        self.HEURISTIC_KEYWORDS = {
            'mean_e': {'core': ['mean', 'm', 'avg'], 'context': ['exp', 'e', 'treat', 'experimental', 'treatment']},
            'sd_e': {'core': ['sd', 'std'], 'context': ['exp', 'e', 'treat', 'experimental', 'treatment']},
            'n_e': {'core': ['n', 'num', 'size'], 'context': ['exp', 'e', 'treat', 'experimental', 'treatment']},
            'mean_c': {'core': ['mean', 'm', 'avg'], 'context': ['c', 'ctrl', 'control']},
            'sd_c': {'core': ['sd', 'std'], 'context': ['c', 'ctrl', 'control']},
            'n_c': {'core': ['n', 'num', 'size'], 'context': ['c', 'ctrl', 'control']},
            'events_e': {'core': ['event', 'e'], 'context': ['exp', 'treat', 'experimental', 'treatment']},
            'total_e': {'core': ['total', 'n'], 'context': ['exp', 'treat', 'experimental', 'treatment']},
            'events_c': {'core': ['event', 'e'], 'context': ['c', 'ctrl', 'control']},
            'total_c': {'core': ['total', 'n'], 'context': ['c', 'ctrl', 'control']},
            'r': {'core': ['r', 'cor', 'corr'], 'context': []},
            'n': {'core': ['n', 'size'], 'context': []}
        }
        
    def load_data(self) -> pd.DataFrame:
        """阶段1：数据摄取与智能加载"""
        self.logger.info("开始阶段1：数据摄取与模式解析")
        
        try:
            # 1.1 智能加载
            df = pd.read_csv(self.FILE_PATH, encoding=self.ENCODING)
            self.logger.info(f"成功加载数据文件，包含 {len(df)} 行，{len(df.columns)} 列")
            
            # 1.2 列名标准化
            df.columns = df.columns.str.lower().str.strip()
            self.logger.info("完成列名标准化处理")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"文件未找到: {self.FILE_PATH}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("数据文件为空")
            raise
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise
            
    def detect_effect_size_pattern(self, df: pd.DataFrame) -> str:
        """1.3 Effect Size模式解析"""
        self.logger.info("开始Effect Size模式检测")
        
        # 检查强制覆盖
        if self.FORCED_EFFECT_SIZE_PATTERN:
            self.logger.info(f"使用强制指定的Effect Size模式: {self.FORCED_EFFECT_SIZE_PATTERN}")
            return self.FORCED_EFFECT_SIZE_PATTERN
            
        # 定义模式所需的列
        pattern_requirements = {
            'A': ['mean_e', 'sd_e', 'n_e', 'mean_c', 'sd_c', 'n_c'],
            'B': ['events_e', 'total_e', 'events_c', 'total_c'],
            'C': ['r', 'n']
        }
        
        self.define_effect_size_patterns()
        
        # 存储每个模式的匹配结果
        pattern_matches = {}
        
        for pattern, required_cols in pattern_requirements.items():
            matches = {}
            
            # 第一层：精确匹配
            for std_col in required_cols:
                matched = False
                
                # 优先使用自定义别名
                if self.CUSTOM_COLUMN_ALIASES and std_col in self.CUSTOM_COLUMN_ALIASES:
                    for alias in self.CUSTOM_COLUMN_ALIASES[std_col]:
                        if alias in df.columns:
                            matches[std_col] = alias
                            matched = True
                            break
                
                # 使用内置精确匹配
                if not matched and std_col in self.EXACT_MATCH_ALIASES:
                    for alias in self.EXACT_MATCH_ALIASES[std_col]:
                        if alias in df.columns:
                            matches[std_col] = alias
                            matched = True
                            break
                
                # 第二层：启发式匹配
                if not matched and std_col in self.HEURISTIC_KEYWORDS:
                    best_score = 0
                    best_match = None
                    
                    for col in df.columns:
                        if col not in matches.values():  # 避免重复匹配
                            score = self.calculate_heuristic_score(col, std_col)
                            if score > best_score and score >= 10:  # 阈值
                                best_score = score
                                best_match = col
                    
                    if best_match:
                        matches[std_col] = best_match
                        matched = True
                
                if not matched:
                    break  # 如果有任何必需列未找到，该模式失败
            
            # 检查是否所有必需列都找到了匹配
            if len(matches) == len(required_cols):
                pattern_matches[pattern] = matches
                
        # 最终裁决
        if len(pattern_matches) == 1:
            detected_pattern = list(pattern_matches.keys())[0]
            self.column_mapping = pattern_matches[detected_pattern]
            self.logger.info(f"检测到Effect Size模式: {detected_pattern}")
            self.logger.info(f"列映射: {self.column_mapping}")
            return detected_pattern
        elif len(pattern_matches) == 0:
            self.logger.error("未检测到任何有效的Effect Size模式")
            raise ValueError("未检测到任何有效的Effect Size模式，请使用FORCED_EFFECT_SIZE_PATTERN或CUSTOM_COLUMN_ALIASES进行手动指定")
        else:
            self.logger.error(f"检测到多个Effect Size模式: {list(pattern_matches.keys())}")
            raise ValueError("检测到多个Effect Size模式，请使用FORCED_EFFECT_SIZE_PATTERN进行手动指定")
            
    def calculate_heuristic_score(self, column_name: str, standard_column: str) -> int:
        """计算启发式匹配分数"""
        if standard_column not in self.HEURISTIC_KEYWORDS:
            return 0
            
        keywords = self.HEURISTIC_KEYWORDS[standard_column]
        score = 0
        
        # 分词处理
        col_tokens = re.split(r'[_\-\s]+', column_name.lower())
        
        # 核心关键词匹配
        for core_word in keywords['core']:
            if core_word in col_tokens:
                score += 10
                
        # 上下文关键词匹配
        for context_word in keywords['context']:
            if context_word in col_tokens:
                score += 5
                
        return score
        
    def detect_cluster_variable(self, df: pd.DataFrame) -> str:
        """1.4 依赖结构解析"""
        self.logger.info("开始聚类变量检测")
        
        # 检查强制覆盖
        if self.FORCED_CLUSTER_VARIABLE:
            if self.FORCED_CLUSTER_VARIABLE in df.columns:
                self.logger.info(f"使用强制指定的聚类变量: {self.FORCED_CLUSTER_VARIABLE}")
                return self.FORCED_CLUSTER_VARIABLE
            else:
                self.logger.error(f"强制指定的聚类变量不存在: {self.FORCED_CLUSTER_VARIABLE}")
                raise ValueError(f"强制指定的聚类变量不存在: {self.FORCED_CLUSTER_VARIABLE}")
        
        # 候选变量列表
        candidates = []
        
        # 优先使用自定义候选
        if self.CUSTOM_CLUSTER_CANDIDATES:
            candidates.extend(self.CUSTOM_CLUSTER_CANDIDATES)
            
        # 添加内置候选
        builtin_candidates = ['study_id', 'paper_id', 'author_year', 'authors', 'author', 'study']
        candidates.extend(builtin_candidates)
        
        # 检查每个候选变量
        for candidate in candidates:
            if candidate in df.columns:
                # 计算重复率
                unique_count = df[candidate].nunique()
                total_count = len(df)
                repeat_rate = 1 - (unique_count / total_count)
                
                self.logger.info(f"候选变量 {candidate}: 重复率 = {repeat_rate:.2%}")
                
                if 0.1 <= repeat_rate <= 0.95:  # 重复率在10%-95%之间
                    self.logger.info(f"选择聚类变量: {candidate}")
                    return candidate
                    
        # 如果没有找到合适的聚类变量，使用行索引
        self.logger.warning("未找到合适的聚类变量，将使用行索引")
        df['_row_index'] = range(len(df))
        return '_row_index'
        
    def data_integrity_protocol(self, df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        """阶段2：数据完整性协议"""
        self.logger.info("开始阶段2：数据完整性协议")
        
        # 创建工作副本
        df_work = df.copy()
        
        # 2.1 强制类型验证与转换
        self.logger.info("执行强制类型验证与转换")
        
        numeric_columns = []
        if pattern == 'A':
            numeric_columns = ['mean_e', 'sd_e', 'n_e', 'mean_c', 'sd_c', 'n_c']
        elif pattern == 'B':
            numeric_columns = ['events_e', 'total_e', 'events_c', 'total_c']
        elif pattern == 'C':
            numeric_columns = ['r', 'n']
            
        type_conversion_count = 0
        for std_col in numeric_columns:
            if std_col in self.column_mapping:
                actual_col = self.column_mapping[std_col]
                original_non_numeric = pd.to_numeric(df_work[actual_col], errors='coerce').isna().sum() - df_work[actual_col].isna().sum()
                df_work[actual_col] = pd.to_numeric(df_work[actual_col], errors='coerce')
                type_conversion_count += original_non_numeric
                
        self.logger.info(f"类型转换处理了 {type_conversion_count} 个非数字单元格")
        
        # 2.2 非破坏性质量审计
        self.logger.info("执行数据质量审计")
        
        df_work['qa_status'] = 'OK'
        df_work['continuity_correction_applied'] = False
        
        for idx, row in df_work.iterrows():
            exclude_reason = None
            
            if pattern == 'A':
                # 检查连续变量模式的数据质量
                required_cols = ['mean_e', 'sd_e', 'n_e', 'mean_c', 'sd_c', 'n_c']
                
                for std_col in required_cols:
                    actual_col = self.column_mapping[std_col]
                    value = row[actual_col]
                    
                    if pd.isna(value):
                        exclude_reason = f'Exclude: Missing Value ({std_col})'
                        break
                        
                # 检查Sample Size
                if not exclude_reason:
                    n_e_val = row[self.column_mapping['n_e']]
                    n_c_val = row[self.column_mapping['n_c']]
                    if n_e_val <= 1 or n_c_val <= 1:
                        exclude_reason = 'Exclude: n <= 1'
                        
                # 检查标准差
                if not exclude_reason:
                    sd_e_val = row[self.column_mapping['sd_e']]
                    sd_c_val = row[self.column_mapping['sd_c']]
                    if sd_e_val <= 0 or sd_c_val <= 0:
                        exclude_reason = 'Exclude: SD <= 0'
                        
            elif pattern == 'B':
                # 检查二元变量模式的数据质量
                required_cols = ['events_e', 'total_e', 'events_c', 'total_c']
                
                for std_col in required_cols:
                    actual_col = self.column_mapping[std_col]
                    value = row[actual_col]
                    
                    if pd.isna(value):
                        exclude_reason = f'Exclude: Missing Value ({std_col})'
                        break
                        
                # 检查事件数是否超过总数
                if not exclude_reason:
                    events_e = row[self.column_mapping['events_e']]
                    total_e = row[self.column_mapping['total_e']]
                    events_c = row[self.column_mapping['events_c']]
                    total_c = row[self.column_mapping['total_c']]
                    
                    if events_e > total_e or events_c > total_c:
                        exclude_reason = 'Exclude: Events > Total'
                    elif total_e <= 1 or total_c <= 1:
                        exclude_reason = 'Exclude: n <= 1'
                    elif events_e == 0 or events_c == 0 or events_e == total_e or events_c == total_c:
                        # 零事件处理，应用连续性校正
                        df_work.loc[idx, 'continuity_correction_applied'] = True
                        
            elif pattern == 'C':
                # 检查相关系数模式的数据质量
                r_val = row[self.column_mapping['r']]
                n_val = row[self.column_mapping['n']]
                
                if pd.isna(r_val) or pd.isna(n_val):
                    exclude_reason = 'Exclude: Missing Value'
                elif not (-1 <= r_val <= 1):
                    exclude_reason = 'Exclude: Invalid r'
                elif n_val <= 1:
                    exclude_reason = 'Exclude: n <= 1'
                    
            # 更新状态并记录审计日志
            if exclude_reason:
                df_work.loc[idx, 'qa_status'] = exclude_reason
                self.audit_log.append({
                    'row_index': idx,
                    'reason': exclude_reason,
                    'data': dict(row)
                })
                
        # 统计质量审计结果
        qa_summary = df_work['qa_status'].value_counts()
        self.logger.info(f"数据质量审计完成:")
        for status, count in qa_summary.items():
            self.logger.info(f"  {status}: {count}")
            
        return df_work
        
    def compute_effect_sizes(self, df: pd.DataFrame, pattern: str) -> pd.DataFrame:
        """阶段3：Effect Size计算引擎"""
        self.logger.info("开始阶段3：Effect Size计算引擎")
        
        # 3.1 创建安全计算视图
        df_valid = df[df['qa_status'] == 'OK'].copy()
        self.logger.info(f"有效数据行数: {len(df_valid)}")
        
        if len(df_valid) == 0:
            self.logger.error("没有有效数据进行Effect Size计算")
            raise ValueError("没有有效数据进行Effect Size计算")
            
        # 3.2 Effect Size计算
        effect_sizes = []
        variances = []
        
        for idx, row in df_valid.iterrows():
            try:
                if pattern == 'A':
                    # Cohen's d 计算
                    mean_e = row[self.column_mapping['mean_e']]
                    sd_e = row[self.column_mapping['sd_e']]
                    n_e = int(row[self.column_mapping['n_e']])
                    mean_c = row[self.column_mapping['mean_c']]
                    sd_c = row[self.column_mapping['sd_c']]
                    n_c = int(row[self.column_mapping['n_c']])
                    
                    # 手动计算Cohen's d
                    pooled_sd = np.sqrt(((n_e - 1) * sd_e**2 + (n_c - 1) * sd_c**2) / (n_e + n_c - 2))
                    es = (mean_e - mean_c) / pooled_sd
                    
                    # 计算方差
                    pooled_sd = np.sqrt(((n_e - 1) * sd_e**2 + (n_c - 1) * sd_c**2) / (n_e + n_c - 2))
                    v = ((n_e + n_c) / (n_e * n_c)) + (es**2 / (2 * (n_e + n_c)))
                    
                elif pattern == 'B':
                    # Odds Ratio 计算
                    events_e = int(row[self.column_mapping['events_e']])
                    total_e = int(row[self.column_mapping['total_e']])
                    events_c = int(row[self.column_mapping['events_c']])
                    total_c = int(row[self.column_mapping['total_c']])
                    
                    # 连续性校正
                    if row['continuity_correction_applied']:
                        events_e += 0.5
                        events_c += 0.5
                        total_e += 0.5
                        total_c += 0.5
                        
                    # 计算对数几率比
                    non_events_e = total_e - events_e
                    non_events_c = total_c - events_c
                    
                    es = np.log((events_e * non_events_c) / (non_events_e * events_c))
                    v = (1/events_e) + (1/non_events_e) + (1/events_c) + (1/non_events_c)
                    
                elif pattern == 'C':
                    # Fisher's Z 变换
                    r = row[self.column_mapping['r']]
                    n = int(row[self.column_mapping['n']])
                    
                    # Fisher's Z 变换
                    es = 0.5 * np.log((1 + r) / (1 - r))
                    v = 1 / (n - 3)
                    
                effect_sizes.append(es)
                variances.append(v)
                
            except Exception as e:
                self.logger.warning(f"行 {idx} Effect Size计算失败: {str(e)}")
                effect_sizes.append(np.nan)
                variances.append(np.nan)
                
        # 3.3 数据扩充
        df_result = df.copy()
        df_result['es'] = np.nan
        df_result['v'] = np.nan
        df_result['se'] = np.nan
        df_result['es_ci_lower'] = np.nan
        df_result['es_ci_upper'] = np.nan
        
        # 将计算结果合并回完整数据框
        valid_indices = df_valid.index
        df_result.loc[valid_indices, 'es'] = effect_sizes
        df_result.loc[valid_indices, 'v'] = variances
        
        # 计算Standard Error和置信区间
        df_result['se'] = np.sqrt(df_result['v'])
        df_result['es_ci_lower'] = df_result['es'] - 1.96 * df_result['se']
        df_result['es_ci_upper'] = df_result['es'] + 1.96 * df_result['se']
        
        self.logger.info(f"成功计算 {len(valid_indices)} 个Effect Size")
        
        return df_result
        
    def save_audit_log(self):
        """保存审计日志"""
        with open(self.OUTPUT_AUDIT_LOG_FILENAME, 'w', encoding='utf-8-sig') as f:
            f.write("元分析数据质量审计日志\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总排除记录数: {len(self.audit_log)}\n\n")
            
            for i, log_entry in enumerate(self.audit_log, 1):
                f.write(f"记录 {i}:\n")
                f.write(f"  行索引: {log_entry['row_index']}\n")
                f.write(f"  排除原因: {log_entry['reason']}\n")
                f.write(f"  相关数据: {log_entry['data']}\n")
                f.write("-" * 30 + "\n")
                
    def create_visualization(self, df: pd.DataFrame):
        """创建Effect Size Distribution可视化"""
        self.logger.info("创建Effect Size Distribution可视化")
        
        # 获取有效的Effect Size数据
        valid_es = df[df['qa_status'] == 'OK']['es'].dropna()
        
        if len(valid_es) == 0:
            self.logger.warning("没有有效的Effect Size数据用于可视化")
            return
            
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 主直方图
        plt.subplot(2, 2, 1)
        plt.hist(valid_es, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Effect Size Distribution直方图', fontsize=14, fontweight='bold')
        plt.xlabel('Effect Size (Effect Size)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 箱线图
        plt.subplot(2, 2, 2)
        plt.boxplot(valid_es, vert=True)
        plt.title('Effect Size箱线图', fontsize=14, fontweight='bold')
        plt.ylabel('Effect Size (Effect Size)')
        plt.grid(True, alpha=0.3)
        
        # Q-Q图
        plt.subplot(2, 2, 3)
        stats.probplot(valid_es, dist="norm", plot=plt)
        plt.title('Effect Size正态性Q-Q图', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 描述性统计文本
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""
        描述性统计:
        
        Sample Size: {len(valid_es)}
        均值: {valid_es.mean():.4f}
        标准差: {valid_es.std():.4f}
        最小值: {valid_es.min():.4f}
        最大值: {valid_es.max():.4f}
        中位数: {valid_es.median():.4f}
        
        分位数:
        25%: {valid_es.quantile(0.25):.4f}
        75%: {valid_es.quantile(0.75):.4f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.OUTPUT_VISUALIZATION_FILENAME, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Effect Size Distribution图已保存: {self.OUTPUT_VISUALIZATION_FILENAME}")
        
    def generate_comprehensive_report(self, df: pd.DataFrame, pattern: str, cluster_var: str):
        """阶段4：生成综合分析报告"""
        self.logger.info("开始阶段4：生成综合分析报告")
        
        # 计算统计信息
        total_studies = len(df)
        valid_studies = len(df[df['qa_status'] == 'OK'])
        excluded_studies = total_studies - valid_studies
        
        qa_summary = df['qa_status'].value_counts()
        valid_es = df[df['qa_status'] == 'OK']['es'].dropna()
        
        # 生成报告
        report_content = f"""# 元分析Effect Size计算综合分析报告 V1.4

## 执行摘要

本报告基于Effect Size计算自动化工作流构建协议 V1.4.1 生成，对元分析数据进行了全面的预处理和Effect Size计算。

### 分析概览
- **原始研究数量**: {total_studies}
- **最终纳入研究数量**: {valid_studies}
- **排除研究数量**: {excluded_studies}
- **检测到的Effect Size模式**: {pattern}
- **聚类变量**: {cluster_var}
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 方法学详述

### 可复现性声明
- **随机种子**: {self.RANDOM_SEED}
- **软件版本**: Python 3.x, pandas, pingouin, numpy, matplotlib
- **编码格式**: {self.ENCODING}

### Effect Size模式检测
"""

        if self.FORCED_EFFECT_SIZE_PATTERN:
            report_content += f"Effect Size模式通过用户强制指定确定为: **{pattern}**\n\n"
        else:
            report_content += f"Effect Size模式通过自动检测确定为: **{pattern}**\n\n"
            
        report_content += f"**列映射关系**:\n"
        for std_col, actual_col in self.column_mapping.items():
            report_content += f"- {std_col} → {actual_col}\n"
        report_content += "\n"

        # 聚类变量检测
        if self.FORCED_CLUSTER_VARIABLE:
            report_content += f"**聚类变量检测**: 通过用户强制指定确定为 `{cluster_var}`\n\n"
        else:
            report_content += f"**聚类变量检测**: 通过内置优先级列表自动检测，最终选择 `{cluster_var}` 作为聚类变量\n\n"

        # 数据清洗协议
        report_content += f"""### 数据清洗协议

本分析执行了严格的数据完整性协议：

1. **强制类型转换**: 所有数值列都经过了 `pandas.to_numeric()` 处理，确保数据类型的一致性
2. **数据质量审计**: 应用了以下排除规则：
   - 缺失值检查
   - Sample Size验证 (n > 1)
   - 标准差验证 (SD > 0)
   - 相关系数范围验证 (-1 ≤ r ≤ 1)
   - 二元数据逻辑验证 (events ≤ total)

### Effect Size计算

**Effect Size指标**: """

        if pattern == 'A':
            report_content += "Cohen's d (标准化均值差)\n"
        elif pattern == 'B':
            report_content += "对数几率比 (Log Odds Ratio)\n"
        elif pattern == 'C':
            report_content += "Fisher's Z 变换相关系数\n"
            
        continuity_correction_count = df['continuity_correction_applied'].sum()
        if continuity_correction_count > 0:
            report_content += f"**连续性校正**: 应用于 {continuity_correction_count} 项研究\n"
        report_content += "\n"

        # 数据质量审计报告
        report_content += f"""## 数据质量审计报告

### 数据纳入流程图
```
原始数据集: {total_studies} 项研究
    ↓
经过数据完整性协议筛选
    ↓
排除: {excluded_studies} 项研究
    ↓
最终纳入: {valid_studies} 项研究
```

### 排除原因汇总表

| 状态 | 数量 | 百分比 |
|------|------|--------|
"""

        for status, count in qa_summary.items():
            percentage = (count / total_studies) * 100
            report_content += f"| {status} | {count} | {percentage:.1f}% |\n"

        report_content += f"""
### 详细审计日志
详细的逐行排除记录已保存至文件 `{self.OUTPUT_AUDIT_LOG_FILENAME}` 中，以供深入审查。

### 极端值审查
"""

        if len(valid_es) > 0:
            # 识别极端值 (超过均值±2个标准差)
            mean_es = valid_es.mean()
            std_es = valid_es.std()
            outliers = df[(df['qa_status'] == 'OK') & 
                         ((df['es'] > mean_es + 2*std_es) | (df['es'] < mean_es - 2*std_es))]
            
            if len(outliers) > 0:
                report_content += f"识别出 {len(outliers)} 个潜在极端值:\n\n"
                for idx, row in outliers.iterrows():
                    report_content += f"- 研究ID {idx}: ES = {row['es']:.4f} (95% CI: {row['es_ci_lower']:.4f}, {row['es_ci_upper']:.4f})\n"
            else:
                report_content += "未识别出显著的极端值。\n"
        else:
            report_content += "无有效数据进行极端值分析。\n"

        # 视觉诊断
        report_content += f"""
## 视觉诊断

### Effect Size Distribution
下图展示了最终纳入研究的Effect Size Distribution情况。该图已保存为 `{self.OUTPUT_VISUALIZATION_FILENAME}`。

![Effect Size Distribution图]({self.OUTPUT_VISUALIZATION_FILENAME})

"""

        # 计算结果概览
        if len(valid_es) > 0:
            report_content += f"""## 计算结果概览

### Effect Size描述性统计

| 统计量 | 值 |
|--------|-----|
| Sample Size | {len(valid_es)} |
| 均值 | {valid_es.mean():.4f} |
| 标准差 | {valid_es.std():.4f} |
| 最小值 | {valid_es.min():.4f} |
| 最大值 | {valid_es.max():.4f} |
| 中位数 | {valid_es.median():.4f} |
| 25%分位数 | {valid_es.quantile(0.25):.4f} |
| 75%分位数 | {valid_es.quantile(0.75):.4f} |

### 置信区间统计
"""
            valid_ci = df[df['qa_status'] == 'OK'][['es_ci_lower', 'es_ci_upper']].dropna()
            if len(valid_ci) > 0:
                report_content += f"""
- 平均置信区间宽度: {(valid_ci['es_ci_upper'] - valid_ci['es_ci_lower']).mean():.4f}
- 包含零的置信区间数量: {((valid_ci['es_ci_lower'] <= 0) & (valid_ci['es_ci_upper'] >= 0)).sum()}
"""
        else:
            report_content += "## 计算结果概览\n\n无有效数据进行统计分析。\n"

        # 复现性与配置清单
        report_content += f"""
## 复现性与配置清单

### 核心参数配置
```python
FILE_PATH = "{self.FILE_PATH}"
ENCODING = "{self.ENCODING}"
RANDOM_SEED = {self.RANDOM_SEED}
FORCED_EFFECT_SIZE_PATTERN = {self.FORCED_EFFECT_SIZE_PATTERN}
FORCED_CLUSTER_VARIABLE = {self.FORCED_CLUSTER_VARIABLE}
CUSTOM_COLUMN_ALIASES = {self.CUSTOM_COLUMN_ALIASES}
CUSTOM_CLUSTER_CANDIDATES = {self.CUSTOM_CLUSTER_CANDIDATES}
```

### 输出文件清单
- 预处理数据: `{self.OUTPUT_DATA_FILENAME}`
- 审计日志: `{self.OUTPUT_AUDIT_LOG_FILENAME}`
- 可视化图表: `{self.OUTPUT_VISUALIZATION_FILENAME}`
- 综合报告: `{self.OUTPUT_REPORT_FILENAME}`

### 软件环境
- Python版本: 3.x
- 主要依赖包: pandas, numpy, pingouin, matplotlib, seaborn, scipy

---

*本报告由元分析Effect Size计算自动化工作流协议 V1.4.1 自动生成*
"""

        # 保存报告
        with open(self.OUTPUT_REPORT_FILENAME, 'w', encoding='utf-8-sig') as f:
            f.write(report_content)
            
        self.logger.info(f"综合分析报告已生成: {self.OUTPUT_REPORT_FILENAME}")
        
    def run_workflow(self):
        """执行完整的工作流"""
        try:
            self.logger.info("开始执行元分析Effect Size计算自动化工作流 V1.4.1")
            
            # 阶段0：环境配置（已在初始化中完成）
            self.logger.info("阶段0：环境配置与用户意图解析 - 完成")
            
            # 阶段1：数据摄取与模式解析
            df = self.load_data()
            pattern = self.detect_effect_size_pattern(df)
            cluster_var = self.detect_cluster_variable(df)
            
            # 阶段2：数据完整性协议
            df_processed = self.data_integrity_protocol(df, pattern)
            
            # 阶段3：Effect Size计算引擎
            df_final = self.compute_effect_sizes(df_processed, pattern)
            
            # 保存预处理后的数据
            df_final.to_csv(self.OUTPUT_DATA_FILENAME, index=False, encoding='utf-8-sig')
            self.logger.info(f"预处理数据已保存: {self.OUTPUT_DATA_FILENAME}")
            
            # 保存审计日志
            self.save_audit_log()
            
            # 创建可视化
            self.create_visualization(df_final)
            
            # 阶段4：生成综合报告
            self.generate_comprehensive_report(df_final, pattern, cluster_var)
            
            self.logger.info("元分析Effect Size计算自动化工作流执行完成！")
            
            return df_final
            
        except Exception as e:
            self.logger.error(f"工作流执行失败: {str(e)}")
            raise

def main():
    """主函数"""
    print("=" * 60)
    print("元分析Effect Size计算自动化工作流 V1.4.1")
    print("Meta-Analysis Effect Size Calculation Automated Workflow")
    print("=" * 60)
    
    try:
        # 创建工作流实例
        workflow = MetaAnalysisWorkflow()
        
        # 执行工作流
        result_df = workflow.run_workflow()
        
        print("\n" + "=" * 60)
        print("工作流执行成功完成！")
        print("=" * 60)
        print(f"输出文件:")
        print(f"- 预处理数据: {workflow.OUTPUT_DATA_FILENAME}")
        print(f"- 审计日志: {workflow.OUTPUT_AUDIT_LOG_FILENAME}")
        print(f"- 可视化图表: {workflow.OUTPUT_VISUALIZATION_FILENAME}")
        print(f"- 综合报告: {workflow.OUTPUT_REPORT_FILENAME}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("工作流执行失败，请检查日志文件获取详细信息。")

if __name__ == "__main__":
    main()

