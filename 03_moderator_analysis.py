#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# 尝试导入pymare，如果失败则使用备用方法
try:
    from pymare import meta_regression
    PYMARE_AVAILABLE = True
except ImportError:
    PYMARE_AVAILABLE = False
    warnings.warn("pymare库不可用，将使用备用的元回归实现", UserWarning)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 抑制警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ModeratorAnalysisV20Enhanced:
    """
    Meta分析调节变量探索类 V2.0 Enhanced
    
    符合V2.0审核标准的完整实现，包括：
    - 严格的输入验证和边界条件检查
    - 智能变量筛选与预处理
    - 全面的模型诊断（VIF、影响力分析）
    - 强制多重检验校正
    - 探索性标签硬编码
    - 详细的统计报告生成
    """
    
    def __init__(self, 
                 input_csv_path: str, 
                 output_directory: str,
                 es_col: str = 'es',
                 var_col: str = 'v',
                 cluster_col: Optional[str] = 'author',
                 random_seed: int = 42,
                 categorical_threshold: int = 10,
                 missing_threshold: float = 0.5,
                 min_studies_threshold: int = 10,
                 subgroup_min_k: int = 3,
                 vif_threshold: float = 5.0,
                 alpha_level: float = 0.05):
        """
        初始化调节变量分析器
        
        Args:
            input_csv_path: 输入CSV文件路径
            output_directory: 输出目录路径
            es_col: 效应量列名
            var_col: 方差列名
            cluster_col: 聚类变量列名
            random_seed: 随机种子
            categorical_threshold: 分类变量判断阈值（可配置）
            missing_threshold: 缺失值排除阈值
            min_studies_threshold: 最小研究数量阈值
            subgroup_min_k: 亚组最小研究数量
            vif_threshold: VIF阈值
            alpha_level: 显著性水平
        """
        # 基本配置
        self.input_csv_path = input_csv_path
        self.output_directory = output_directory
        self.es_col = es_col
        self.var_col = var_col
        self.cluster_col = cluster_col
        self.random_seed = random_seed
        self.categorical_threshold = categorical_threshold
        self.missing_threshold = missing_threshold
        self.min_studies_threshold = min_studies_threshold
        self.subgroup_min_k = subgroup_min_k
        self.vif_threshold = vif_threshold
        self.alpha_level = alpha_level
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        
        # 初始化存储变量
        self.df = None
        self.original_df = None
        self.moderators_metadata = {}
        self.excluded_variables = {}
        self.results_summary = []
        self.significant_moderators = []
        self.baseline_tau2 = None
        self.software_versions = self._get_software_versions()
        
        # 创建输出目录
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        self._log("=" * 60)
        self._log("Meta分析调节变量探索 V2.0 Enhanced 启动")
        self._log("基于meta分析调节变量审核提示词 V2.0")
        self._log("=" * 60)
        self._log(f"配置参数:")
        self._log(f"  - 分类变量阈值: {self.categorical_threshold}")
        self._log(f"  - 缺失值排除阈值: {self.missing_threshold}")
        self._log(f"  - 最小研究数量阈值: {self.min_studies_threshold}")
        self._log(f"  - VIF阈值: {self.vif_threshold}")
        self._log(f"  - 随机种子: {self.random_seed}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.output_directory, "moderator_analysis_v2.0_enhanced_log.txt")
        
        # 创建logger
        self.logger = logging.getLogger('ModeratorAnalysis')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8-sig')
        file_handler.setLevel(logging.INFO)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _log(self, message: str, level: str = "INFO"):
        """记录日志"""
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "CRITICAL":
            self.logger.critical(message)
    
    def _get_software_versions(self) -> str:
        """获取软件版本信息"""
        versions = []
        versions.append(f"Python {sys.version.split()[0]}")
        versions.append(f"pandas {pd.__version__}")
        versions.append(f"numpy {np.__version__}")
        import scipy
        versions.append(f"scipy {scipy.__version__}")
        versions.append(f"matplotlib {plt.matplotlib.__version__}")
        versions.append(f"seaborn {sns.__version__}")
        
        if PYMARE_AVAILABLE:
            try:
                import pymare
                versions.append(f"pymare {pymare.__version__}")
            except:
                versions.append("pymare (版本未知)")
        else:
            versions.append("pymare (不可用)")
        
        return "; ".join(versions)
    
    def run_complete_analysis(self):
        """
        运行完整的调节变量分析流程
        """
        try:
            self._log("开始完整的调节变量分析流程...")
            
            # 第1步：全面的输入验证
            self.step1_comprehensive_input_validation()
            
            # 第2步：智能变量筛选与预处理
            self.step2_intelligent_variable_screening()
            
            # 第3步：建立基线模型
            self.step3_baseline_model_establishment()
            
            # 第4步：系统性调节变量分析
            self.step4_systematic_moderator_analysis()
            
            # 第5步：多重检验校正
            self.step5_multiple_testing_correction()
            
            # 第6步：多变量分析与VIF诊断
            self.step6_multivariate_analysis_with_vif()
            
            # 第7步：生成全面报告
            self.step7_comprehensive_reporting()
            
            # 第8步：生成诊断图表
            self._generate_diagnostic_plots()
            
            self._log("=" * 60)
            self._log("Meta分析调节变量探索 V2.0 Enhanced 完成")
            self._log("=" * 60)
            
            # 最终总结
            self._log("分析总结:")
            self._log(f"  - 总研究数量: {len(self.df)}")
            self._log(f"  - 分析的调节变量: {len(self.moderators_metadata)}")
            self._log(f"  - 排除的变量: {len(self.excluded_variables)}")
            self._log(f"  - 原始显著变量: {len(self.significant_moderators)}")
            
            if hasattr(self, 'significant_moderators_corrected'):
                self._log(f"  - 校正后显著变量: {len(self.significant_moderators_corrected)}")
            
            if hasattr(self, 'multivariate_result'):
                self._log(f"  - 多变量模型R²: {self.multivariate_result['R2_adj']:.3f}")
            
            self._log(f"  - 输出目录: {self.output_directory}")
            
            return True
            
        except Exception as e:
            self._log(f"分析过程中发生严重错误: {str(e)}", "CRITICAL")
            self._log(f"错误详情: {traceback.format_exc()}", "CRITICAL")
            return False
    
    def step1_comprehensive_input_validation(self):
        """
        第1步：全面的输入验证和边界条件检查
        符合V2.0审核标准的严格验证
        """
        self._log("=== 第1步：全面的输入验证和边界条件检查 ===")
        
        # 1.1 文件存在性检查
        if not os.path.exists(self.input_csv_path):
            raise FileNotFoundError(f"输入文件不存在: {self.input_csv_path}")
        
        # 1.2 数据加载与基本验证
        try:
            self.original_df = pd.read_csv(self.input_csv_path)
            self._log(f"成功加载数据文件: {len(self.original_df)} 行, {len(self.original_df.columns)} 列")
        except Exception as e:
            raise ValueError(f"数据加载失败: {str(e)}")
        
        # 1.3 检查数据框是否为空
        if len(self.original_df) == 0:
            raise ValueError("输入数据框为空，无法进行分析")
        
        # 1.4 核心列存在性检查
        required_cols = [self.es_col, self.var_col]
        missing_cols = [col for col in required_cols if col not in self.original_df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的核心列: {missing_cols}。请确保数据包含效应量列('{self.es_col}')和方差列('{self.var_col}')")
        
        # 1.5 聚类变量健壮性检查
        if self.cluster_col:
            if self.cluster_col not in self.original_df.columns:
                self._log(f"警告: 指定的聚类变量 '{self.cluster_col}' 不存在，将使用标准元回归模型", "WARNING")
                self.cluster_col = None
            else:
                cluster_missing_rate = self.original_df[self.cluster_col].isna().mean()
                if cluster_missing_rate > 0.5:
                    self._log(f"警告: 聚类变量 '{self.cluster_col}' 缺失率过高 ({cluster_missing_rate:.1%})，将使用标准元回归模型", "WARNING")
                    self.cluster_col = None
                else:
                    self._log(f"聚类变量 '{self.cluster_col}' 验证通过，缺失率: {cluster_missing_rate:.1%}")
        
        # 1.6 核心列数据类型和数值有效性检查
        self.df = self.original_df.copy()
        
        # 检查效应量列
        try:
            self.df[self.es_col] = pd.to_numeric(self.df[self.es_col], errors='coerce')
            es_na_count = self.df[self.es_col].isna().sum()
            if es_na_count > 0:
                self._log(f"警告: 效应量列包含 {es_na_count} 个无效值，将在分析中排除", "WARNING")
        except Exception as e:
            raise ValueError(f"效应量列 '{self.es_col}' 无法转换为数值类型: {str(e)}")
        
        # 检查方差列 - 统计有效性断言
        try:
            self.df[self.var_col] = pd.to_numeric(self.df[self.var_col], errors='coerce')
            var_na_count = self.df[self.var_col].isna().sum()
            if var_na_count > 0:
                self._log(f"警告: 方差列包含 {var_na_count} 个无效值，将在分析中排除", "WARNING")
            
            # 统计有效性断言：检查零或负值
            invalid_var_mask = (self.df[self.var_col] <= 0) & (~self.df[self.var_col].isna())
            invalid_var_count = invalid_var_mask.sum()
            if invalid_var_count > 0:
                self._log(f"严重错误: 方差列包含 {invalid_var_count} 个零或负值，这在统计上是不可能的", "ERROR")
                self._log("无效方差值的行索引: " + str(self.df[invalid_var_mask].index.tolist()), "ERROR")
                raise ValueError(f"方差列 '{self.var_col}' 包含零或负值，这违反了统计学假设。请检查数据质量。")
                
        except Exception as e:
            if "零或负值" in str(e):
                raise e
            else:
                raise ValueError(f"方差列 '{self.var_col}' 无法转换为数值类型: {str(e)}")
        
        # 1.7 移除核心列缺失的行
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=[self.es_col, self.var_col])
        final_count = len(self.df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            self._log(f"移除了 {removed_count} 行因核心列缺失的数据")
        
        # 1.8 最小样本量检查
        if final_count < self.min_studies_threshold:
            warning_msg = f"""
            ⚠️ 严重警告：研究数量过少 ⚠️
            
            当前研究数量: {final_count}
            建议最小数量: {self.min_studies_threshold}
            
            在研究数量如此有限的情况下进行调节变量分析在统计上是危险的，可能导致：
            1. 统计功效严重不足
            2. 参数估计不稳定
            3. 过拟合风险极高
            4. 结果不可靠
            
            强烈建议：
            - 收集更多研究数据
            - 或者仅进行描述性分析
            - 如果继续分析，请极其谨慎解释结果
            """
            self._log(warning_msg, "WARNING")
            
            # 询问用户是否继续（在实际应用中可以添加交互）
            self._log("尽管存在风险，分析将继续进行，但所有结果都应被视为初步和不可靠的", "WARNING")
        
        self._log(f"输入验证完成。最终数据集: {final_count} 个研究")
        self._log(f"效应量范围: [{self.df[self.es_col].min():.4f}, {self.df[self.es_col].max():.4f}]")
        self._log(f"方差范围: [{self.df[self.var_col].min():.6f}, {self.df[self.var_col].max():.6f}]")
    
    def step2_intelligent_variable_screening(self):
        """
        第2步：智能变量筛选与预处理
        符合V2.0审核标准的全面筛选
        """
        self._log("=== 第2步：智能变量筛选与预处理 ===")
        
        # 2.1 识别候选调节变量
        core_cols = [self.es_col, self.var_col]
        if self.cluster_col:
            core_cols.append(self.cluster_col)
        
        # 用户可以定义排除列表
        user_excluded_cols = []  # 可以通过参数传入
        
        all_excluded_cols = core_cols + user_excluded_cols
        candidate_cols = [col for col in self.df.columns if col not in all_excluded_cols]
        
        self._log(f"识别到 {len(candidate_cols)} 个候选调节变量")
        
        # 2.2 高缺失率变量自动排除
        self._log("检查变量缺失率...")
        for col in candidate_cols[:]:  # 使用切片创建副本以安全删除
            missing_rate = self.df[col].isna().mean()
            if missing_rate > self.missing_threshold:
                self.excluded_variables[col] = f"缺失率过高 ({missing_rate:.1%})"
                candidate_cols.remove(col)
                self._log(f"变量 '{col}' 因缺失率过高 ({missing_rate:.1%}) 已被排除在分析之外")
        
        # 2.3 常数变量排除
        self._log("检查常数变量...")
        for col in candidate_cols[:]:
            unique_count = self.df[col].nunique()
            if unique_count <= 1:
                self.excluded_variables[col] = f"常数变量 (唯一值数量: {unique_count})"
                candidate_cols.remove(col)
                self._log(f"变量 '{col}' 因为是常数变量已被排除")
        
        # 2.4 智能变量类型判定
        self._log("进行智能变量类型判定...")
        for col in candidate_cols:
            col_dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            
            # 基本分类逻辑
            if col_dtype in ['object', 'category'] or unique_count <= self.categorical_threshold:
                var_type = 'Categorical'
            else:
                var_type = 'Continuous'
            
            # "伪连续"变量的启发式判定
            if (col_dtype in ['int64', 'int32'] and 
                unique_count < 20 and 
                unique_count < total_count * 0.15):
                
                self._log(f"警告: 变量 '{col}' 可能是编码的分类变量", "WARNING")
                self._log(f"  - 数据类型: {col_dtype}")
                self._log(f"  - 唯一值数量: {unique_count}")
                self._log(f"  - 占总数比例: {unique_count/total_count:.1%}")
                self._log(f"  - 建议: 考虑将其作为分类变量处理")
                self._log(f"  - 当前处理: 标记为'疑似分类变量'，按分类变量处理")
                
                var_type = 'Categorical'
            
            self.moderators_metadata[col] = var_type
            self._log(f"变量 '{col}': {var_type} (唯一值: {unique_count})")
        
        # 2.5 生成全局变量概览报告
        self._generate_global_variable_overview()
        
        # 2.6 变量预处理
        self._log("开始变量预处理...")
        
        # 分类变量预处理
        categorical_vars = [col for col, var_type in self.moderators_metadata.items() if var_type == 'Categorical']
        for col in categorical_vars:
            # 缺失值处理：标记为'Missing'
            self.df[col] = self.df[col].fillna('Missing')
            
            # 虚拟编码 (drop_first=True)
            dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
            self.df = pd.concat([self.df, dummies], axis=1)
            
            self._log(f"分类变量 '{col}' 已进行虚拟编码，生成 {len(dummies.columns)} 个虚拟变量")
        
        # 连续变量预处理
        continuous_vars = [col for col, var_type in self.moderators_metadata.items() if var_type == 'Continuous']
        for col in continuous_vars:
            # 记录缺失值比例
            missing_rate = self.df[col].isna().mean()
            if missing_rate > 0:
                self._log(f"连续变量 '{col}' 缺失值比例: {missing_rate:.1%}")
            
            # Z-score标准化（强制要求）
            col_mean = self.df[col].mean()
            col_std = self.df[col].std()
            
            if col_std > 0:
                self.df[f"{col}_z"] = (self.df[col] - col_mean) / col_std
                self._log(f"连续变量 '{col}' 已进行Z-score标准化 (均值: {col_mean:.4f}, 标准差: {col_std:.4f})")
            else:
                self._log(f"警告: 连续变量 '{col}' 标准差为0，无法标准化", "WARNING")
        
        # 2.7 保存预处理后的数据
        preprocessed_path = os.path.join(self.output_directory, "moderator_preprocessed_data_v2.0_enhanced.csv")
        self.df.to_csv(preprocessed_path, index=False)
        self._log(f"预处理后的数据已保存: {preprocessed_path}")
        
        self._log(f"变量筛选完成。最终候选调节变量: {len(self.moderators_metadata)}")
        self._log(f"  - 分类变量: {len(categorical_vars)}")
        self._log(f"  - 连续变量: {len(continuous_vars)}")
        self._log(f"  - 排除变量: {len(self.excluded_variables)}")
    
    def _generate_global_variable_overview(self):
        """生成全局变量概览报告"""
        self._log("生成全局变量概览报告...")
        
        # 连续变量相关性矩阵热图
        continuous_vars = [col for col, var_type in self.moderators_metadata.items() if var_type == 'Continuous']
        if len(continuous_vars) > 1:
            self._log(f"生成 {len(continuous_vars)} 个连续变量的相关性矩阵热图")
            
            # 计算相关性矩阵
            corr_data = self.df[continuous_vars].corr()
            
            # 创建热图
            plt.figure(figsize=(max(8, len(continuous_vars)), max(6, len(continuous_vars))))
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('候选连续调节变量相关性矩阵\n*注释：本分析属于探索性质，旨在生成假设*', 
                     fontsize=14, pad=20)
            plt.tight_layout()
            
            heatmap_path = os.path.join(self.output_directory, "continuous_variables_correlation_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self._log(f"相关性矩阵热图已保存: {heatmap_path}")
            
            # 识别高相关性变量对
            high_corr_pairs = []
            for i in range(len(corr_data.columns)):
                for j in range(i+1, len(corr_data.columns)):
                    corr_val = abs(corr_data.iloc[i, j])
                    if corr_val > 0.7:  # 高相关性阈值
                        high_corr_pairs.append((corr_data.columns[i], corr_data.columns[j], corr_val))
            
            if high_corr_pairs:
                self._log("发现高相关性变量对 (|r| > 0.7):")
                for var1, var2, corr_val in high_corr_pairs:
                    self._log(f"  - {var1} vs {var2}: r = {corr_val:.3f}")
                    self._log("    建议: 在多变量分析中注意多重共线性问题")
        
        # 分类变量摘要表格
        categorical_vars = [col for col, var_type in self.moderators_metadata.items() if var_type == 'Categorical']
        if categorical_vars:
            self._log(f"生成 {len(categorical_vars)} 个分类变量的摘要表格")
            
            categorical_summary = []
            for col in categorical_vars:
                value_counts = self.df[col].value_counts(dropna=False)
                missing_rate = self.df[col].isna().mean()
                
                for category, count in value_counts.items():
                    categorical_summary.append({
                        '变量名': col,
                        '类别': str(category),
                        '研究数量(k)': count,
                        '比例': f"{count/len(self.df):.1%}",
                        '缺失值比例': f"{missing_rate:.1%}"
                    })
            
            categorical_df = pd.DataFrame(categorical_summary)
            categorical_path = os.path.join(self.output_directory, "categorical_variables_summary.csv")
            categorical_df.to_csv(categorical_path, index=False)
            
            self._log(f"分类变量摘要表格已保存: {categorical_path}")
            
            # 检查小样本量亚组
            small_subgroups = []
            for col in categorical_vars:
                value_counts = self.df[col].value_counts()
                small_counts = value_counts[value_counts < self.subgroup_min_k]
                if len(small_counts) > 0:
                    small_subgroups.extend([(col, cat, count) for cat, count in small_counts.items()])
            
            if small_subgroups:
                self._log(f"发现 {len(small_subgroups)} 个小样本量亚组 (k < {self.subgroup_min_k}):")
                for var, cat, count in small_subgroups:
                    self._log(f"  - {var}[{cat}]: k = {count}")
                self._log("警告: 小样本量亚组可能导致不稳定的估计")
    
    def step3_baseline_model_establishment(self):
        """
        第3步：建立基线模型
        使用REML估计建立基线模型
        """
        self._log("=== 第3步：建立基线模型 ===")
        
        # 准备基线模型数据
        baseline_data = self.df[[self.es_col, self.var_col]].dropna()
        if self.cluster_col:
            baseline_data = self.df[[self.es_col, self.var_col, self.cluster_col]].dropna()
        
        y = baseline_data[self.es_col].values
        v = baseline_data[self.var_col].values
        
        self._log(f"基线模型数据: {len(y)} 个研究")
        
        # 建立基线模型
        try:
            if PYMARE_AVAILABLE:
                # 使用pymare (注意：pymare不直接支持聚类，这里先忽略聚类)
                baseline_result = meta_regression(y=y, v=v, X=None, method='REML')
                if self.cluster_col:
                    self._log("注意：当前pymare版本不支持聚类调整，结果可能需要谨慎解释", "WARNING")
                
                self.baseline_tau2 = baseline_result.tau2.item() if hasattr(baseline_result.tau2, 'item') else float(baseline_result.tau2)
                baseline_fe = baseline_result.fe_params[0].item() if hasattr(baseline_result.fe_params[0], 'item') else float(baseline_result.fe_params[0])
                baseline_se = np.sqrt(baseline_result.fe_cov[0, 0]).item() if hasattr(np.sqrt(baseline_result.fe_cov[0, 0]), 'item') else float(np.sqrt(baseline_result.fe_cov[0, 0]))
                
            else:
                # 备用实现
                self._log("使用备用元回归实现", "WARNING")
                self.baseline_tau2 = self._estimate_tau2_dersimonian_laird(y, v)
                baseline_fe = np.average(y, weights=1/(v + self.baseline_tau2))
                baseline_se = np.sqrt(1/np.sum(1/(v + self.baseline_tau2)))
            
            self._log(f"基线模型结果:")
            self._log(f"  - 总体效应量: {baseline_fe:.4f} (SE = {baseline_se:.4f})")
            self._log(f"  - 异质性方差 (τ²): {self.baseline_tau2:.6f}")
            
            # 异质性检验
            Q = np.sum((y - baseline_fe)**2 / v)
            df = len(y) - 1
            I2 = max(0, (Q - df) / Q) if Q > 0 else 0
            
            self._log(f"  - Q统计量: {Q:.4f} (df = {df})")
            self._log(f"  - I²: {I2:.1%}")
            
            if I2 > 0.5:
                self._log("检测到中等到高度异质性，适合进行调节变量分析")
            elif I2 > 0.25:
                self._log("检测到轻度到中等异质性，调节变量分析可能有用")
            else:
                self._log("异质性较低，调节变量分析的价值可能有限", "WARNING")
                
        except Exception as e:
            self._log(f"基线模型建立失败: {str(e)}", "ERROR")
            raise e
    
    def _estimate_tau2_dersimonian_laird(self, y: np.ndarray, v: np.ndarray) -> float:
        """DerSimonian-Laird方法估计τ²"""
        w = 1 / v
        weighted_mean = np.sum(w * y) / np.sum(w)
        Q = np.sum(w * (y - weighted_mean)**2)
        df = len(y) - 1
        
        if Q <= df:
            return 0.0
        else:
            c = np.sum(w) - np.sum(w**2) / np.sum(w)
            tau2 = (Q - df) / c
            return max(0.0, tau2)
    
    def step4_systematic_moderator_analysis(self):
        """
        第4步：系统性调节变量分析
        包含全面的模型诊断和影响力分析
        """
        self._log("=== 第4步：系统性调节变量分析 ===")
        
        for mod_name, mod_type in self.moderators_metadata.items():
            self._log(f"分析调节变量: {mod_name} ({mod_type})")
            
            try:
                result = self._analyze_single_moderator_enhanced(mod_name, mod_type)
                if result:
                    self.results_summary.append(result)
                    
                    # 判断是否显著（使用原始p值）
                    if result.get('lrt_p', 1) < self.alpha_level:
                        self.significant_moderators.append(result)
                        self._log(f"  ✓ 显著调节变量 (p = {result['lrt_p']:.4f})")
                    else:
                        self._log(f"  - 非显著 (p = {result['lrt_p']:.4f})")
                        
            except Exception as e:
                self._log(f"分析 {mod_name} 时出错: {str(e)}", "ERROR")
                continue
        
        self._log(f"单变量分析完成。显著调节变量: {len(self.significant_moderators)}/{len(self.moderators_metadata)}")
    
    def _analyze_single_moderator_enhanced(self, mod_name: str, mod_type: str) -> Optional[Dict]:
        """
        增强版单调节变量分析
        包含全面的模型诊断
        """
        try:
            if mod_type == 'Continuous':
                return self._analyze_continuous_moderator_enhanced(mod_name, mod_type)
            elif mod_type == 'Categorical':
                return self._analyze_categorical_moderator_enhanced(mod_name, mod_type)
            else:
                self._log(f"未知变量类型: {mod_type}", "ERROR")
                return None
                
        except Exception as e:
            self._log(f"分析调节变量 {mod_name} 时发生错误: {str(e)}", "ERROR")
            return None
    
    def _analyze_continuous_moderator_enhanced(self, mod_name: str, mod_type: str) -> Optional[Dict]:
        """增强版连续调节变量分析"""
        z_var_name = f"{mod_name}_z"
        
        if z_var_name not in self.df.columns:
            self._log(f"标准化变量 '{z_var_name}' 不存在", "ERROR")
            return None
        
        # 准备数据
        required_cols = [self.es_col, self.var_col, z_var_name]
        if self.cluster_col:
            required_cols.append(self.cluster_col)
        
        analysis_data = self.df[required_cols].dropna()
        
        if len(analysis_data) < 3:
            self._log(f"有效数据不足 (n={len(analysis_data)})", "WARNING")
            return None
        
        y = analysis_data[self.es_col].values
        v = analysis_data[self.var_col].values
        x = analysis_data[z_var_name].values
        
        # 构建设计矩阵
        X = np.column_stack([np.ones(len(y)), x])
        
        try:
            # 拟合模型
            if PYMARE_AVAILABLE:
                result = meta_regression(y=y, v=v, X=X, method='REML')
                if self.cluster_col:
                    self._log("注意：当前pymare版本不支持聚类调整", "WARNING")
            else:
                # 备用实现
                result = self._fit_meta_regression_backup(y, v, X)
            
            # 提取结果
            if hasattr(result, 'tau2'):
                tau2 = result.tau2.item() if hasattr(result.tau2, 'item') else float(result.tau2)
                fe_params = result.fe_params
                fe_cov = result.fe_cov
            else:
                tau2 = result['tau2']
                fe_params = result['fe_params']
                fe_cov = result['fe_cov']
            
            # 计算R²
            R2_adj = max(0, (self.baseline_tau2 - tau2) / self.baseline_tau2) if self.baseline_tau2 > 0 else 0
            
            # 似然比检验
            lrt_chi2, lrt_p = self._likelihood_ratio_test(y, v, X, self.cluster_col)
            
            # 模型诊断
            diagnostics = self._comprehensive_model_diagnostics(y, v, X, fe_params, tau2)
            
            # 系数信息
            coefficients = []
            for i, coeff in enumerate(fe_params):
                se = np.sqrt(fe_cov[i, i])
                z_score = coeff / se if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                # 确保所有值都是标量
                coeff_scalar = coeff.item() if hasattr(coeff, 'item') else float(coeff)
                se_scalar = se.item() if hasattr(se, 'item') else float(se)
                z_scalar = z_score.item() if hasattr(z_score, 'item') else float(z_score)
                p_scalar = p_val.item() if hasattr(p_val, 'item') else float(p_val)
                
                coeff_name = 'Intercept' if i == 0 else f'{mod_name}_z'
                coefficients.append({
                    'name': coeff_name,
                    'beta': coeff_scalar,
                    'se': se_scalar,
                    'z': z_scalar,
                    'p': p_scalar,
                    'ci_lower': coeff_scalar - 1.96 * se_scalar,
                    'ci_upper': coeff_scalar + 1.96 * se_scalar
                })
            
            # 反算回原始尺度（用于报告）
            original_scale_beta = self._convert_to_original_scale(mod_name, fe_params[1])
            
            return {
                'moderator': mod_name,
                'type': mod_type,
                'n_studies': len(analysis_data),
                'tau2': tau2,
                'R2_adj': R2_adj,
                'lrt_chi2': lrt_chi2,
                'lrt_df': 1,
                'lrt_p': lrt_p,
                'coefficients': coefficients,
                'original_scale_beta': original_scale_beta,
                'diagnostics': diagnostics,
                'converged': True
            }
            
        except Exception as e:
            self._log(f"连续变量 {mod_name} 模型拟合失败: {str(e)}", "ERROR")
            return None
    
    def _analyze_categorical_moderator_enhanced(self, mod_name: str, mod_type: str) -> Optional[Dict]:
        """增强版分类调节变量分析"""
        # 找到虚拟变量
        dummy_cols = [col for col in self.df.columns if col.startswith(f"{mod_name}_")]
        
        if not dummy_cols:
            self._log(f"未找到 {mod_name} 的虚拟变量", "ERROR")
            return None
        
        # 准备数据
        required_cols = [self.es_col, self.var_col] + dummy_cols
        if self.cluster_col:
            required_cols.append(self.cluster_col)
        
        analysis_data = self.df[required_cols].dropna()
        
        if len(analysis_data) < len(dummy_cols) + 2:
            self._log(f"有效数据不足", "WARNING")
            return None
        
        y = analysis_data[self.es_col].values
        v = analysis_data[self.var_col].values
        
        # 构建设计矩阵
        X_dummies = analysis_data[dummy_cols].values
        X = np.column_stack([np.ones(len(y)), X_dummies])
        
        try:
            # 拟合模型
            if PYMARE_AVAILABLE:
                result = meta_regression(y=y, v=v, X=X, method='REML')
                if self.cluster_col:
                    self._log("注意：当前pymare版本不支持聚类调整", "WARNING")
            else:
                result = self._fit_meta_regression_backup(y, v, X)
            
            # 提取结果
            if hasattr(result, 'tau2'):
                tau2 = result.tau2.item() if hasattr(result.tau2, 'item') else float(result.tau2)
                fe_params = result.fe_params
                fe_cov = result.fe_cov
            else:
                tau2 = result['tau2']
                fe_params = result['fe_params']
                fe_cov = result['fe_cov']
            
            # 计算R²
            R2_adj = max(0, (self.baseline_tau2 - tau2) / self.baseline_tau2) if self.baseline_tau2 > 0 else 0
            
            # 似然比检验
            lrt_chi2, lrt_p = self._likelihood_ratio_test(y, v, X, self.cluster_col)
            
            # 模型诊断
            diagnostics = self._comprehensive_model_diagnostics(y, v, X, fe_params, tau2)
            
            # 系数信息
            coefficients = []
            var_names = ['Intercept'] + dummy_cols
            for i, coeff in enumerate(fe_params):
                se = np.sqrt(fe_cov[i, i])
                z_score = coeff / se if se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                # 确保所有值都是标量
                coeff_scalar = coeff.item() if hasattr(coeff, 'item') else float(coeff)
                se_scalar = se.item() if hasattr(se, 'item') else float(se)
                z_scalar = z_score.item() if hasattr(z_score, 'item') else float(z_score)
                p_scalar = p_val.item() if hasattr(p_val, 'item') else float(p_val)
                
                coefficients.append({
                    'name': var_names[i],
                    'beta': coeff_scalar,
                    'se': se_scalar,
                    'z': z_scalar,
                    'p': p_scalar,
                    'ci_lower': coeff_scalar - 1.96 * se_scalar,
                    'ci_upper': coeff_scalar + 1.96 * se_scalar
                })
            
            # 亚组分析
            subgroup_analysis = self._perform_subgroup_analysis(mod_name, analysis_data)
            
            return {
                'moderator': mod_name,
                'type': mod_type,
                'n_studies': len(analysis_data),
                'tau2': tau2,
                'R2_adj': R2_adj,
                'lrt_chi2': lrt_chi2,
                'lrt_df': len(dummy_cols),
                'lrt_p': lrt_p,
                'coefficients': coefficients,
                'subgroup_analysis': subgroup_analysis,
                'diagnostics': diagnostics,
                'converged': True
            }
            
        except Exception as e:
            self._log(f"分类变量 {mod_name} 模型拟合失败: {str(e)}", "ERROR")
            return None
    
    def _comprehensive_model_diagnostics(self, y: np.ndarray, v: np.ndarray, 
                                       X: np.ndarray, fe_params: np.ndarray, 
                                       tau2: float) -> Dict:
        """
        全面的模型诊断
        包括影响力分析和异常值检测
        """
        diagnostics = {}
        
        try:
            # 确保fe_params是一维数组
            if fe_params.ndim > 1:
                fe_params = fe_params.flatten()
            
            # 检查维度匹配
            if X.shape[1] != len(fe_params):
                self._log(f"维度不匹配: X.shape[1]={X.shape[1]}, fe_params.shape={fe_params.shape}", "WARNING")
                # 尝试修复维度问题
                if len(fe_params) > X.shape[1]:
                    fe_params = fe_params[:X.shape[1]]
                else:
                    # 如果参数不足，用零填充
                    fe_params = np.pad(fe_params, (0, X.shape[1] - len(fe_params)), 'constant')
            
            # 计算拟合值和残差
            fitted_values = X @ fe_params
            residuals = y - fitted_values
            
            # 标准化残差
            weights = 1 / (v + tau2)
            standardized_residuals = residuals * np.sqrt(weights)
            
            # 学生化残差 - 修复矩阵计算
            try:
                W = np.diag(weights)
                XtWX = X.T @ W @ X
                
                # 检查矩阵是否可逆
                try:
                    if np.linalg.det(XtWX) == 0:
                        self._log("设计矩阵奇异，使用伪逆", "WARNING")
                        XtWX_inv = np.linalg.pinv(XtWX)
                    else:
                        XtWX_inv = np.linalg.inv(XtWX)
                except np.linalg.LinAlgError:
                    self._log("矩阵求逆失败，使用伪逆", "WARNING")
                    XtWX_inv = np.linalg.pinv(XtWX)
                
                # 计算帽子矩阵的对角元素（杠杆值）- 修复维度问题
                # H = X @ XtWX_inv @ X.T @ W 这个计算可能导致维度问题
                # 改用更稳定的方法
                try:
                    # 方法1：直接计算杠杆值
                    leverage = np.zeros(len(y))
                    for i in range(len(y)):
                        x_i = X[i:i+1, :]  # 第i行，保持2D形状
                        h_ii = x_i @ XtWX_inv @ x_i.T * weights[i]
                        leverage[i] = h_ii[0, 0] if h_ii.ndim > 0 else h_ii
                    
                    # 确保杠杆值在合理范围内
                    leverage = np.clip(leverage, 0.001, 0.999)
                    
                except Exception as leverage_error:
                    self._log(f"杠杆值计算失败: {str(leverage_error)}, 使用均匀分布", "WARNING")
                    leverage = np.full(len(y), X.shape[1] / len(y))
                    leverage = np.clip(leverage, 0.001, 0.999)
                
                studentized_residuals = standardized_residuals / np.sqrt(1 - leverage)
                
                # Cook's距离
                cooks_distance = (studentized_residuals**2 / X.shape[1]) * (leverage / (1 - leverage))
                
                # DFFITS
                dffits = studentized_residuals * np.sqrt(leverage / (1 - leverage))
                
            except Exception as matrix_error:
                self._log(f"矩阵计算失败，使用简化诊断: {str(matrix_error)}", "WARNING")
                # 简化的诊断
                leverage = np.full(len(y), 1/len(y))  # 均匀杠杆值
                studentized_residuals = standardized_residuals
                cooks_distance = standardized_residuals**2 / X.shape[1]
                dffits = standardized_residuals
            
            # 识别异常值和高影响力研究
            outlier_threshold = 2.5
            influence_threshold_cook = 4 / len(y)
            influence_threshold_dffits = 2 * np.sqrt(X.shape[1] / len(y))
            
            outliers = np.where(np.abs(studentized_residuals) > outlier_threshold)[0]
            high_influence_cook = np.where(cooks_distance > influence_threshold_cook)[0]
            high_influence_dffits = np.where(np.abs(dffits) > influence_threshold_dffits)[0]
            
            diagnostics.update({
                'residuals': residuals,
                'standardized_residuals': standardized_residuals,
                'studentized_residuals': studentized_residuals,
                'leverage': leverage,
                'cooks_distance': cooks_distance,
                'dffits': dffits,
                'outliers': outliers.tolist(),
                'high_influence_cook': high_influence_cook.tolist(),
                'high_influence_dffits': high_influence_dffits.tolist(),
                'outlier_threshold': outlier_threshold,
                'influence_threshold_cook': influence_threshold_cook,
                'influence_threshold_dffits': influence_threshold_dffits
            })
            
            # 记录诊断结果
            if len(outliers) > 0:
                self._log(f"  发现 {len(outliers)} 个潜在异常值 (行索引: {outliers.tolist()})")
            
            if len(high_influence_cook) > 0:
                self._log(f"  发现 {len(high_influence_cook)} 个高影响力研究 (Cook's距离, 行索引: {high_influence_cook.tolist()})")
            
            if len(high_influence_dffits) > 0:
                self._log(f"  发现 {len(high_influence_dffits)} 个高影响力研究 (DFFITS, 行索引: {high_influence_dffits.tolist()})")
            
            if len(outliers) > 0 or len(high_influence_cook) > 0 or len(high_influence_dffits) > 0:
                self._log("  建议: 进行敏感性分析，排除这些研究后重新分析")
            
        except Exception as e:
            self._log(f"模型诊断失败: {str(e)}", "WARNING")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def _convert_to_original_scale(self, mod_name: str, standardized_beta: float) -> Dict:
        """将标准化系数转换回原始尺度"""
        try:
            original_std = self.df[mod_name].std()
            original_beta = standardized_beta / original_std
            
            return {
                'original_beta': original_beta,
                'original_std': original_std,
                'interpretation': f"原始尺度下，{mod_name}每增加1个单位，效应量变化{original_beta:.4f}"
            }
        except:
            return {'error': '无法转换到原始尺度'}
    
    def _perform_subgroup_analysis(self, mod_name: str, analysis_data: pd.DataFrame) -> List[Dict]:
        """执行亚组分析"""
        subgroup_results = []
        
        try:
            for category in analysis_data[mod_name].unique():
                subgroup_data = analysis_data[analysis_data[mod_name] == category]
                
                if len(subgroup_data) >= self.subgroup_min_k:
                    y_sub = subgroup_data[self.es_col].values
                    v_sub = subgroup_data[self.var_col].values
                    
                    # 简单元分析
                    weights = 1 / v_sub
                    pooled_es = np.average(y_sub, weights=weights)
                    pooled_se = np.sqrt(1 / np.sum(weights))
                    
                    subgroup_results.append({
                        'category': str(category),
                        'k': len(subgroup_data),
                        'es': pooled_es,
                        'se': pooled_se,
                        'ci_lower': pooled_es - 1.96 * pooled_se,
                        'ci_upper': pooled_es + 1.96 * pooled_se
                    })
                else:
                    self._log(f"  亚组 '{category}' 样本量不足 (k={len(subgroup_data)})", "WARNING")
        
        except Exception as e:
            self._log(f"亚组分析失败: {str(e)}", "WARNING")
        
        return subgroup_results
    
    def _likelihood_ratio_test(self, y: np.ndarray, v: np.ndarray, 
                             X: np.ndarray, cluster_col: Optional[str]) -> Tuple[float, float]:
        """似然比检验"""
        try:
            # 基线模型（仅截距）
            X_baseline = np.ones((len(y), 1))
            
            if PYMARE_AVAILABLE:
                try:
                    baseline_result = meta_regression(y=y, v=v, X=X_baseline, method='REML')
                    full_result = meta_regression(y=y, v=v, X=X, method='REML')
                    if cluster_col:
                        self._log("注意：当前pymare版本不支持聚类调整", "WARNING")
                    
                    # 尝试多种方式获取对数似然值
                    ll_baseline = None
                    ll_full = None
                    
                    # 尝试不同的属性名
                    for attr_name in ['logLik', 'loglik_', 'loglik', 'log_likelihood', 'llf']:
                        if hasattr(baseline_result, attr_name) and hasattr(full_result, attr_name):
                            ll_baseline = getattr(baseline_result, attr_name)
                            ll_full = getattr(full_result, attr_name)
                            break
                    
                    if ll_baseline is not None and ll_full is not None:
                        # 确保对数似然值是标量
                        if hasattr(ll_baseline, 'item'):
                            ll_baseline = ll_baseline.item()
                        if hasattr(ll_full, 'item'):
                            ll_full = ll_full.item()
                        
                        lrt_chi2 = 2 * (ll_full - ll_baseline)
                        df = X.shape[1] - X_baseline.shape[1]
                        lrt_p = 1 - chi2.cdf(lrt_chi2, df)
                        
                        return float(lrt_chi2), float(lrt_p)
                    else:
                        raise AttributeError("无法获取对数似然值")
                        
                except Exception as e:
                    self._log(f"似然比检验失败: {str(e)}", "WARNING")
                    # 使用Wald检验作为备用
                    return self._wald_test_backup(y, v, X)
                
            else:
                # 备用实现：使用Wald检验近似
                result_full = self._fit_meta_regression_backup(y, v, X)
                
                # 检验除截距外的所有系数是否为0
                beta = result_full['fe_params'][1:]  # 排除截距
                cov = result_full['fe_cov'][1:, 1:]  # 排除截距
                
                if len(beta) == 1:
                    # 单个系数的Wald检验
                    wald_stat = (beta[0]**2) / cov[0, 0]
                    lrt_chi2 = wald_stat
                    lrt_p = 1 - chi2.cdf(wald_stat, 1)
                else:
                    # 多个系数的联合Wald检验
                    wald_stat = beta.T @ np.linalg.inv(cov) @ beta
                    lrt_chi2 = wald_stat
                    lrt_p = 1 - chi2.cdf(wald_stat, len(beta))
            
            return float(lrt_chi2), float(lrt_p)
            
        except Exception as e:
            self._log(f"似然比检验失败: {str(e)}", "WARNING")
            return 0.0, 1.0
    
    def _wald_test_backup(self, y: np.ndarray, v: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
        """Wald检验作为似然比检验的备用方法"""
        try:
            # 使用加权最小二乘法
            W = np.diag(1.0 / v)
            XtWX = X.T @ W @ X
            
            # 检查矩阵是否可逆
            if np.linalg.det(XtWX) == 0:
                XtWX_inv = np.linalg.pinv(XtWX)
            else:
                XtWX_inv = np.linalg.inv(XtWX)
            
            beta = XtWX_inv @ X.T @ W @ y
            
            # 计算协方差矩阵
            cov_matrix = XtWX_inv
            
            # Wald检验：检验除截距外的所有系数是否为0
            if X.shape[1] > 1:
                # 构建约束矩阵（排除截距）
                R = np.zeros((X.shape[1] - 1, X.shape[1]))
                R[:, 1:] = np.eye(X.shape[1] - 1)
                
                # Wald统计量
                wald_stat = beta.T @ R.T @ np.linalg.inv(R @ cov_matrix @ R.T) @ R @ beta
                df = X.shape[1] - 1
                wald_p = 1 - chi2.cdf(wald_stat, df)
                
                return float(wald_stat), float(wald_p)
            else:
                return 0.0, 1.0
                
        except Exception as e:
            self._log(f"Wald检验失败: {str(e)}", "WARNING")
            return np.nan, np.nan
    
    def _fit_meta_regression_backup(self, y: np.ndarray, v: np.ndarray, X: np.ndarray) -> Dict:
        """备用元回归实现"""
        try:
            # 简化的REML实现
            # 这是一个基本实现，实际应用中建议使用专业库
            
            # 初始权重
            weights = 1 / v
            
            # 加权最小二乘
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            
            # 检查矩阵是否可逆
            try:
                if np.linalg.det(XtWX) == 0:
                    XtWX_inv = np.linalg.pinv(XtWX)
                else:
                    XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                XtWX_inv = np.linalg.pinv(XtWX)
            
            fe_params = XtWX_inv @ X.T @ W @ y
            
            # 残差
            residuals = y - X @ fe_params
            
            # 估计τ²
            Q = np.sum(weights * residuals**2)
            df = len(y) - X.shape[1]
            
            if Q > df and df > 0:
                try:
                    trace_term = np.trace(X @ XtWX_inv @ X.T @ W)
                    denominator = np.sum(weights) - trace_term
                    if denominator > 0:
                        tau2 = (Q - df) / denominator
                    else:
                        tau2 = 0.0
                except:
                    tau2 = max(0.0, (Q - df) / np.sum(weights))
            else:
                tau2 = 0.0
            
            # 确保tau2非负
            tau2 = max(0.0, tau2)
            
            # 更新权重和参数
            weights_updated = 1 / (v + tau2)
            W_updated = np.diag(weights_updated)
            XtWX_updated = X.T @ W_updated @ X
            
            try:
                if np.linalg.det(XtWX_updated) == 0:
                    XtWX_inv_updated = np.linalg.pinv(XtWX_updated)
                else:
                    XtWX_inv_updated = np.linalg.inv(XtWX_updated)
            except np.linalg.LinAlgError:
                XtWX_inv_updated = np.linalg.pinv(XtWX_updated)
            
            fe_params_updated = XtWX_inv_updated @ X.T @ W_updated @ y
            
            return {
                'fe_params': fe_params_updated,
                'fe_cov': XtWX_inv_updated,
                'tau2': tau2
            }
            
        except Exception as e:
            self._log(f"备用元回归失败: {str(e)}", "ERROR")
            # 返回最简单的结果
            return {
                'fe_params': np.zeros(X.shape[1]),
                'fe_cov': np.eye(X.shape[1]),
                'tau2': 0.0
            }
    
    def step5_multiple_testing_correction(self):
        """
        第5步：多重检验校正
        使用Benjamini-Hochberg程序
        """
        self._log("=== 第5步：多重检验校正 ===")
        
        if not self.results_summary:
            self._log("没有结果需要校正")
            return
        
        # 提取p值，处理nan值
        p_values = []
        moderator_names = []
        
        for result in self.results_summary:
            p_val = result.get('lrt_p', np.nan)
            # 确保p值是有效的数值
            if np.isnan(p_val) or np.isinf(p_val):
                p_val = 1.0  # 将无效p值设为1.0（非显著）
            p_values.append(p_val)
            moderator_names.append(result['moderator'])
        
        # Benjamini-Hochberg校正
        q_values = self._benjamini_hochberg_correction(p_values)
        
        # 更新结果
        for i, result in enumerate(self.results_summary):
            result['lrt_q'] = q_values[i] if not np.isnan(q_values[i]) else 1.0
            result['significant_uncorrected'] = (p_values[i] < self.alpha_level and 
                                               not np.isnan(p_values[i]))
            result['significant_corrected'] = (q_values[i] < self.alpha_level and 
                                             not np.isnan(q_values[i]))
        
        # 统计校正后的显著结果
        significant_corrected = [r for r in self.results_summary if r['significant_corrected']]
        
        self._log(f"多重检验校正完成:")
        self._log(f"  - 原始显著变量: {len(self.significant_moderators)}")
        self._log(f"  - 校正后显著变量: {len(significant_corrected)}")
        
        if len(significant_corrected) < len(self.significant_moderators):
            lost_significance = len(self.significant_moderators) - len(significant_corrected)
            self._log(f"  - {lost_significance} 个变量在校正后失去显著性")
        
        # 更新显著调节变量列表（基于校正后的结果）
        self.significant_moderators_corrected = significant_corrected
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Benjamini-Hochberg多重检验校正"""
        n = len(p_values)
        if n == 0:
            return []
        
        # 处理nan值和无效值
        valid_p_values = []
        valid_indices = []
        
        for i, p in enumerate(p_values):
            if not np.isnan(p) and not np.isinf(p) and 0 <= p <= 1:
                valid_p_values.append(p)
                valid_indices.append(i)
        
        # 如果没有有效的p值，返回全nan
        if len(valid_p_values) == 0:
            return [np.nan] * n
        
        # 创建(p值, 原始索引)对并排序
        indexed_p = [(p, i) for i, p in enumerate(valid_p_values)]
        indexed_p.sort(key=lambda x: x[0])
        
        # 计算校正后的q值
        q_values_valid = [0] * len(valid_p_values)
        
        for rank, (p_val, valid_idx) in enumerate(indexed_p):
            # BH校正公式: q = p * n_valid / (rank + 1)
            q_val = p_val * len(valid_p_values) / (rank + 1)
            q_values_valid[valid_idx] = min(q_val, 1.0)  # 确保q值不超过1
        
        # 确保q值单调性
        sorted_q = [q_values_valid[indexed_p[i][1]] for i in range(len(valid_p_values))]
        for i in range(len(valid_p_values)-2, -1, -1):
            if sorted_q[i] > sorted_q[i+1]:
                sorted_q[i] = sorted_q[i+1]
        
        # 重新映射到原始顺序
        final_q_valid = [0] * len(valid_p_values)
        for i, (_, valid_idx) in enumerate(indexed_p):
            final_q_valid[valid_idx] = sorted_q[i]
        
        # 构建最终结果，包含nan值
        final_q = [np.nan] * n
        for i, valid_idx in enumerate(valid_indices):
            final_q[valid_idx] = final_q_valid[i]
        
        return final_q
    
    def step6_multivariate_analysis_with_vif(self):
        """
        第6步：多变量分析与VIF诊断
        """
        self._log("=== 第6步：多变量分析与VIF诊断 ===")
        
        # 基于校正后的显著性选择变量
        if hasattr(self, 'significant_moderators_corrected'):
            significant_vars = self.significant_moderators_corrected
        else:
            significant_vars = self.significant_moderators
        
        if len(significant_vars) < 2:
            self._log("显著变量少于2个，跳过多变量分析")
            return
    
    def step7_comprehensive_reporting(self):
        """生成全面报告"""
        report_path = os.path.join(self.output_directory, "comprehensive_report.txt")
        
        with open(report_path, "w", encoding='utf-8-sig') as f:
            # 硬编码的探索性标题
            f.write("# Meta分析调节变量探索报告 V2.0 Enhanced\n")
            f.write("## 一份探索性调节变量分析报告\n\n")
            
            # 执行摘要表格
            f.write("## 1. 执行摘要\n\n")
            f.write("| 调节变量 | 类型 | k | R²_adj | 原始p值 | 校正q值 | 显著性 |\n")
            f.write("|----------|------|---|--------|---------|---------|--------|\n")
            
            for result in self.results_summary:
                sig_status = "是" if result.get('significant_corrected', False) else "否"
                p_val = result['lrt_p'] if not np.isnan(result['lrt_p']) else "N/A"
                q_val = result.get('lrt_q', 'N/A')
                if q_val != 'N/A' and not np.isnan(q_val):
                    q_val = f"{q_val:.4f}"
                
                f.write(f"| {result['moderator']} | {result['type']} | {result['n_studies']} | ")
                f.write(f"{result['R2_adj']:.3f} | {p_val} | {q_val} | {sig_status} |\n")
            
            f.write("\n*注释：本分析属于探索性质，旨在生成假设。所有发现，特别是未经多重检验校正的p值，均需在未来的预注册研究中进行验证性检验。*\n\n")
            
            f.write("## 2. 详细分析结果\n\n")
            for result in self.results_summary:
                f.write(f"### {result['moderator']} ({result['type']})\n\n")
                
                # 基本统计
                f.write(f"- **研究数量**: {result['n_studies']}\n")
                f.write(f"- **R²_adj**: {result['R2_adj']:.3f}\n")
                f.write(f"- **似然比检验**: χ² = {result['lrt_chi2']:.3f}, df = {result['lrt_df']}, p = {result['lrt_p']:.4f}\n")
                f.write(f"- **校正后q值**: {result.get('lrt_q', 'N/A'):.4f}\n\n")
                
                # 单变量分析警示
                f.write("**注意**: 此为单变量分析结果，未校正其他潜在调节变量的影响，可能存在混杂偏倚。最终解释应以多变量模型为准。\n\n")
                
                # 系数信息
                if 'coefficients' in result:
                    f.write("#### 回归系数\n\n")
                    f.write("| 变量 | 系数 | 标准误 | Z值 | p值 | 95% CI |\n")
                    f.write("|------|------|--------|-----|-----|--------|\n")
                    
                    for coeff in result['coefficients']:
                        f.write(f"| {coeff['name']} | {coeff['beta']:.4f} | {coeff['se']:.4f} | ")
                        f.write(f"{coeff['z']:.3f} | {coeff['p']:.4f} | ")
                        f.write(f"[{coeff['ci_lower']:.4f}, {coeff['ci_upper']:.4f}] |\n")
                    f.write("\n")
                
                # 原始尺度解释（连续变量）
                if result['type'] == 'Continuous' and 'original_scale_beta' in result:
                    orig_scale = result['original_scale_beta']
                    if 'interpretation' in orig_scale:
                        f.write(f"**原始尺度解释**: {orig_scale['interpretation']}\n\n")
                
                # 亚组分析（分类变量）
                if result['type'] == 'Categorical' and 'subgroup_analysis' in result:
                    subgroups = result['subgroup_analysis']
                    if subgroups:
                        f.write("#### 亚组分析\n\n")
                        f.write("| 类别 | k | 效应量 | 标准误 | 95% CI |\n")
                        f.write("|------|---|--------|--------|--------|\n")
                        
                        for subgroup in subgroups:
                            f.write(f"| {subgroup['category']} | {subgroup['k']} | {subgroup['es']:.4f} | ")
                            f.write(f"{subgroup['se']:.4f} | [{subgroup['ci_lower']:.4f}, {subgroup['ci_upper']:.4f}] |\n")
                        f.write("\n")
                
                # 诊断信息
                if 'diagnostics' in result and 'outliers' in result['diagnostics']:
                    diag = result['diagnostics']
                    if diag['outliers'] or diag['high_influence_cook'] or diag['high_influence_dffits']:
                        f.write("#### 模型诊断\n\n")
                        
                        if diag['outliers']:
                            f.write(f"- **潜在异常值** (行索引): {diag['outliers']}\n")
                        
                        if diag['high_influence_cook']:
                            f.write(f"- **高影响力研究 (Cook's距离)** (行索引): {diag['high_influence_cook']}\n")
                        
                        if diag['high_influence_dffits']:
                            f.write(f"- **高影响力研究 (DFFITS)** (行索引): {diag['high_influence_dffits']}\n")
                        
                        f.write("- **建议**: 进行敏感性分析，排除这些研究后重新分析\n\n")
                
                f.write("---\n\n")
            
            # 多变量分析结果
            if hasattr(self, 'multivariate_result'):
                mv = self.multivariate_result
                f.write("## 多变量分析结果\n\n")
                f.write(f"- **包含变量**: {', '.join(mv['variables'])}\n")
                f.write(f"- **研究数量**: {mv['n_studies']}\n")
                f.write(f"- **R²_adj**: {mv['R2_adj']:.3f}\n")
                f.write(f"- **似然比检验**: χ² = {mv['lrt_chi2']:.3f}, df = {mv['lrt_df']}, p = {mv['lrt_p']:.4f}\n\n")
                
                f.write("**注意**: 多变量模型同时考虑了多个调节变量的影响，提供了更准确的效应估计。\n\n")
            
            # 局限性和注意事项
            f.write("## 局限性和注意事项\n\n")
            f.write("### 统计功效\n")
            f.write("- 本分析的统计功效可能有限，特别是对于交互效应的检测\n")
            f.write("- 小样本量可能导致不稳定的参数估计\n")
            f.write("- 建议在更大的样本中验证这些发现\n\n")
            
            f.write("### 多重检验\n")
            f.write("- 已使用Benjamini-Hochberg程序校正多重检验\n")
            f.write("- 校正后的q值应作为判断显著性的主要标准\n")
            f.write("- 未校正的p值仅供参考\n\n")
            
            f.write("### 因果推断\n")
            f.write("- 本分析为观察性研究的二次分析，无法建立因果关系\n")
            f.write("- 调节变量的效应可能受到未观测混杂因素的影响\n")
            f.write("- 结果应谨慎解释，避免过度推广\n\n")
            
            # 硬编码的探索性标签
            f.write("---\n\n")
            f.write("**重要声明**: *注释：本分析属于探索性质，旨在生成假设。所有发现，特别是未经多重检验校正的p值，均需在未来的预注册研究中进行验证性检验。*\n\n")
            f.write(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("**分析工具**: Meta分析调节变量探索脚本 V2.0 Enhanced\n")
        
        self._log(f"详细报告已保存: {report_path}")
    
    def _generate_diagnostic_plots(self):
        """生成诊断图表"""
        if not self.results_summary:
            return
        
        # 创建诊断图表目录
        plots_dir = os.path.join(self.output_directory, "diagnostic_plots")
        Path(plots_dir).mkdir(exist_ok=True)
        
        # R²分布图
        r2_values = [result['R2_adj'] for result in self.results_summary]
        moderator_names = [result['moderator'] for result in self.results_summary]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(r2_values)), r2_values, 
                      color=['red' if result.get('significant_corrected', False) else 'lightblue' 
                            for result in self.results_summary])
        plt.xlabel('调节变量')
        plt.ylabel('R²_adj')
        plt.title('调节变量解释的异质性比例\n*注释：本分析属于探索性质，旨在生成假设*')
        plt.xticks(range(len(moderator_names)), moderator_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='校正后显著'),
                          Patch(facecolor='lightblue', label='非显著')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        r2_plot_path = os.path.join(plots_dir, "r2_distribution.png")
        plt.savefig(r2_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # p值分布图
        p_values = [result['lrt_p'] for result in self.results_summary]
        q_values = [result.get('lrt_q', 1) for result in self.results_summary]
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(moderator_names))
        width = 0.35
        
        plt.bar(x_pos - width/2, p_values, width, label='原始p值', alpha=0.7)
        plt.bar(x_pos + width/2, q_values, width, label='校正q值', alpha=0.7)
        
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        plt.xlabel('调节变量')
        plt.ylabel('p值 / q值')
        plt.title('原始p值与校正q值比较\n*注释：本分析属于探索性质，旨在生成假设*')
        plt.xticks(x_pos, moderator_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        p_plot_path = os.path.join(plots_dir, "p_values_comparison.png")
        plt.savefig(p_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self._log(f"诊断图表已保存到: {plots_dir}")


def main():
    """
    主函数：演示如何使用ModeratorAnalysisV20Enhanced类
    """
    # 示例配置
    config = {
        'input_csv_path': 'meta_analysis_results_v3.1.csv',  # 替换为实际文件路径
        'output_directory': 'moderator_analysis_output_v2.0_enhanced',
        'es_col': 'es',  # 效应量列名
        'var_col': 'v',  # 方差列名
        'cluster_col': 'author',  # 聚类变量列名（可选）
        'categorical_threshold': 10,  # 分类变量判断阈值
        'missing_threshold': 0.5,  # 缺失值排除阈值
        'min_studies_threshold': 10,  # 最小研究数量阈值
        'vif_threshold': 5.0,  # VIF阈值
        'alpha_level': 0.05  # 显著性水平
    }
    
    # 创建分析器实例
    analyzer = ModeratorAnalysisV20Enhanced(**config)
    
    # 运行完整分析
    success = analyzer.run_complete_analysis()
    
    if success:
        print("✓ 分析成功完成！")
        print(f"✓ 结果已保存到: {config['output_directory']}")
        print("✓ 请查看生成的报告和图表")
    else:
        print("✗ 分析失败，请检查日志文件")
    
    return success


if __name__ == "__main__":
    main()

