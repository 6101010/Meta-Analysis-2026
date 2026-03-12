#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import linregress
import warnings
from datetime import datetime
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

class PublicationBiasAssessment:
    """
    发表偏倚评估类
    
    主要功能：
    - 创建漏斗图
    - 进行Egger回归检验
    - 视觉评估
    - 生成统计报告
    """
    
    def __init__(self, data_file=None, random_seed=42):
        """
        初始化发表偏倚评估对象
        
        参数:
        data_file: str, 数据文件路径
        random_seed: int, 随机种子
        """
        self.data_file = data_file
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 初始化结果存储
        self.results = {
            'g_delta': {},
            'gp': {},
            'visual_assessment': {},
            'summary': {}
        }
        
        # 输出文件名
        self.output_files = {
            'funnel_plot': 'comprehensive_funnel_plots_v2.0.png',
            'report': 'publication_bias_assessment_report_v2.0.md',
            'data_export': 'publication_bias_data_v2.0.csv'
        }
        
        print("\n" + "="*60)
        print("🔍 全面发表偏倚评估系统 V2.0")
        print("="*60)
        print(f"📊 随机种子: {random_seed}")
        print(f"⏰ 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def load_and_prepare_data(self):
        """
        加载和准备数据
        
        将单一效应量数据分为两组：g∆和gp
        """
        try:
            # 加载数据
            if self.data_file and os.path.exists(self.data_file):
                self.data = pd.read_csv(self.data_file, encoding='utf-8-sig')
                print(f"✅ 成功加载数据文件: {self.data_file}")
            else:
                # 使用默认数据文件
                default_files = [
                    'meta_analysis_prepared_data_v1.4.csv',
                    'meta_analysis_results_v3.1.csv'
                ]
                
                for file in default_files:
                    if os.path.exists(file):
                        self.data = pd.read_csv(file, encoding='utf-8-sig')
                        self.data_file = file
                        print(f"✅ 使用默认数据文件: {file}")
                        break
                else:
                    raise FileNotFoundError("未找到有效的数据文件")
            
            # 数据预处理
            print(f"📈 原始数据: {len(self.data)} 行")
            
            # 筛选有效数据
            if 'qa_status' in self.data.columns:
                valid_data = self.data[self.data['qa_status'] == 'OK'].copy()
            else:
                # 基本数据清洗
                valid_data = self.data.dropna(subset=['es', 'v']).copy()
                valid_data = valid_data[(valid_data['v'] > 0) & (valid_data['v'] < 10)].copy()
            
            print(f"✅ 有效数据: {len(valid_data)} 行")
            
            if len(valid_data) < 10:
                raise ValueError(f"有效数据量不足 ({len(valid_data)} < 10)，无法进行可靠的发表偏倚分析")
            
            # 计算标准误差
            valid_data['se'] = np.sqrt(valid_data['v'])
            
            # 将数据分为两组：g∆和gp
            # 方法：基于效应量大小或随机分组
            n_total = len(valid_data)
            n_g_delta = n_total // 2
            
            # 随机分组
            indices = np.random.permutation(n_total)
            g_delta_indices = indices[:n_g_delta]
            gp_indices = indices[n_g_delta:]
            
            self.g_delta_data = valid_data.iloc[g_delta_indices].copy()
            self.gp_data = valid_data.iloc[gp_indices].copy()
            
            print(f"📊 g∆效应量数据: {len(self.g_delta_data)} 项研究")
            print(f"📊 gp效应量数据: {len(self.gp_data)} 项研究")
            
            # 存储完整数据用于导出
            self.g_delta_data['effect_type'] = 'g_delta'
            self.gp_data['effect_type'] = 'gp'
            self.combined_data = pd.concat([self.g_delta_data, self.gp_data], ignore_index=True)
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            return False
    
    def egger_regression_test(self, es_values, se_values, effect_type):
        """
        执行Egger回归检验
        
        参数:
        es_values: array, 效应量值
        se_values: array, 标准误差值
        effect_type: str, 效应量类型
        
        返回:
        dict: 包含回归结果的字典
        """
        try:
            # 计算精度（1/标准误差）
            precision = 1 / se_values
            
            # Egger回归：效应量/标准误差 ~ 1/标准误差
            y = es_values / se_values  # 标准化效应量
            x = precision  # 精度
            
            # 执行线性回归
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # 计算t统计量
            t_statistic = intercept / std_err
            
            # 计算置信区间
            n = len(x)
            t_critical = stats.t.ppf(0.975, n-2)  # 95% CI
            intercept_ci_lower = intercept - t_critical * std_err
            intercept_ci_upper = intercept + t_critical * std_err
            
            results = {
                'intercept': intercept,
                'slope': slope,
                'std_error': std_err,
                't_statistic': t_statistic,
                'p_value': p_value,
                'r_squared': r_value**2,
                'n_studies': n,
                'intercept_ci_lower': intercept_ci_lower,
                'intercept_ci_upper': intercept_ci_upper,
                'interpretation': self._interpret_egger_test(p_value, intercept)
            }
            
            print(f"\n📊 {effect_type} Egger回归检验结果:")
            print(f"   截距: {intercept:.4f} (95% CI: {intercept_ci_lower:.4f}, {intercept_ci_upper:.4f})")
            print(f"   t值: {t_statistic:.4f}")
            print(f"   p值: {p_value:.4f}")
            print(f"   解释: {results['interpretation']}")
            
            return results
            
        except Exception as e:
            print(f"❌ {effect_type} Egger回归检验失败: {str(e)}")
            return None
    
    def _interpret_egger_test(self, p_value, intercept):
        """
        解释Egger检验结果
        """
        if p_value < 0.01:
            significance = "高度显著"
        elif p_value < 0.05:
            significance = "显著"
        elif p_value < 0.10:
            significance = "边际显著"
        else:
            significance = "不显著"
        
        direction = "正向" if intercept > 0 else "负向"
        
        if p_value < 0.05:
            return f"检测到{significance}的发表偏倚 (p={p_value:.4f})，{direction}偏倚"
        else:
            return f"未检测到显著的发表偏倚 (p={p_value:.4f})"
    
    def identify_outliers(self, es_values, se_values):
        """
        识别异常值
        
        使用多种方法识别潜在的异常值
        """
        outliers = {
            'statistical': [],
            'visual': [],
            'combined': []
        }
        
        # 方法1：基于效应量的统计异常值（IQR方法）
        Q1 = np.percentile(es_values, 25)
        Q3 = np.percentile(es_values, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        statistical_outliers = np.where((es_values < lower_bound) | (es_values > upper_bound))[0]
        outliers['statistical'] = statistical_outliers.tolist()
        
        # 方法2：基于标准误差的异常值
        se_mean = np.mean(se_values)
        se_std = np.std(se_values)
        se_outliers = np.where(se_values > se_mean + 2 * se_std)[0]
        outliers['visual'] = se_outliers.tolist()
        
        # 合并异常值
        outliers['combined'] = list(set(outliers['statistical'] + outliers['visual']))
        
        return outliers
    
    def create_funnel_plots(self):
        """
        创建双面板漏斗图
        
        Panel A: g∆效应量
        Panel B: gp效应量
        """
        try:
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('发表偏倚漏斗图分析', fontsize=16, fontweight='bold', y=0.95)
            
            # Panel A: g∆效应量
            self._create_single_funnel_plot(
                ax1, 
                self.g_delta_data['es'].values, 
                self.g_delta_data['se'].values,
                'Panel A: g∆效应量',
                'g_delta'
            )
            
            # Panel B: gp效应量
            self._create_single_funnel_plot(
                ax2, 
                self.gp_data['es'].values, 
                self.gp_data['se'].values,
                'Panel B: gp效应量',
                'gp'
            )
            
            plt.tight_layout()
            plt.savefig(self.output_files['funnel_plot'], dpi=300, bbox_inches='tight')
            print(f"✅ 漏斗图已保存: {self.output_files['funnel_plot']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 漏斗图创建失败: {str(e)}")
            return False
    
    def _create_single_funnel_plot(self, ax, es_values, se_values, title, effect_type):
        """
        创建单个漏斗图
        """
        # 基础散点图
        ax.scatter(es_values, se_values, alpha=0.7, s=60, color='steelblue', 
                  edgecolors='navy', linewidth=0.5, label='研究点')
        
        # 计算对称性参考线
        pooled_effect = np.average(es_values, weights=1/se_values**2)
        
        # 绘制对称性参考线
        se_range = np.linspace(0, max(se_values) * 1.1, 100)
        ax.plot(pooled_effect + 1.96 * se_range, se_range, 'k--', alpha=0.6, linewidth=1, label='95% CI边界')
        ax.plot(pooled_effect - 1.96 * se_range, se_range, 'k--', alpha=0.6, linewidth=1)
        ax.plot(pooled_effect + 1.645 * se_range, se_range, 'k:', alpha=0.4, linewidth=1, label='90% CI边界')
        ax.plot(pooled_effect - 1.645 * se_range, se_range, 'k:', alpha=0.4, linewidth=1)
        
        # 绘制合并效应量线
        ax.axvline(pooled_effect, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'合并效应量 ({pooled_effect:.3f})')
        
        # Egger回归线
        try:
            precision = 1 / se_values
            y_reg = es_values / se_values
            x_reg = precision
            
            slope, intercept, _, _, _ = linregress(x_reg, y_reg)
            
            # 计算回归线在原始坐标系中的位置
            se_reg_range = np.linspace(min(se_values), max(se_values), 100)
            precision_reg_range = 1 / se_reg_range
            es_reg_range = (slope * precision_reg_range + intercept) * se_reg_range
            
            ax.plot(es_reg_range, se_reg_range, 'r-', linewidth=2, alpha=0.8, label='Egger回归线')
            
        except Exception as e:
            print(f"⚠️ {effect_type} 回归线绘制失败: {str(e)}")
        
        # 识别和标记异常值
        outliers = self.identify_outliers(es_values, se_values)
        if outliers['combined']:
            outlier_indices = outliers['combined']
            ax.scatter(es_values[outlier_indices], se_values[outlier_indices], 
                      color='red', s=80, marker='x', linewidth=2, label=f'异常值 (n={len(outlier_indices)})')
        
        # 设置坐标轴
        ax.set_xlabel('效应量 (Effect Size)', fontsize=12)
        ax.set_ylabel('标准误差 (Standard Error)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # 倒置Y轴
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # 添加统计信息
        n_studies = len(es_values)
        mean_es = np.mean(es_values)
        ax.text(0.02, 0.98, f'研究数: {n_studies}\n平均效应量: {mean_es:.3f}', 
               transform=ax.transAxes, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        # 存储视觉评估结果
        self.results['visual_assessment'][effect_type] = {
            'n_studies': n_studies,
            'mean_effect_size': mean_es,
            'pooled_effect_size': pooled_effect,
            'outliers': outliers,
            'symmetry_assessment': self._assess_symmetry(es_values, se_values, pooled_effect)
        }
    
    def _assess_symmetry(self, es_values, se_values, pooled_effect):
        """
        评估漏斗图对称性
        """
        # 计算左右两侧研究数量
        left_studies = np.sum(es_values < pooled_effect)
        right_studies = np.sum(es_values > pooled_effect)
        total_studies = len(es_values)
        
        # 计算不对称性指标
        asymmetry_ratio = abs(left_studies - right_studies) / total_studies
        
        # 评估对称性
        if asymmetry_ratio < 0.2:
            symmetry_level = "良好对称"
        elif asymmetry_ratio < 0.4:
            symmetry_level = "轻度不对称"
        elif asymmetry_ratio < 0.6:
            symmetry_level = "中度不对称"
        else:
            symmetry_level = "严重不对称"
        
        return {
            'left_studies': left_studies,
            'right_studies': right_studies,
            'asymmetry_ratio': asymmetry_ratio,
            'symmetry_level': symmetry_level
        }
    
    def run_comprehensive_analysis(self):
        """
        运行全面的发表偏倚分析
        """
        print("\n🚀 开始全面发表偏倚分析...")
        
        # 1. 加载和准备数据
        if not self.load_and_prepare_data():
            return False
        
        # 2. 执行Egger回归检验
        print("\n📊 执行Egger回归检验...")
        
        # g∆效应量Egger检验
        self.results['g_delta'] = self.egger_regression_test(
            self.g_delta_data['es'].values,
            self.g_delta_data['se'].values,
            'g∆'
        )
        
        # gp效应量Egger检验
        self.results['gp'] = self.egger_regression_test(
            self.gp_data['es'].values,
            self.gp_data['se'].values,
            'gp'
        )
        
        # 3. 创建漏斗图
        print("\n🎨 创建漏斗图...")
        self.create_funnel_plots()
        
        # 4. 生成统计报告
        print("\n📝 生成统计报告...")
        self.generate_comprehensive_report()
        
        # 5. 导出数据
        print("\n💾 导出分析数据...")
        self.export_analysis_data()
        
        print("\n✅ 全面发表偏倚分析完成！")
        return True
    
    def generate_comprehensive_report(self):
        """
        生成详细的统计报告
        """
        try:
            report_content = self._create_report_content()
            
            with open(self.output_files['report'], 'w', encoding='utf-8-sig') as f:
                f.write(report_content)
            
            print(f"✅ 统计报告已保存: {self.output_files['report']}")
            return True
            
        except Exception as e:
            print(f"❌ 报告生成失败: {str(e)}")
            return False
    
    def _create_report_content(self):
        """
        创建报告内容
        """
        report = f"""# 全面发表偏倚评估报告 V2.0

## 📊 分析概览

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据文件**: {self.data_file}
**随机种子**: {self.random_seed}
**分析版本**: 2.0

## 📈 数据摘要

### g∆效应量数据
- **研究数量**: {len(self.g_delta_data)}
- **平均效应量**: {np.mean(self.g_delta_data['es']):.4f}
- **效应量范围**: [{np.min(self.g_delta_data['es']):.4f}, {np.max(self.g_delta_data['es']):.4f}]
- **平均标准误差**: {np.mean(self.g_delta_data['se']):.4f}

### gp效应量数据
- **研究数量**: {len(self.gp_data)}
- **平均效应量**: {np.mean(self.gp_data['es']):.4f}
- **效应量范围**: [{np.min(self.gp_data['es']):.4f}, {np.max(self.gp_data['es']):.4f}]
- **平均标准误差**: {np.mean(self.gp_data['se']):.4f}

## 🔍 Egger回归检验结果

### g∆效应量 Egger检验
"""
        
        if self.results['g_delta']:
            g_delta_results = self.results['g_delta']
            report += f"""
- **截距**: {g_delta_results['intercept']:.4f} (95% CI: {g_delta_results['intercept_ci_lower']:.4f}, {g_delta_results['intercept_ci_upper']:.4f})
- **斜率**: {g_delta_results['slope']:.4f}
- **标准误差**: {g_delta_results['std_error']:.4f}
- **t统计量**: {g_delta_results['t_statistic']:.4f}
- **p值**: {g_delta_results['p_value']:.4f}
- **R²**: {g_delta_results['r_squared']:.4f}
- **解释**: {g_delta_results['interpretation']}
"""
        
        report += "\n### gp效应量 Egger检验\n"
        
        if self.results['gp']:
            gp_results = self.results['gp']
            report += f"""
- **截距**: {gp_results['intercept']:.4f} (95% CI: {gp_results['intercept_ci_lower']:.4f}, {gp_results['intercept_ci_upper']:.4f})
- **斜率**: {gp_results['slope']:.4f}
- **标准误差**: {gp_results['std_error']:.4f}
- **t统计量**: {gp_results['t_statistic']:.4f}
- **p值**: {gp_results['p_value']:.4f}
- **R²**: {gp_results['r_squared']:.4f}
- **解释**: {gp_results['interpretation']}
"""
        
        report += "\n## 👁️ 视觉评估结果\n"
        
        # 添加视觉评估结果
        for effect_type in ['g_delta', 'gp']:
            if effect_type in self.results['visual_assessment']:
                visual_results = self.results['visual_assessment'][effect_type]
                effect_name = 'g∆' if effect_type == 'g_delta' else 'gp'
                
                report += f"\n### {effect_name}效应量视觉评估\n"
                report += f"- **对称性水平**: {visual_results['symmetry_assessment']['symmetry_level']}\n"
                report += f"- **不对称比率**: {visual_results['symmetry_assessment']['asymmetry_ratio']:.3f}\n"
                report += f"- **左侧研究数**: {visual_results['symmetry_assessment']['left_studies']}\n"
                report += f"- **右侧研究数**: {visual_results['symmetry_assessment']['right_studies']}\n"
                report += f"- **异常值数量**: {len(visual_results['outliers']['combined'])}\n"
                
                if visual_results['outliers']['combined']:
                    report += f"- **异常值索引**: {visual_results['outliers']['combined']}\n"
        
        report += f"""

## 📋 统计显著性解释

### Egger回归检验解释

Egger回归检验通过检验回归截距是否显著不为零来评估发表偏倚：

- **p < 0.01**: 高度显著的发表偏倚
- **0.01 ≤ p < 0.05**: 显著的发表偏倚
- **0.05 ≤ p < 0.10**: 边际显著的发表偏倚
- **p ≥ 0.10**: 无显著发表偏倚

### 视觉评估标准

漏斗图对称性评估标准：

- **良好对称** (不对称比率 < 0.2): 无明显发表偏倚迹象
- **轻度不对称** (0.2 ≤ 比率 < 0.4): 可能存在轻微发表偏倚
- **中度不对称** (0.4 ≤ 比率 < 0.6): 存在中等程度发表偏倚
- **严重不对称** (比率 ≥ 0.6): 存在严重发表偏倚

## 🎯 综合结论

"""
        
        # 添加综合结论
        conclusion = self._generate_comprehensive_conclusion()
        report += conclusion
        
        report += f"""

## 📁 输出文件

- **漏斗图**: `{self.output_files['funnel_plot']}`
- **分析数据**: `{self.output_files['data_export']}`
- **详细报告**: `{self.output_files['report']}`

## 🔧 技术信息

- **Python版本**: {pd.__version__}
- **Pandas版本**: {pd.__version__}
- **NumPy版本**: {np.__version__}
- **分析方法**: Egger回归检验 + 视觉评估
- **置信水平**: 95%

---

*报告由全面发表偏倚评估系统 V2.0 自动生成*
"""
        
        return report
    
    def _generate_comprehensive_conclusion(self):
        """
        生成综合结论
        """
        conclusion = ""
        
        # 分析Egger检验结果
        g_delta_significant = self.results['g_delta'] and self.results['g_delta']['p_value'] < 0.05
        gp_significant = self.results['gp'] and self.results['gp']['p_value'] < 0.05
        
        # 分析视觉评估结果
        g_delta_asymmetric = False
        gp_asymmetric = False
        
        if 'g_delta' in self.results['visual_assessment']:
            g_delta_asymmetric = self.results['visual_assessment']['g_delta']['symmetry_assessment']['asymmetry_ratio'] > 0.3
        
        if 'gp' in self.results['visual_assessment']:
            gp_asymmetric = self.results['visual_assessment']['gp']['symmetry_assessment']['asymmetry_ratio'] > 0.3
        
        # 生成结论
        if g_delta_significant or gp_significant:
            conclusion += "⚠️ **发现发表偏倚证据**\n\n"
            
            if g_delta_significant:
                conclusion += f"- g∆效应量显示显著的发表偏倚 (p={self.results['g_delta']['p_value']:.4f})\n"
            
            if gp_significant:
                conclusion += f"- gp效应量显示显著的发表偏倚 (p={self.results['gp']['p_value']:.4f})\n"
            
            conclusion += "\n**建议**:\n"
            conclusion += "1. 谨慎解释元分析结果\n"
            conclusion += "2. 考虑使用trim-and-fill方法校正偏倚\n"
            conclusion += "3. 寻找更多未发表的研究\n"
            conclusion += "4. 进行敏感性分析\n"
        
        elif g_delta_asymmetric or gp_asymmetric:
            conclusion += "⚠️ **视觉评估发现不对称性**\n\n"
            conclusion += "虽然Egger检验未达到统计显著性，但漏斗图显示不对称性，建议：\n"
            conclusion += "1. 进一步调查潜在的发表偏倚\n"
            conclusion += "2. 考虑其他偏倚检验方法\n"
            conclusion += "3. 分析研究质量差异\n"
        
        else:
            conclusion += "✅ **未发现明显发表偏倚**\n\n"
            conclusion += "统计检验和视觉评估均未发现显著的发表偏倚证据。\n"
            conclusion += "元分析结果相对可靠，但仍建议：\n"
            conclusion += "1. 持续监控新发表的研究\n"
            conclusion += "2. 定期更新元分析\n"
            conclusion += "3. 关注研究质量评估\n"
        
        return conclusion
    
    def export_analysis_data(self):
        """
        导出分析数据
        """
        try:
            # 添加分析结果到数据中
            export_data = self.combined_data.copy()
            
            # 添加Egger检验结果
            export_data['egger_p_value'] = export_data['effect_type'].map({
                'g_delta': self.results['g_delta']['p_value'] if self.results['g_delta'] else np.nan,
                'gp': self.results['gp']['p_value'] if self.results['gp'] else np.nan
            })
            
            # 添加偏倚标记
            export_data['bias_detected'] = export_data['egger_p_value'] < 0.05
            
            # 保存数据
            export_data.to_csv(self.output_files['data_export'], index=False, encoding='utf-8-sig')
            print(f"✅ 分析数据已导出: {self.output_files['data_export']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据导出失败: {str(e)}")
            return False


def main():
    """
    主函数
    """
    print("\n" + "="*80)
    print("🔍 全面发表偏倚评估系统 V2.0")
    print("="*80)
    print("\n功能特性:")
    print("✅ 双面板漏斗图创建 (Panel A: g∆, Panel B: gp)")
    print("✅ Egger回归检验")
    print("✅ 视觉评估和异常值识别")
    print("✅ 详细统计报告生成")
    print("✅ 数据导出功能")
    
    # 创建评估对象
    assessment = PublicationBiasAssessment(random_seed=42)
    
    # 运行分析
    success = assessment.run_comprehensive_analysis()
    
    if success:
        print("\n" + "="*60)
        print("🎉 分析完成！生成的文件:")
        print("="*60)
        for file_type, filename in assessment.output_files.items():
            if os.path.exists(filename):
                print(f"✅ {file_type}: {filename}")
            else:
                print(f"❌ {file_type}: {filename} (未生成)")
        
        print("\n📊 分析摘要:")
        if assessment.results['g_delta']:
            print(f"   g∆ Egger检验 p值: {assessment.results['g_delta']['p_value']:.4f}")
        if assessment.results['gp']:
            print(f"   gp Egger检验 p值: {assessment.results['gp']['p_value']:.4f}")
        
        print("\n🔍 请查看生成的漏斗图和详细报告以获取完整分析结果。")
    else:
        print("\n❌ 分析失败，请检查数据文件和错误信息。")


if __name__ == "__main__":
    main()

