"""
XGBoost多因子量化交易策略主程序
演示完整的策略开发流程
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from config.settings import Config
from utils.logger import LoggerManager, log_info
from utils.helpers import ensure_dir, timer
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from data.factor_engine import FactorEngine
from models.feature_engineer import FeatureEngineer
from models.xgboost_model import XGBoostModel

# 初始化日志
logger = LoggerManager.get_logger("MainProgram")

@timer
def main():
    """主程序"""
    log_info("=" * 60)
    log_info("XGBoost多因子量化交易策略系统启动")
    log_info("=" * 60)
    
    try:
        # 1. 初始化配置和模块
        config = Config()
        data_loader = DataLoader()
        data_processor = DataProcessor()
        factor_engine = FactorEngine()
        feature_engineer = FeatureEngineer()
        
        log_info("✅ 模块初始化完成")
        
        # 2. 获取股票列表
        log_info("📊 获取股票列表...")
        stock_list = data_loader.get_stock_list()
        
        if stock_list.empty:
            log_info("❌ 无法获取股票列表，使用模拟数据")
            symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH']
        else:
            # 选择前10只股票进行演示
            symbols = stock_list['ts_code'].head(10).tolist()
        
        log_info(f"📈 选择股票: {symbols}")
        
        # 3. 设置时间范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        log_info(f"📅 数据时间范围: {start_date} 到 {end_date}")
        
        # 4. 准备训练数据
        log_info("🔄 准备训练数据...")
        features, targets = data_processor.prepare_training_data(symbols, start_date, end_date)
        
        if features.empty:
            log_info("❌ 无法获取训练数据，程序退出")
            return
        
        log_info(f"✅ 数据准备完成: {features.shape[0]} 条记录, {features.shape[1]} 个特征")
        
        # 5. 特征工程
        log_info("🔧 开始特征工程...")
        
        # 选择目标变量
        target_col = 'return_20d'
        if target_col not in targets.columns:
            target_col = targets.columns[0]
        
        y = targets[target_col].dropna()
        X = features.loc[y.index]

        # 移除非数值列（如symbol列）
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]

        # 应用特征工程
        X_engineered = feature_engineer.engineer_features(
            X_numeric, y,
            feature_selection=True,
            interaction_features=True,
            polynomial_features=False,
            scaling=True,
            dimensionality_reduction=False
        )
        
        log_info(f"✅ 特征工程完成: {X_engineered.shape[1]} 个最终特征")
        
        # 6. 数据分割
        log_info("📊 分割数据集...")
        data_splits = data_processor.split_data(X_engineered, y)
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        log_info(f"✅ 数据分割完成:")
        log_info(f"   训练集: {len(X_train)} 条记录")
        log_info(f"   验证集: {len(X_val)} 条记录")
        log_info(f"   测试集: {len(X_test)} 条记录")
        
        # 7. 模型训练
        log_info("🤖 开始模型训练...")
        
        # 创建XGBoost模型
        xgb_model = XGBoostModel()
        
        # 训练模型
        training_result = xgb_model.train(
            X_train, y_train,
            X_val, y_val,
            early_stopping_rounds=50
        )
        
        log_info("✅ 模型训练完成")
        
        # 8. 模型评估
        log_info("📈 模型评估...")
        
        # 测试集预测
        test_predictions = xgb_model.predict(X_test)
        test_metrics = xgb_model._calculate_metrics(y_test, test_predictions)
        
        log_info("📊 测试集性能:")
        for metric, value in test_metrics.items():
            log_info(f"   {metric}: {value:.4f}")
        
        # 9. 特征重要性分析
        log_info("🔍 特征重要性分析...")
        
        importance_df = xgb_model.get_feature_importance()
        log_info("📊 Top 10 重要特征:")
        for idx, row in importance_df.head(10).iterrows():
            log_info(f"   {row['feature']}: {row['importance']:.4f}")
        
        # 10. 交叉验证
        log_info("🔄 执行交叉验证...")
        cv_result = xgb_model.cross_validate(X_engineered, y, cv_folds=3)
        
        log_info("📊 交叉验证结果:")
        for metric, value in cv_result['avg_metrics'].items():
            log_info(f"   {metric}: {value:.4f}")
        
        # 11. 保存模型和结果
        log_info("💾 保存模型和结果...")
        
        # 确保输出目录存在
        model_dir = ensure_dir(config.get_data_path('model_data'))
        
        # 保存模型
        model_path = model_dir / 'xgboost_model.pkl'
        xgb_model.save_model(str(model_path))
        
        # 保存特征工程器
        feature_engineer_path = model_dir / 'feature_engineer.pkl'
        feature_engineer.save_feature_engineer(str(feature_engineer_path))
        
        # 保存结果
        results = {
            'training_result': training_result,
            'test_metrics': test_metrics,
            'cv_result': cv_result,
            'feature_importance': importance_df.to_dict('records'),
            'model_params': xgb_model.params
        }
        
        from utils.helpers import save_json
        results_path = model_dir / 'training_results.json'
        save_json(results, str(results_path))
        
        log_info("✅ 模型和结果保存完成")
        
        # 12. 生成报告
        log_info("📄 生成策略报告...")
        generate_strategy_report(results, symbols, start_date, end_date)
        
        log_info("=" * 60)
        log_info("🎉 XGBoost多因子量化交易策略开发完成!")
        log_info("=" * 60)
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

def generate_strategy_report(results: dict, symbols: list, 
                           start_date: str, end_date: str):
    """生成策略报告"""
    
    report_content = f"""
# XGBoost多因子量化交易策略报告

## 策略概述
- **策略名称**: XGBoost多因子量化交易策略
- **开发时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据时间范围**: {start_date} 到 {end_date}
- **股票池**: {len(symbols)} 只股票

## 模型性能

### 测试集指标
"""
    
    for metric, value in results['test_metrics'].items():
        report_content += f"- **{metric}**: {value:.4f}\n"
    
    report_content += "\n### 交叉验证指标\n"
    for metric, value in results['cv_result']['avg_metrics'].items():
        if 'mean_' in metric:
            clean_metric = metric.replace('mean_', '')
            report_content += f"- **{clean_metric}**: {value:.4f}\n"
    
    report_content += "\n### Top 10 重要特征\n"
    for i, feature_info in enumerate(results['feature_importance'][:10]):
        report_content += f"{i+1}. **{feature_info['feature']}**: {feature_info['importance']:.4f}\n"
    
    report_content += f"""

## 模型参数
```json
{results['model_params']}
```

## 策略特点
1. **多因子模型**: 结合技术指标、基本面指标等多类因子
2. **机器学习**: 使用XGBoost算法进行预测
3. **特征工程**: 包含特征选择、交互特征、标准化等
4. **时间序列验证**: 使用时间序列交叉验证避免未来信息泄露
5. **风险控制**: 内置多层次风险控制机制

## 下一步计划
1. 策略信号生成
2. 组合优化
3. 回测验证
4. 实盘部署

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    config = Config()
    docs_dir = ensure_dir('./docs')
    report_path = docs_dir / 'strategy_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    log_info(f"📄 策略报告已保存到: {report_path}")

if __name__ == "__main__":
    main()
