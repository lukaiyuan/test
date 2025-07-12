"""
基础功能测试
验证系统核心模块是否正常工作
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from data.data_loader import DataLoader
from data.factor_engine import FactorEngine
from models.feature_engineer import FeatureEngineer
from models.xgboost_model import XGBoostModel

class TestBasicFunctionality(unittest.TestCase):
    """基础功能测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = Config()
        self.data_loader = DataLoader()
        self.factor_engine = FactorEngine()
        self.feature_engineer = FeatureEngineer()
        
        # 创建测试数据
        self.test_data = self._create_test_data()
    
    def _create_test_data(self):
        """创建测试用的股票数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        n_days = len(dates)
        
        # 生成模拟价格数据
        np.random.seed(42)
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(1000000, 10000000)
            amount = volume * close
            
            data.append({
                'date': date,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume,
                'amount': amount
            })
        
        return pd.DataFrame(data).set_index('date')
    
    def test_config_loading(self):
        """测试配置加载"""
        self.assertIsNotNone(self.config)
        self.assertTrue(hasattr(self.config, 'XGBOOST_PARAMS'))
        self.assertTrue(hasattr(self.config, 'DATA_SOURCES'))
        print("✅ 配置加载测试通过")
    
    def test_data_loader(self):
        """测试数据加载器"""
        # 测试股票列表获取
        stock_list = self.data_loader.get_stock_list()
        self.assertIsInstance(stock_list, pd.DataFrame)
        self.assertFalse(stock_list.empty)
        
        # 测试股票数据获取
        symbol = '000001.SZ'
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        stock_data = self.data_loader.get_stock_data(symbol, start_date, end_date)
        self.assertIsInstance(stock_data, pd.DataFrame)
        self.assertFalse(stock_data.empty)
        
        # 检查必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, stock_data.columns)
        
        print("✅ 数据加载器测试通过")
    
    def test_factor_engine(self):
        """测试因子计算引擎"""
        # 测试技术因子计算
        technical_factors = self.factor_engine.calculate_technical_factors(self.test_data)
        
        self.assertIsInstance(technical_factors, pd.DataFrame)
        self.assertFalse(technical_factors.empty)
        self.assertEqual(len(technical_factors), len(self.test_data))
        
        # 检查是否包含一些基本的技术因子
        expected_factors = ['sma_5', 'sma_20', 'ema_12', 'rsi_14', 'macd']
        for factor in expected_factors:
            self.assertTrue(any(factor in col for col in technical_factors.columns),
                          f"未找到因子: {factor}")
        
        print(f"✅ 因子计算引擎测试通过，计算了 {technical_factors.shape[1]} 个因子")
    
    def test_feature_engineer(self):
        """测试特征工程器"""
        # 创建测试特征和目标
        factors = self.factor_engine.calculate_technical_factors(self.test_data)
        
        # 移除包含NaN的行
        factors = factors.dropna()
        
        if factors.empty:
            self.skipTest("因子数据为空，跳过特征工程测试")
        
        # 创建目标变量
        target = self.test_data['close'].pct_change(5).dropna()
        
        # 对齐数据
        common_index = factors.index.intersection(target.index)
        if len(common_index) < 10:
            self.skipTest("有效数据点不足，跳过特征工程测试")
        
        factors_aligned = factors.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # 测试特征选择
        selected_features = self.feature_engineer.select_features(
            factors_aligned, target_aligned, method='importance'
        )
        
        self.assertIsInstance(selected_features, list)
        self.assertGreater(len(selected_features), 0)
        
        # 测试特征缩放
        scaled_features = self.feature_engineer.scale_features(factors_aligned)
        self.assertEqual(scaled_features.shape, factors_aligned.shape)
        
        print(f"✅ 特征工程器测试通过，选择了 {len(selected_features)} 个特征")
    
    def test_xgboost_model(self):
        """测试XGBoost模型"""
        # 准备测试数据
        factors = self.factor_engine.calculate_technical_factors(self.test_data)
        factors = factors.dropna()
        
        if factors.empty:
            self.skipTest("因子数据为空，跳过模型测试")
        
        # 创建目标变量
        target = self.test_data['close'].pct_change(5).dropna()
        
        # 对齐数据
        common_index = factors.index.intersection(target.index)
        if len(common_index) < 50:
            self.skipTest("有效数据点不足，跳过模型测试")
        
        X = factors.loc[common_index]
        y = target.loc[common_index]
        
        # 选择数值列
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # 分割数据
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 创建和训练模型
        model = XGBoostModel()
        
        # 使用较小的参数进行快速测试
        model.params.update({
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        })
        
        training_result = model.train(X_train, y_train, X_test, y_test)
        
        self.assertIsNotNone(model.model)
        self.assertIn('train_metrics', training_result)
        self.assertIn('val_metrics', training_result)
        
        # 测试预测
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        
        print("✅ XGBoost模型测试通过")
    
    def test_integration(self):
        """集成测试"""
        print("🔄 开始集成测试...")
        
        try:
            # 1. 获取数据
            symbol = '000001.SZ'
            start_date = '2023-01-01'
            end_date = '2023-12-31'
            
            stock_data = self.data_loader.get_stock_data(symbol, start_date, end_date)
            self.assertFalse(stock_data.empty)
            
            # 2. 计算因子
            factors = self.factor_engine.calculate_technical_factors(stock_data)
            factors = factors.dropna()
            
            if factors.empty:
                print("⚠️ 因子数据为空，使用测试数据")
                factors = self.factor_engine.calculate_technical_factors(self.test_data)
                factors = factors.dropna()
            
            # 3. 创建目标变量
            target = stock_data['close'].pct_change(5).dropna()
            if target.empty:
                target = self.test_data['close'].pct_change(5).dropna()
            
            # 4. 对齐数据
            common_index = factors.index.intersection(target.index)
            if len(common_index) < 30:
                self.skipTest("有效数据点不足，跳过集成测试")
            
            X = factors.loc[common_index]
            y = target.loc[common_index]
            
            # 选择数值列
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_columns]
            
            # 5. 特征工程
            if len(X) > 20:  # 确保有足够的数据
                X_engineered = self.feature_engineer.engineer_features(
                    X, y,
                    feature_selection=True,
                    interaction_features=False,  # 简化测试
                    scaling=True
                )
            else:
                X_engineered = X
            
            # 6. 训练模型
            if len(X_engineered) > 10:
                split_idx = max(int(len(X_engineered) * 0.8), len(X_engineered) - 5)
                X_train = X_engineered.iloc[:split_idx]
                y_train = y.iloc[:split_idx]
                X_test = X_engineered.iloc[split_idx:]
                y_test = y.iloc[split_idx:]
                
                model = XGBoostModel()
                model.params.update({
                    'n_estimators': 5,
                    'max_depth': 2,
                    'learning_rate': 0.3
                })
                
                training_result = model.train(X_train, y_train)
                predictions = model.predict(X_test)
                
                self.assertEqual(len(predictions), len(X_test))
                
            print("✅ 集成测试通过")
            
        except Exception as e:
            self.fail(f"集成测试失败: {e}")

def run_tests():
    """运行所有测试"""
    print("🧪 开始运行基础功能测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicFunctionality)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("=" * 50)
    if result.wasSuccessful():
        print("🎉 所有测试通过!")
    else:
        print(f"❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_tests()
