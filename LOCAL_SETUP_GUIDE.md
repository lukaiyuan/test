# 🏠 本地部署和使用指南

## 📋 概述

本指南将帮助您在本地环境中部署和使用XGBoost多因子量化交易策略系统。

## 🛠️ 环境要求

### 系统要求
- **操作系统**: Windows 10+, macOS 10.14+, 或 Linux
- **Python版本**: 3.8 或更高版本
- **内存**: 建议 8GB 以上
- **存储空间**: 至少 2GB 可用空间

### 软件依赖
- Python 3.8+
- pip (Python包管理器)
- Git (可选，用于版本控制)

## 🚀 快速部署

### 方法一：使用部署脚本（推荐）

1. **下载部署脚本**
   ```bash
   # 创建项目目录
   mkdir xgboost_quant_strategy
   cd xgboost_quant_strategy
   
   # 将deploy_local.py文件复制到此目录
   ```

2. **运行部署脚本**
   ```bash
   python deploy_local.py
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **运行测试**
   ```bash
   python main.py
   python tests/test_basic.py
   ```

### 方法二：手动创建项目

1. **创建项目结构**
   ```bash
   mkdir xgboost_quant_strategy
   cd xgboost_quant_strategy
   
   # 创建目录结构
   mkdir -p config data/models models utils tests docs notebooks logs
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   
   # 激活虚拟环境
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **安装核心依赖**
   ```bash
   pip install pandas numpy xgboost scikit-learn matplotlib seaborn jupyter
   ```

## 📁 完整项目文件

如果您想获得完整的项目文件，需要创建以下核心文件：

### 1. 配置文件

**config/settings.py** - 全局配置
```python
class Config:
    # 数据配置
    DATA_SOURCES = {
        'primary': 'yfinance',
        'backup': 'akshare'
    }
    
    # XGBoost参数
    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    
    # 特征选择配置
    FEATURE_SELECTION = {
        'max_features': 50,
        'min_importance': 0.001
    }
```

### 2. 数据模块

**data/data_loader.py** - 数据获取
```python
import pandas as pd
import numpy as np
import yfinance as yf

class DataLoader:
    def get_stock_data(self, symbol, start_date, end_date):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except:
            # 返回模拟数据
            return self._generate_mock_data(start_date, end_date)
    
    def _generate_mock_data(self, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        # 生成模拟OHLCV数据
        # ... 实现细节
```

### 3. 因子计算

**data/factor_engine.py** - 因子计算引擎
```python
class FactorEngine:
    def calculate_technical_factors(self, data):
        factors = pd.DataFrame(index=data.index)
        
        # 移动平均
        for period in [5, 10, 20, 60]:
            factors[f'sma_{period}'] = data['Close'].rolling(period).mean()
            factors[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        factors['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 更多技术指标...
        return factors
```

## 🎯 使用方法

### 1. 基础使用

```python
# 导入模块
from data.data_loader import DataLoader
from data.factor_engine import FactorEngine

# 获取数据
loader = DataLoader()
data = loader.get_stock_data('AAPL', '2023-01-01', '2024-01-01')

# 计算因子
engine = FactorEngine()
factors = engine.calculate_technical_factors(data)

print(f"计算了 {factors.shape[1]} 个因子")
```

### 2. 模型训练

```python
from models.xgboost_model import XGBoostModel
from models.feature_engineer import FeatureEngineer

# 特征工程
fe = FeatureEngineer()
X_processed = fe.engineer_features(factors, target)

# 模型训练
model = XGBoostModel()
result = model.train(X_train, y_train, X_val, y_val)

print(f"模型训练完成，R²: {result['val_metrics']['r2']:.4f}")
```

### 3. 策略回测

```python
# 生成交易信号
predictions = model.predict(X_test)
signals = ['买入' if p > 0.02 else '卖出' if p < -0.02 else '持有' 
          for p in predictions]

# 计算收益
returns = calculate_strategy_returns(signals, prices)
sharpe_ratio = calculate_sharpe_ratio(returns)

print(f"策略夏普比率: {sharpe_ratio:.4f}")
```

## 📊 数据源配置

### 1. Yahoo Finance (免费)
```python
# 已集成在系统中，无需额外配置
import yfinance as yf
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')
```

### 2. AKShare (免费，中国市场)
```bash
pip install akshare
```

```python
import akshare as ak
# 获取A股数据
stock_data = ak.stock_zh_a_hist(symbol="000001", period="daily")
```

### 3. Tushare (需要注册)
```bash
pip install tushare
```

```python
import tushare as ts
ts.set_token('your_token_here')  # 需要注册获取token
pro = ts.pro_api()
```

## 🔧 常见问题解决

### 1. 依赖安装问题

**问题**: pip安装失败
```bash
# 解决方案：使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy xgboost
```

**问题**: TA-Lib安装失败
```bash
# Windows解决方案
pip install TA-Lib-0.4.24-cp39-cp39-win_amd64.whl

# macOS解决方案
brew install ta-lib
pip install TA-Lib

# Linux解决方案
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

### 2. 数据获取问题

**问题**: 网络连接失败
```python
# 解决方案：使用代理或本地数据
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

**问题**: 数据格式错误
```python
# 解决方案：数据验证和清洗
def validate_data(data):
    if data.empty:
        raise ValueError("数据为空")
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"缺少必要列: {missing_columns}")
    
    return data
```

### 3. 性能优化

**内存优化**:
```python
# 使用更小的数据类型
data = data.astype({
    'Open': 'float32',
    'High': 'float32', 
    'Low': 'float32',
    'Close': 'float32',
    'Volume': 'int32'
})
```

**计算优化**:
```python
# 使用向量化操作
import numba

@numba.jit
def fast_rsi(prices, period=14):
    # 优化的RSI计算
    pass
```

## 📈 扩展功能

### 1. 添加新的数据源
```python
class CustomDataLoader(DataLoader):
    def get_data_from_custom_source(self, symbol):
        # 实现自定义数据源
        pass
```

### 2. 添加新的因子
```python
class CustomFactorEngine(FactorEngine):
    def calculate_custom_factors(self, data):
        # 实现自定义因子
        pass
```

### 3. 集成新的模型
```python
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor()
    
    def train(self, X, y):
        self.model.fit(X, y)
        return self.model
```

## 🎯 下一步

1. **熟悉系统**: 运行基础示例，了解各模块功能
2. **数据准备**: 配置您偏好的数据源
3. **策略开发**: 基于模板开发您的交易策略
4. **回测验证**: 使用历史数据验证策略效果
5. **实盘准备**: 连接券商接口，准备实盘交易

## 📞 技术支持

如果在部署过程中遇到问题：

1. **查看日志**: 检查logs/目录下的日志文件
2. **运行测试**: 执行测试用例定位问题
3. **查看文档**: 参考docs/目录下的详细文档
4. **社区支持**: 在GitHub Issues中提问

---

🎉 **祝您使用愉快！开始您的量化交易之旅吧！** 🎉
