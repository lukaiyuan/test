"""
全局配置文件
XGBoost多因子量化交易策略系统配置
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class Config:
    """全局配置类"""
    
    # ==================== 基础配置 ====================
    PROJECT_NAME = "XGBoost多因子量化交易策略"
    VERSION = "1.0.0"
    DEBUG = True
    
    # ==================== 数据配置 ====================
    # 数据源配置
    DATA_SOURCES = {
        'primary': 'tushare',      # 主要数据源: tushare, yfinance, akshare
        'backup': 'yfinance',      # 备用数据源
        'realtime': 'tushare'      # 实时数据源
    }
    
    # 数据库配置
    DATABASE = {
        'type': 'sqlite',          # sqlite, mysql, postgresql
        'host': 'localhost',
        'port': 3306,
        'username': '',
        'password': '',
        'database': 'quant_data.db'
    }
    
    # 数据存储路径
    DATA_PATH = {
        'raw_data': './data/raw/',
        'processed_data': './data/processed/',
        'factor_data': './data/factors/',
        'model_data': './data/models/',
        'backtest_results': './data/backtest/'
    }
    
    # ==================== 交易配置 ====================
    # 交易市场
    MARKET = {
        'exchange': 'SSE',         # SSE(上交所), SZSE(深交所), NASDAQ, NYSE
        'trading_calendar': 'china_sse',
        'currency': 'CNY'
    }
    
    # 股票池配置
    STOCK_POOL = {
        'universe': 'hs300',       # hs300, zz500, zz1000, all_a
        'min_market_cap': 50,      # 最小市值(亿元)
        'min_liquidity': 1000000,  # 最小日均成交额
        'exclude_st': True,        # 排除ST股票
        'exclude_new': 30          # 排除上市不足N天的新股
    }
    
    # ==================== 因子配置 ====================
    # 因子类别
    FACTOR_CATEGORIES = {
        'technical': True,         # 技术因子
        'fundamental': True,       # 基本面因子
        'momentum': True,          # 动量因子
        'volatility': True,        # 波动率因子
        'volume': True,            # 成交量因子
        'sentiment': False         # 情绪因子(暂不启用)
    }
    
    # 因子计算参数
    FACTOR_PARAMS = {
        'lookback_periods': [5, 10, 20, 60, 120],  # 回看期
        'rolling_windows': [5, 10, 20, 60],        # 滚动窗口
        'decay_factors': [0.9, 0.95, 0.99],       # 衰减因子
    }
    
    # ==================== 模型配置 ====================
    # XGBoost模型参数
    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # 模型训练配置
    MODEL_TRAINING = {
        'train_start': '2018-01-01',
        'train_end': '2023-12-31',
        'validation_split': 0.2,
        'test_split': 0.1,
        'cv_folds': 5,
        'rebalance_freq': 'monthly',  # daily, weekly, monthly, quarterly
        'retrain_freq': 'quarterly'   # monthly, quarterly, yearly
    }
    
    # 特征选择配置
    FEATURE_SELECTION = {
        'method': 'importance',    # importance, correlation, mutual_info
        'max_features': 50,        # 最大特征数
        'min_importance': 0.001,   # 最小重要性阈值
        'correlation_threshold': 0.95  # 相关性阈值
    }
    
    # ==================== 策略配置 ====================
    # 信号生成配置
    SIGNAL_CONFIG = {
        'prediction_horizon': 20,   # 预测期(天)
        'signal_threshold': 0.02,   # 信号阈值
        'signal_decay': 0.95,       # 信号衰减
        'min_signal_strength': 0.1  # 最小信号强度
    }
    
    # 组合优化配置
    PORTFOLIO_CONFIG = {
        'max_position': 0.05,       # 单股最大仓位
        'min_position': 0.01,       # 单股最小仓位
        'max_stocks': 50,           # 最大持股数
        'min_stocks': 20,           # 最小持股数
        'turnover_limit': 0.3,      # 换手率限制
        'sector_limit': 0.3         # 行业集中度限制
    }
    
    # 风险控制配置
    RISK_CONFIG = {
        'max_drawdown': 0.15,       # 最大回撤
        'stop_loss': 0.08,          # 止损线
        'var_limit': 0.02,          # VaR限制
        'beta_neutral': False,      # 是否市场中性
        'sector_neutral': False     # 是否行业中性
    }
    
    # ==================== 回测配置 ====================
    BACKTEST_CONFIG = {
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'initial_capital': 10000000,  # 初始资金(元)
        'commission': 0.0003,         # 手续费率
        'slippage': 0.001,           # 滑点
        'benchmark': '000300.SH',     # 基准指数
        'rebalance_freq': 'monthly'   # 调仓频率
    }
    
    # ==================== 实盘配置 ====================
    TRADING_CONFIG = {
        'broker': 'simulation',      # simulation, ths, citic, etc.
        'account_id': '',
        'api_key': '',
        'secret_key': '',
        'trading_mode': 'paper',     # paper, live
        'order_type': 'market',      # market, limit
        'max_order_value': 1000000   # 单笔最大订单金额
    }
    
    # ==================== 监控配置 ====================
    MONITORING_CONFIG = {
        'enable_alerts': True,
        'alert_channels': ['email', 'wechat'],
        'performance_check_freq': 'daily',
        'risk_check_freq': 'realtime',
        'model_drift_threshold': 0.1
    }
    
    # ==================== 日志配置 ====================
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': './logs/',
        'max_file_size': '10MB',
        'backup_count': 5
    }

    @classmethod
    def get_data_path(cls, data_type: str) -> str:
        """获取数据路径"""
        return cls.DATA_PATH.get(data_type, './data/')
    
    @classmethod
    def get_trading_dates(cls, start_date: str = None, end_date: str = None) -> List[str]:
        """获取交易日期列表"""
        # 这里应该根据交易日历返回实际的交易日期
        # 暂时返回示例日期
        return ['2024-01-01', '2024-01-02']  # 实际实现需要交易日历
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置有效性"""
        # 检查必要的配置项
        required_paths = ['raw_data', 'processed_data', 'factor_data']
        for path_key in required_paths:
            if path_key not in cls.DATA_PATH:
                return False
        return True

# 创建配置实例
config = Config()

# 验证配置
if not config.validate_config():
    raise ValueError("配置验证失败，请检查配置文件")
