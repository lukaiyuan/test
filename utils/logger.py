"""
日志系统模块
提供统一的日志记录功能
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

from config.settings import Config

class QuantLogger:
    """量化交易日志系统"""
    
    def __init__(self, name: str = "QuantStrategy", level: str = None):
        """
        初始化日志系统
        
        Args:
            name: 日志器名称
            level: 日志级别
        """
        self.name = name
        self.level = level or Config.LOGGING_CONFIG['level']
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        # 创建日志器
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.upper()))
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        # 创建日志目录
        log_dir = Path(Config.LOGGING_CONFIG['file_path'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志格式
        formatter = logging.Formatter(Config.LOGGING_CONFIG['format'])
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器 - 所有日志
        all_log_file = log_dir / f"{self.name}_all.log"
        file_handler = logging.FileHandler(all_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 文件处理器 - 错误日志
        error_log_file = log_dir / f"{self.name}_error.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        # 文件处理器 - 交易日志
        if 'trading' in self.name.lower():
            trading_log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
            trading_handler = logging.FileHandler(trading_log_file, encoding='utf-8')
            trading_handler.setLevel(logging.INFO)
            trading_handler.setFormatter(formatter)
            logger.addHandler(trading_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def log_performance(self, metrics: dict):
        """记录性能指标"""
        self.info("=== 性能指标 ===")
        for key, value in metrics.items():
            self.info(f"{key}: {value}")
    
    def log_trade(self, trade_info: dict):
        """记录交易信息"""
        self.info("=== 交易记录 ===")
        for key, value in trade_info.items():
            self.info(f"{key}: {value}")
    
    def log_model_metrics(self, model_name: str, metrics: dict):
        """记录模型指标"""
        self.info(f"=== {model_name} 模型指标 ===")
        for key, value in metrics.items():
            self.info(f"{key}: {value}")
    
    def log_factor_analysis(self, factor_name: str, analysis: dict):
        """记录因子分析结果"""
        self.info(f"=== {factor_name} 因子分析 ===")
        for key, value in analysis.items():
            self.info(f"{key}: {value}")

class LoggerManager:
    """日志管理器"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, level: str = None) -> QuantLogger:
        """获取日志器实例"""
        if name not in cls._loggers:
            cls._loggers[name] = QuantLogger(name, level)
        return cls._loggers[name]
    
    @classmethod
    def get_data_logger(cls) -> QuantLogger:
        """获取数据模块日志器"""
        return cls.get_logger("DataModule")
    
    @classmethod
    def get_model_logger(cls) -> QuantLogger:
        """获取模型模块日志器"""
        return cls.get_logger("ModelModule")
    
    @classmethod
    def get_strategy_logger(cls) -> QuantLogger:
        """获取策略模块日志器"""
        return cls.get_logger("StrategyModule")
    
    @classmethod
    def get_backtest_logger(cls) -> QuantLogger:
        """获取回测模块日志器"""
        return cls.get_logger("BacktestModule")
    
    @classmethod
    def get_trading_logger(cls) -> QuantLogger:
        """获取交易模块日志器"""
        return cls.get_logger("TradingModule")

# 创建默认日志器
default_logger = LoggerManager.get_logger("QuantStrategy")

# 便捷函数
def log_info(message: str, **kwargs):
    """记录信息日志"""
    default_logger.info(message, **kwargs)

def log_error(message: str, **kwargs):
    """记录错误日志"""
    default_logger.error(message, **kwargs)

def log_warning(message: str, **kwargs):
    """记录警告日志"""
    default_logger.warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    """记录调试日志"""
    default_logger.debug(message, **kwargs)
