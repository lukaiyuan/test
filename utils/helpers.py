"""
辅助函数模块
提供通用的工具函数
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path

from utils.logger import LoggerManager

logger = LoggerManager.get_logger("Helpers")

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """保存对象到pickle文件"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"对象已保存到: {filepath}")
    except Exception as e:
        logger.error(f"保存pickle文件失败: {e}")
        raise

def load_pickle(filepath: Union[str, Path]) -> Any:
    """从pickle文件加载对象"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"对象已从 {filepath} 加载")
        return obj
    except Exception as e:
        logger.error(f"加载pickle文件失败: {e}")
        raise

def save_json(obj: Dict, filepath: Union[str, Path]) -> None:
    """保存字典到JSON文件"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON已保存到: {filepath}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")
        raise

def load_json(filepath: Union[str, Path]) -> Dict:
    """从JSON文件加载字典"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"文件不存在: {filepath}")
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        logger.info(f"JSON已从 {filepath} 加载")
        return obj
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        raise

def get_trading_dates(start_date: str, end_date: str, 
                     freq: str = 'D') -> List[str]:
    """
    获取交易日期列表
    
    Args:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        freq: 频率 'D'(日), 'W'(周), 'M'(月)
    
    Returns:
        交易日期列表
    """
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
        return [date.strftime('%Y-%m-%d') for date in date_range]
    except Exception as e:
        logger.error(f"获取交易日期失败: {e}")
        return []

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    计算收益率
    
    Args:
        prices: 价格序列
        method: 计算方法 'simple'(简单收益率) 或 'log'(对数收益率)
    
    Returns:
        收益率序列
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")

def winsorize(data: pd.Series, lower: float = 0.01, 
              upper: float = 0.99) -> pd.Series:
    """
    缩尾处理
    
    Args:
        data: 数据序列
        lower: 下分位数
        upper: 上分位数
    
    Returns:
        处理后的数据序列
    """
    lower_bound = data.quantile(lower)
    upper_bound = data.quantile(upper)
    return data.clip(lower=lower_bound, upper=upper_bound)

def standardize(data: pd.DataFrame, method: str = 'z_score') -> pd.DataFrame:
    """
    数据标准化
    
    Args:
        data: 数据框
        method: 标准化方法 'z_score', 'min_max', 'robust'
    
    Returns:
        标准化后的数据框
    """
    if method == 'z_score':
        return (data - data.mean()) / data.std()
    elif method == 'min_max':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'robust':
        median = data.median()
        mad = (data - median).abs().median()
        return (data - median) / mad
    else:
        raise ValueError("method must be 'z_score', 'min_max', or 'robust'")

def calculate_ic(factor: pd.Series, returns: pd.Series, 
                method: str = 'pearson') -> float:
    """
    计算信息系数(IC)
    
    Args:
        factor: 因子值
        returns: 收益率
        method: 相关性计算方法
    
    Returns:
        IC值
    """
    if method == 'pearson':
        return factor.corr(returns)
    elif method == 'spearman':
        return factor.corr(returns, method='spearman')
    elif method == 'kendall':
        return factor.corr(returns, method='kendall')
    else:
        raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率(年化)
    
    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    计算最大回撤
    
    Args:
        cumulative_returns: 累计收益率序列
    
    Returns:
        最大回撤
    """
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """
    计算风险价值(VaR)
    
    Args:
        returns: 收益率序列
        confidence_level: 置信水平
    
    Returns:
        VaR值
    """
    return returns.quantile(confidence_level)

def format_number(number: float, decimal_places: int = 4) -> str:
    """格式化数字显示"""
    if abs(number) >= 1e9:
        return f"{number/1e9:.{decimal_places}f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.{decimal_places}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{decimal_places}f}K"
    else:
        return f"{number:.{decimal_places}f}"

def format_percentage(number: float, decimal_places: int = 2) -> str:
    """格式化百分比显示"""
    return f"{number*100:.{decimal_places}f}%"

def timer(func):
    """装饰器：计算函数执行时间"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"{func.__name__} 执行时间: {execution_time:.2f}秒")
        return result
    return wrapper

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    验证DataFrame的有效性
    
    Args:
        df: 待验证的DataFrame
        required_columns: 必需的列名列表
    
    Returns:
        是否有效
    """
    if df is None or df.empty:
        logger.error("DataFrame为空")
        return False
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"缺少必需的列: {missing_columns}")
            return False
    
    return True

def memory_usage(df: pd.DataFrame) -> str:
    """获取DataFrame内存使用情况"""
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    return f"{memory_mb:.2f} MB"
