"""
数据预处理模块
负责数据清洗、特征工程和数据准备
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from config.factor_config import FactorConfig
from utils.logger import LoggerManager
from utils.helpers import timer, validate_dataframe, calculate_returns
from data.data_loader import DataLoader
from data.factor_engine import FactorEngine

logger = LoggerManager.get_data_logger()

class DataProcessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化数据预处理器"""
        self.config = Config()
        self.factor_config = FactorConfig()
        self.data_loader = DataLoader()
        self.factor_engine = FactorEngine()
        
        logger.info("数据预处理器初始化完成")
    
    @timer
    def prepare_training_data(self, symbols: List[str], start_date: str, 
                            end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备训练数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            (features, targets) 特征数据和目标数据
        """
        logger.info(f"开始准备训练数据: {len(symbols)} 只股票, {start_date} 到 {end_date}")
        
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            try:
                # 获取价格数据
                price_data = self.data_loader.get_stock_data(symbol, start_date, end_date)
                
                if price_data.empty:
                    logger.warning(f"跳过 {symbol}: 无价格数据")
                    continue
                
                # 获取财务数据
                financial_data = self.data_loader.get_financial_data(symbol)
                
                # 计算因子
                factors = self.factor_engine.calculate_all_factors(price_data, financial_data)
                
                # 计算目标变量（未来收益率）
                targets = self._calculate_targets(price_data)
                
                # 对齐数据
                aligned_factors, aligned_targets = self._align_data(factors, targets)
                
                if not aligned_factors.empty and not aligned_targets.empty:
                    # 添加股票标识
                    aligned_factors['symbol'] = symbol
                    aligned_targets['symbol'] = symbol
                    
                    all_features.append(aligned_factors)
                    all_targets.append(aligned_targets)
                    
                    logger.debug(f"{symbol} 数据准备完成: {len(aligned_factors)} 条记录")
                
            except Exception as e:
                logger.error(f"处理 {symbol} 数据时出错: {e}")
                continue
        
        if not all_features:
            logger.error("没有成功处理任何股票数据")
            return pd.DataFrame(), pd.DataFrame()
        
        # 合并所有数据
        features_df = pd.concat(all_features, ignore_index=True)
        targets_df = pd.concat(all_targets, ignore_index=True)
        
        # 最终清洗
        features_df, targets_df = self._final_cleaning(features_df, targets_df)
        
        logger.info(f"训练数据准备完成: {len(features_df)} 条记录, {features_df.shape[1]} 个特征")
        
        return features_df, targets_df
    
    def _calculate_targets(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算目标变量
        
        Args:
            price_data: 价格数据
        
        Returns:
            目标变量DataFrame
        """
        targets = pd.DataFrame(index=price_data.index)
        
        # 计算不同期间的未来收益率
        prediction_horizons = [1, 5, 10, 20]  # 1天、5天、10天、20天
        
        for horizon in prediction_horizons:
            # 简单收益率
            future_returns = price_data['close'].shift(-horizon) / price_data['close'] - 1
            targets[f'return_{horizon}d'] = future_returns
            
            # 对数收益率
            log_returns = np.log(price_data['close'].shift(-horizon) / price_data['close'])
            targets[f'log_return_{horizon}d'] = log_returns
        
        # 计算相对收益率（相对于市场基准）
        # 这里使用简单的市场平均作为基准
        market_return = targets['return_20d'].rolling(window=60).mean()
        targets['excess_return_20d'] = targets['return_20d'] - market_return
        
        # 计算收益率分类标签
        targets['return_label'] = pd.cut(
            targets['return_20d'], 
            bins=[-np.inf, -0.05, 0.05, np.inf], 
            labels=['下跌', '横盘', '上涨']
        )
        
        return targets
    
    def _align_data(self, factors: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        对齐特征数据和目标数据
        
        Args:
            factors: 因子数据
            targets: 目标数据
        
        Returns:
            对齐后的(factors, targets)
        """
        # 找到共同的时间索引
        common_index = factors.index.intersection(targets.index)
        
        if len(common_index) == 0:
            logger.warning("特征数据和目标数据没有共同的时间索引")
            return pd.DataFrame(), pd.DataFrame()
        
        # 对齐数据
        aligned_factors = factors.loc[common_index]
        aligned_targets = targets.loc[common_index]
        
        # 移除包含NaN的行
        valid_mask = ~(aligned_factors.isna().any(axis=1) | aligned_targets.isna().any(axis=1))
        
        aligned_factors = aligned_factors[valid_mask]
        aligned_targets = aligned_targets[valid_mask]
        
        return aligned_factors, aligned_targets
    
    def _final_cleaning(self, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        最终数据清洗
        
        Args:
            features: 特征数据
            targets: 目标数据
        
        Returns:
            清洗后的(features, targets)
        """
        logger.info("开始最终数据清洗")
        
        # 移除无效值
        initial_count = len(features)
        
        # 移除包含无穷大值的行
        inf_mask = ~(np.isinf(features.select_dtypes(include=[np.number])).any(axis=1))
        features = features[inf_mask]
        targets = targets[inf_mask]
        
        # 移除包含NaN的行
        nan_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
        features = features[nan_mask]
        targets = targets[nan_mask]
        
        # 移除异常值（基于IQR方法）
        features = self._remove_outliers(features)
        targets = targets.loc[features.index]
        
        final_count = len(features)
        removed_count = initial_count - final_count
        
        logger.info(f"数据清洗完成: 移除 {removed_count} 条异常记录，保留 {final_count} 条记录")
        
        return features, targets
    
    def _remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            data: 数据DataFrame
            method: 异常值检测方法 'iqr', 'zscore'
        
        Returns:
            移除异常值后的数据
        """
        if method == 'iqr':
            # 使用IQR方法
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col == 'symbol':  # 跳过非数值列
                    continue
                    
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 标记异常值
                outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                data = data[~outlier_mask]
        
        elif method == 'zscore':
            # 使用Z-score方法
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col == 'symbol':
                    continue
                    
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > 3
                data = data[~outlier_mask]
        
        return data
    
    def create_time_series_features(self, data: pd.DataFrame, 
                                  lookback_periods: List[int] = None) -> pd.DataFrame:
        """
        创建时间序列特征
        
        Args:
            data: 原始数据
            lookback_periods: 回看期列表
        
        Returns:
            包含时间序列特征的数据
        """
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 60]
        
        logger.info("开始创建时间序列特征")
        
        enhanced_data = data.copy()
        
        # 为每个数值列创建滞后特征
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col == 'symbol':
                continue
                
            # 滞后特征
            for lag in lookback_periods:
                enhanced_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            # 滚动统计特征
            for window in lookback_periods:
                enhanced_data[f'{col}_mean_{window}'] = data[col].rolling(window).mean()
                enhanced_data[f'{col}_std_{window}'] = data[col].rolling(window).std()
                enhanced_data[f'{col}_min_{window}'] = data[col].rolling(window).min()
                enhanced_data[f'{col}_max_{window}'] = data[col].rolling(window).max()
                
                # 相对位置特征
                enhanced_data[f'{col}_rank_{window}'] = data[col].rolling(window).rank(pct=True)
        
        # 移除包含NaN的行（由于滞后和滚动计算产生）
        enhanced_data = enhanced_data.dropna()
        
        logger.info(f"时间序列特征创建完成: {enhanced_data.shape[1]} 个特征")
        
        return enhanced_data
    
    def split_data(self, features: pd.DataFrame, targets: pd.DataFrame,
                  train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        分割数据集
        
        Args:
            features: 特征数据
            targets: 目标数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        
        Returns:
            包含训练、验证、测试集的字典
        """
        logger.info("开始分割数据集")
        
        # 按时间顺序分割（避免未来信息泄露）
        n_samples = len(features)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        data_splits = {
            'X_train': features.iloc[:train_end],
            'y_train': targets.iloc[:train_end],
            'X_val': features.iloc[train_end:val_end],
            'y_val': targets.iloc[train_end:val_end],
            'X_test': features.iloc[val_end:],
            'y_test': targets.iloc[val_end:]
        }
        
        # 记录分割信息
        for split_name, split_data in data_splits.items():
            logger.info(f"{split_name}: {len(split_data)} 条记录")
        
        return data_splits
    
    def get_feature_importance_data(self, symbols: List[str], 
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取用于特征重要性分析的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            特征重要性分析数据
        """
        logger.info("准备特征重要性分析数据")
        
        features, targets = self.prepare_training_data(symbols, start_date, end_date)
        
        if features.empty:
            logger.error("无法获取特征重要性分析数据")
            return pd.DataFrame()
        
        # 选择主要目标变量
        target_col = 'return_20d'
        if target_col not in targets.columns:
            target_col = targets.columns[0]
        
        # 合并特征和目标
        analysis_data = features.copy()
        analysis_data['target'] = targets[target_col]
        
        # 移除非数值列
        numeric_columns = analysis_data.select_dtypes(include=[np.number]).columns
        analysis_data = analysis_data[numeric_columns]
        
        logger.info(f"特征重要性分析数据准备完成: {analysis_data.shape}")
        
        return analysis_data
