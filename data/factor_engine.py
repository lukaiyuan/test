"""
因子计算引擎
实现各类量化因子的计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from config.factor_config import FactorConfig
from utils.logger import LoggerManager
from utils.helpers import timer, winsorize, standardize

logger = LoggerManager.get_data_logger()

class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self):
        """初始化因子引擎"""
        self.factor_config = FactorConfig()
        logger.info("因子计算引擎初始化完成")
    
    @timer
    def calculate_technical_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术因子
        
        Args:
            data: 包含OHLCV数据的DataFrame
        
        Returns:
            技术因子DataFrame
        """
        logger.info("开始计算技术因子")
        
        factors = pd.DataFrame(index=data.index)
        
        # 趋势因子
        factors = pd.concat([factors, self._calculate_trend_factors(data)], axis=1)
        
        # 动量因子
        factors = pd.concat([factors, self._calculate_momentum_factors(data)], axis=1)
        
        # 波动率因子
        factors = pd.concat([factors, self._calculate_volatility_factors(data)], axis=1)
        
        # 成交量因子
        factors = pd.concat([factors, self._calculate_volume_factors(data)], axis=1)
        
        logger.info(f"技术因子计算完成，共 {factors.shape[1]} 个因子")
        return factors
    
    def _calculate_trend_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算趋势因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 移动平均线
        ma_config = self.factor_config.TECHNICAL_FACTORS['trend']['ma']
        for period in ma_config['periods']:
            # 简单移动平均
            factors[f'sma_{period}'] = data['close'].rolling(period).mean()
            # 指数移动平均
            factors[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            # 价格相对移动平均的位置
            factors[f'price_to_sma_{period}'] = data['close'] / factors[f'sma_{period}'] - 1
        
        # MACD
        if TALIB_AVAILABLE:
            macd_config = self.factor_config.TECHNICAL_FACTORS['trend']['macd']
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'].values,
                fastperiod=macd_config['fast_period'],
                slowperiod=macd_config['slow_period'],
                signalperiod=macd_config['signal_period']
            )
            factors['macd'] = macd
            factors['macd_signal'] = macd_signal
            factors['macd_hist'] = macd_hist
        else:
            # 手动计算MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            factors['macd'] = ema12 - ema26
            factors['macd_signal'] = factors['macd'].ewm(span=9).mean()
            factors['macd_hist'] = factors['macd'] - factors['macd_signal']
        
        return factors
    
    def _calculate_momentum_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算动量因子"""
        factors = pd.DataFrame(index=data.index)
        
        # RSI
        rsi_config = self.factor_config.TECHNICAL_FACTORS['momentum']['rsi']
        for period in rsi_config['periods']:
            if TALIB_AVAILABLE:
                factors[f'rsi_{period}'] = talib.RSI(data['close'].values, timeperiod=period)
            else:
                # 手动计算RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                factors[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 动量因子
        momentum_config = self.factor_config.TECHNICAL_FACTORS['momentum']['momentum']
        for period in momentum_config['periods']:
            factors[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # ROC (Rate of Change)
        roc_config = self.factor_config.TECHNICAL_FACTORS['momentum']['roc']
        for period in roc_config['periods']:
            factors[f'roc_{period}'] = data['close'].pct_change(period)
        
        return factors
    
    def _calculate_volatility_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率因子"""
        factors = pd.DataFrame(index=data.index)
        
        # ATR (Average True Range)
        if TALIB_AVAILABLE:
            atr_period = self.factor_config.TECHNICAL_FACTORS['volatility']['atr']['period']
            factors['atr'] = talib.ATR(data['high'].values, data['low'].values, 
                                     data['close'].values, timeperiod=atr_period)
        else:
            # 手动计算ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            factors['atr'] = true_range.rolling(14).mean()
        
        # 布林带
        bb_config = self.factor_config.TECHNICAL_FACTORS['volatility']['bollinger']
        period = bb_config['period']
        std_dev = bb_config['std_dev']
        
        sma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        factors['bb_upper'] = sma + (std * std_dev)
        factors['bb_lower'] = sma - (std * std_dev)
        factors['bb_width'] = (factors['bb_upper'] - factors['bb_lower']) / sma
        factors['bb_position'] = (data['close'] - factors['bb_lower']) / (factors['bb_upper'] - factors['bb_lower'])
        
        # 历史波动率
        hv_config = self.factor_config.TECHNICAL_FACTORS['volatility']['historical_volatility']
        for period in hv_config['periods']:
            returns = data['close'].pct_change()
            factors[f'hv_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        return factors
    
    def _calculate_volume_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量因子"""
        factors = pd.DataFrame(index=data.index)
        
        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        factors['obv'] = obv
        
        # VWAP (Volume Weighted Average Price)
        vwap_config = self.factor_config.TECHNICAL_FACTORS['volume']['vwap']
        for period in vwap_config['periods']:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(period).sum() / data['volume'].rolling(period).sum()
            factors[f'vwap_{period}'] = vwap
            factors[f'price_to_vwap_{period}'] = data['close'] / vwap - 1
        
        # 成交量比率
        volume_ratio_config = self.factor_config.TECHNICAL_FACTORS['volume']['volume_ratio']
        for period in volume_ratio_config['periods']:
            avg_volume = data['volume'].rolling(period).mean()
            factors[f'volume_ratio_{period}'] = data['volume'] / avg_volume
        
        # 成交量移动平均
        for period in [5, 10, 20, 60]:
            factors[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()
            factors[f'volume_std_{period}'] = data['volume'].rolling(period).std()
        
        return factors
    
    def calculate_fundamental_factors(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本面因子
        
        Args:
            financial_data: 财务数据DataFrame
        
        Returns:
            基本面因子DataFrame
        """
        logger.info("开始计算基本面因子")
        
        factors = pd.DataFrame(index=financial_data.index)
        
        # 估值因子
        factors = pd.concat([factors, self._calculate_valuation_factors(financial_data)], axis=1)
        
        # 盈利能力因子
        factors = pd.concat([factors, self._calculate_profitability_factors(financial_data)], axis=1)
        
        # 成长性因子
        factors = pd.concat([factors, self._calculate_growth_factors(financial_data)], axis=1)
        
        # 财务质量因子
        factors = pd.concat([factors, self._calculate_quality_factors(financial_data)], axis=1)
        
        logger.info(f"基本面因子计算完成，共 {factors.shape[1]} 个因子")
        return factors
    
    def _calculate_valuation_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算估值因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 直接使用已有的估值指标
        if 'pe_ratio' in data.columns:
            factors['pe_ratio'] = data['pe_ratio']
        if 'pb_ratio' in data.columns:
            factors['pb_ratio'] = data['pb_ratio']
        
        # 计算其他估值指标
        if 'revenue' in data.columns and 'market_cap' in data.columns:
            factors['ps_ratio'] = data['market_cap'] / data['revenue']
        
        return factors
    
    def _calculate_profitability_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算盈利能力因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 直接使用已有的盈利能力指标
        if 'roe' in data.columns:
            factors['roe'] = data['roe']
        if 'roa' in data.columns:
            factors['roa'] = data['roa']
        
        # 计算毛利率
        if 'revenue' in data.columns and 'cost_of_goods_sold' in data.columns:
            factors['gross_margin'] = (data['revenue'] - data['cost_of_goods_sold']) / data['revenue']
        
        return factors
    
    def _calculate_growth_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成长性因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 收入增长率
        if 'revenue' in data.columns:
            factors['revenue_growth_1y'] = data['revenue'].pct_change(4)  # 年度增长
            factors['revenue_growth_3y'] = data['revenue'].pct_change(12)  # 3年增长
        
        # 净利润增长率
        if 'net_income' in data.columns:
            factors['earnings_growth_1y'] = data['net_income'].pct_change(4)
            factors['earnings_growth_3y'] = data['net_income'].pct_change(12)
        
        return factors
    
    def _calculate_quality_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算财务质量因子"""
        factors = pd.DataFrame(index=data.index)
        
        # 资产负债率
        if 'total_debt' in data.columns and 'total_equity' in data.columns:
            factors['debt_to_equity'] = data['total_debt'] / data['total_equity']
        
        # 流动比率
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            factors['current_ratio'] = data['current_assets'] / data['current_liabilities']
        
        return factors
    
    def preprocess_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        因子预处理
        
        Args:
            factors: 原始因子DataFrame
        
        Returns:
            预处理后的因子DataFrame
        """
        logger.info("开始因子预处理")
        
        processed_factors = factors.copy()
        
        # 异常值处理
        outlier_config = self.factor_config.PREPROCESSING_CONFIG['outlier_treatment']
        if outlier_config['method'] == 'winsorize':
            for col in processed_factors.columns:
                processed_factors[col] = winsorize(
                    processed_factors[col],
                    lower=outlier_config['lower_quantile'],
                    upper=outlier_config['upper_quantile']
                )
        
        # 缺失值处理
        missing_config = self.factor_config.PREPROCESSING_CONFIG['missing_value_treatment']
        if missing_config['method'] == 'forward_fill':
            processed_factors = processed_factors.fillna(method='ffill')
        elif missing_config['method'] == 'backward_fill':
            processed_factors = processed_factors.fillna(method='bfill')
        
        # 标准化
        std_config = self.factor_config.PREPROCESSING_CONFIG['standardization']
        if std_config['cross_sectional']:
            # 截面标准化
            processed_factors = processed_factors.apply(
                lambda x: standardize(x.to_frame(), method=std_config['method']).iloc[:, 0],
                axis=0
            )
        
        logger.info(f"因子预处理完成，处理了 {processed_factors.shape[1]} 个因子")
        return processed_factors
    
    def calculate_all_factors(self, price_data: pd.DataFrame, 
                            financial_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            price_data: 价格数据
            financial_data: 财务数据
        
        Returns:
            所有因子DataFrame
        """
        logger.info("开始计算所有因子")
        
        all_factors = pd.DataFrame(index=price_data.index)
        
        # 技术因子
        technical_factors = self.calculate_technical_factors(price_data)
        all_factors = pd.concat([all_factors, technical_factors], axis=1)
        
        # 基本面因子
        if financial_data is not None:
            fundamental_factors = self.calculate_fundamental_factors(financial_data)
            # 对齐时间索引
            fundamental_factors = fundamental_factors.reindex(price_data.index, method='ffill')
            all_factors = pd.concat([all_factors, fundamental_factors], axis=1)
        
        # 预处理
        all_factors = self.preprocess_factors(all_factors)
        
        logger.info(f"所有因子计算完成，共 {all_factors.shape[1]} 个因子")
        return all_factors
