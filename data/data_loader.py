"""
数据获取模块
支持多种数据源的股票数据获取
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

from config.settings import Config
from utils.logger import LoggerManager
from utils.helpers import ensure_dir, save_pickle, load_pickle

logger = LoggerManager.get_data_logger()

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_source: str = None):
        """
        初始化数据加载器
        
        Args:
            data_source: 数据源 'tushare', 'yfinance', 'akshare'
        """
        self.data_source = data_source or Config.DATA_SOURCES['primary']
        self.cache_dir = ensure_dir(Config.get_data_path('raw_data'))
        
        # 初始化数据源
        self._init_data_source()
        
        logger.info(f"数据加载器初始化完成，使用数据源: {self.data_source}")
    
    def _init_data_source(self):
        """初始化数据源"""
        if self.data_source == 'tushare' and TUSHARE_AVAILABLE:
            # 这里需要设置tushare的token
            # ts.set_token('your_token_here')
            self.ts_pro = None  # ts.pro_api()
            logger.info("Tushare数据源初始化完成")
        elif self.data_source == 'yfinance' and YFINANCE_AVAILABLE:
            logger.info("YFinance数据源初始化完成")
        elif self.data_source == 'akshare' and AKSHARE_AVAILABLE:
            logger.info("AKShare数据源初始化完成")
        else:
            logger.warning(f"数据源 {self.data_source} 不可用，将使用模拟数据")
    
    def get_stock_list(self, market: str = 'all') -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            market: 市场类型 'all', 'main', 'sme', 'gem'
        
        Returns:
            股票列表DataFrame
        """
        cache_file = self.cache_dir / f"stock_list_{market}.pkl"
        
        # 检查缓存
        if cache_file.exists():
            try:
                stock_list = load_pickle(cache_file)
                logger.info(f"从缓存加载股票列表: {len(stock_list)} 只股票")
                return stock_list
            except:
                logger.warning("缓存文件损坏，重新获取数据")
        
        try:
            if self.data_source == 'tushare' and self.ts_pro:
                # 使用Tushare获取股票列表
                stock_list = self._get_stock_list_tushare(market)
            elif self.data_source == 'akshare' and AKSHARE_AVAILABLE:
                # 使用AKShare获取股票列表
                stock_list = self._get_stock_list_akshare(market)
            else:
                # 使用模拟数据
                stock_list = self._get_mock_stock_list()
            
            # 保存到缓存
            save_pickle(stock_list, cache_file)
            logger.info(f"获取股票列表完成: {len(stock_list)} 只股票")
            
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return self._get_mock_stock_list()
    
    def _get_stock_list_tushare(self, market: str) -> pd.DataFrame:
        """使用Tushare获取股票列表"""
        # 这里实现Tushare的股票列表获取逻辑
        # 由于需要token，暂时返回模拟数据
        return self._get_mock_stock_list()
    
    def _get_stock_list_akshare(self, market: str) -> pd.DataFrame:
        """使用AKShare获取股票列表"""
        try:
            # 获取A股股票列表
            stock_list = ak.stock_info_a_code_name()
            stock_list.columns = ['ts_code', 'name']
            stock_list['market'] = 'A股'
            return stock_list
        except Exception as e:
            logger.error(f"AKShare获取股票列表失败: {e}")
            return self._get_mock_stock_list()
    
    def _get_mock_stock_list(self) -> pd.DataFrame:
        """生成模拟股票列表"""
        mock_stocks = [
            ('000001.SZ', '平安银行', 'main'),
            ('000002.SZ', '万科A', 'main'),
            ('600000.SH', '浦发银行', 'main'),
            ('600036.SH', '招商银行', 'main'),
            ('600519.SH', '贵州茅台', 'main'),
            ('000858.SZ', '五粮液', 'main'),
            ('002415.SZ', '海康威视', 'sme'),
            ('300059.SZ', '东方财富', 'gem'),
            ('300750.SZ', '宁德时代', 'gem'),
            ('688981.SH', '中芯国际', 'star')
        ]
        
        return pd.DataFrame(mock_stocks, columns=['ts_code', 'name', 'market'])
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str,
                      fields: List[str] = None) -> pd.DataFrame:
        """
        获取股票价格数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            fields: 需要的字段列表
        
        Returns:
            股票数据DataFrame
        """
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.pkl"
        
        # 检查缓存
        if cache_file.exists():
            try:
                data = load_pickle(cache_file)
                logger.debug(f"从缓存加载 {symbol} 数据")
                return data[fields] if all(f in data.columns for f in fields) else data
            except:
                logger.warning(f"缓存文件损坏: {symbol}")
        
        try:
            if self.data_source == 'tushare' and self.ts_pro:
                data = self._get_stock_data_tushare(symbol, start_date, end_date)
            elif self.data_source == 'yfinance' and YFINANCE_AVAILABLE:
                data = self._get_stock_data_yfinance(symbol, start_date, end_date)
            elif self.data_source == 'akshare' and AKSHARE_AVAILABLE:
                data = self._get_stock_data_akshare(symbol, start_date, end_date)
            else:
                data = self._get_mock_stock_data(symbol, start_date, end_date)
            
            if not data.empty:
                save_pickle(data, cache_file)
                logger.debug(f"获取 {symbol} 数据完成: {len(data)} 条记录")
            
            return data[fields] if all(f in data.columns for f in fields) else data
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return self._get_mock_stock_data(symbol, start_date, end_date)
    
    def _get_stock_data_yfinance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用YFinance获取股票数据"""
        try:
            # 转换股票代码格式
            if symbol.endswith('.SH'):
                yf_symbol = symbol.replace('.SH', '.SS')
            elif symbol.endswith('.SZ'):
                yf_symbol = symbol.replace('.SZ', '.SZ')
            else:
                yf_symbol = symbol
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return self._get_mock_stock_data(symbol, start_date, end_date)
            
            # 重命名列
            data.columns = [col.lower() for col in data.columns]
            data = data.rename(columns={'adj close': 'adj_close'})
            data['amount'] = data['volume'] * data['close']  # 估算成交额
            
            return data.reset_index()
            
        except Exception as e:
            logger.error(f"YFinance获取 {symbol} 数据失败: {e}")
            return self._get_mock_stock_data(symbol, start_date, end_date)
    
    def _get_stock_data_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """使用AKShare获取股票数据"""
        try:
            # 转换股票代码格式
            code = symbol.split('.')[0]
            data = ak.stock_zh_a_hist(symbol=code, period="daily", 
                                    start_date=start_date.replace('-', ''), 
                                    end_date=end_date.replace('-', ''))
            
            if data.empty:
                return self._get_mock_stock_data(symbol, start_date, end_date)
            
            # 重命名列
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close', 
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            }
            data = data.rename(columns=column_mapping)
            data['date'] = pd.to_datetime(data['date'])
            
            return data
            
        except Exception as e:
            logger.error(f"AKShare获取 {symbol} 数据失败: {e}")
            return self._get_mock_stock_data(symbol, start_date, end_date)
    
    def _get_mock_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟股票数据"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(date_range)
        
        # 生成随机价格数据
        np.random.seed(hash(symbol) % 2**32)  # 确保同一股票的数据一致
        
        initial_price = 10 + np.random.random() * 90  # 10-100元
        returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 生成OHLC数据
        data = []
        for i, (date, close) in enumerate(zip(date_range, prices)):
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
        
        return pd.DataFrame(data)
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            指数数据DataFrame
        """
        # 使用股票数据获取逻辑，实际应该有专门的指数数据接口
        return self.get_stock_data(index_code, start_date, end_date)
    
    def get_financial_data(self, symbol: str, report_type: str = 'annual') -> pd.DataFrame:
        """
        获取财务数据
        
        Args:
            symbol: 股票代码
            report_type: 报告类型 'annual', 'quarterly'
        
        Returns:
            财务数据DataFrame
        """
        # 这里应该实现财务数据获取逻辑
        # 暂时返回模拟数据
        return self._get_mock_financial_data(symbol, report_type)
    
    def _get_mock_financial_data(self, symbol: str, report_type: str) -> pd.DataFrame:
        """生成模拟财务数据"""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='Q')
        
        data = []
        for date in dates:
            data.append({
                'date': date,
                'revenue': np.random.uniform(1e8, 1e10),
                'net_income': np.random.uniform(1e7, 1e9),
                'total_assets': np.random.uniform(1e9, 1e11),
                'total_equity': np.random.uniform(1e8, 1e10),
                'roe': np.random.uniform(0.05, 0.25),
                'roa': np.random.uniform(0.02, 0.15),
                'pe_ratio': np.random.uniform(10, 50),
                'pb_ratio': np.random.uniform(0.5, 5)
            })
        
        return pd.DataFrame(data)
