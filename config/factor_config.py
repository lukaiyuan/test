"""
因子配置文件
定义各类因子的计算参数和配置
"""

from typing import Dict, List, Any

class FactorConfig:
    """因子配置类"""
    
    # ==================== 技术因子配置 ====================
    TECHNICAL_FACTORS = {
        # 趋势因子
        'trend': {
            'ma': {
                'periods': [5, 10, 20, 60, 120],
                'types': ['sma', 'ema', 'wma']
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'adx': {
                'period': 14
            },
            'aroon': {
                'period': 14
            }
        },
        
        # 动量因子
        'momentum': {
            'rsi': {
                'periods': [6, 14, 21]
            },
            'stoch': {
                'k_period': 14,
                'd_period': 3
            },
            'williams_r': {
                'period': 14
            },
            'roc': {
                'periods': [5, 10, 20]
            },
            'momentum': {
                'periods': [5, 10, 20, 60]
            }
        },
        
        # 波动率因子
        'volatility': {
            'atr': {
                'period': 14
            },
            'bollinger': {
                'period': 20,
                'std_dev': 2
            },
            'keltner': {
                'period': 20,
                'atr_period': 10
            },
            'historical_volatility': {
                'periods': [10, 20, 60]
            }
        },
        
        # 成交量因子
        'volume': {
            'obv': {},
            'ad_line': {},
            'cmf': {
                'period': 20
            },
            'vwap': {
                'periods': [5, 10, 20]
            },
            'volume_ratio': {
                'periods': [5, 10, 20]
            }
        }
    }
    
    # ==================== 基本面因子配置 ====================
    FUNDAMENTAL_FACTORS = {
        # 估值因子
        'valuation': {
            'pe_ratio': {},
            'pb_ratio': {},
            'ps_ratio': {},
            'pcf_ratio': {},
            'ev_ebitda': {},
            'peg_ratio': {}
        },
        
        # 盈利能力因子
        'profitability': {
            'roe': {},
            'roa': {},
            'roic': {},
            'gross_margin': {},
            'operating_margin': {},
            'net_margin': {}
        },
        
        # 成长性因子
        'growth': {
            'revenue_growth': {
                'periods': ['1y', '3y', '5y']
            },
            'earnings_growth': {
                'periods': ['1y', '3y', '5y']
            },
            'book_value_growth': {
                'periods': ['1y', '3y', '5y']
            }
        },
        
        # 财务质量因子
        'quality': {
            'debt_to_equity': {},
            'current_ratio': {},
            'quick_ratio': {},
            'interest_coverage': {},
            'asset_turnover': {},
            'inventory_turnover': {}
        }
    }
    
    # ==================== 量价因子配置 ====================
    PRICE_VOLUME_FACTORS = {
        # 价格因子
        'price': {
            'returns': {
                'periods': [1, 5, 10, 20, 60, 120]
            },
            'log_returns': {
                'periods': [1, 5, 10, 20, 60, 120]
            },
            'price_relative': {
                'periods': [20, 60, 120]
            },
            'high_low_ratio': {
                'periods': [5, 10, 20]
            }
        },
        
        # 成交量因子
        'volume': {
            'volume_ma': {
                'periods': [5, 10, 20, 60]
            },
            'volume_std': {
                'periods': [5, 10, 20]
            },
            'turnover_rate': {
                'periods': [5, 10, 20]
            },
            'volume_price_trend': {
                'period': 20
            }
        },
        
        # 资金流因子
        'money_flow': {
            'mfi': {
                'period': 14
            },
            'accumulation_distribution': {},
            'ease_of_movement': {
                'period': 14
            },
            'force_index': {
                'period': 13
            }
        }
    }
    
    # ==================== 宏观因子配置 ====================
    MACRO_FACTORS = {
        # 市场因子
        'market': {
            'market_beta': {
                'period': 60
            },
            'market_correlation': {
                'period': 60
            },
            'relative_strength': {
                'period': 20
            }
        },
        
        # 行业因子
        'industry': {
            'industry_momentum': {
                'period': 20
            },
            'industry_relative': {
                'period': 60
            }
        }
    }
    
    # ==================== 因子预处理配置 ====================
    PREPROCESSING_CONFIG = {
        # 异常值处理
        'outlier_treatment': {
            'method': 'winsorize',  # winsorize, clip, remove
            'lower_quantile': 0.01,
            'upper_quantile': 0.99
        },
        
        # 缺失值处理
        'missing_value_treatment': {
            'method': 'forward_fill',  # forward_fill, backward_fill, interpolate, drop
            'max_missing_ratio': 0.3
        },
        
        # 标准化方法
        'standardization': {
            'method': 'z_score',  # z_score, min_max, robust, rank
            'cross_sectional': True,
            'time_series': False
        },
        
        # 中性化处理
        'neutralization': {
            'market_cap': True,
            'industry': True,
            'beta': False
        }
    }
    
    # ==================== 因子评估配置 ====================
    EVALUATION_CONFIG = {
        # IC分析
        'ic_analysis': {
            'periods': [1, 5, 10, 20],
            'method': 'pearson',  # pearson, spearman, kendall
            'rolling_window': 60
        },
        
        # 分层回测
        'layered_backtest': {
            'layers': 10,
            'rebalance_freq': 'monthly',
            'long_short': True
        },
        
        # 因子衰减分析
        'decay_analysis': {
            'max_periods': 20,
            'step': 1
        }
    }
    
    @classmethod
    def get_factor_list(cls, category: str = None) -> List[str]:
        """获取因子列表"""
        if category is None:
            # 返回所有因子
            all_factors = []
            for cat_config in [cls.TECHNICAL_FACTORS, cls.FUNDAMENTAL_FACTORS, 
                             cls.PRICE_VOLUME_FACTORS, cls.MACRO_FACTORS]:
                for sub_cat in cat_config.values():
                    all_factors.extend(sub_cat.keys())
            return all_factors
        
        # 返回特定类别的因子
        category_map = {
            'technical': cls.TECHNICAL_FACTORS,
            'fundamental': cls.FUNDAMENTAL_FACTORS,
            'price_volume': cls.PRICE_VOLUME_FACTORS,
            'macro': cls.MACRO_FACTORS
        }
        
        if category in category_map:
            factors = []
            for sub_cat in category_map[category].values():
                factors.extend(sub_cat.keys())
            return factors
        
        return []
    
    @classmethod
    def get_factor_params(cls, factor_name: str) -> Dict[str, Any]:
        """获取特定因子的参数"""
        # 搜索所有类别中的因子
        all_configs = [cls.TECHNICAL_FACTORS, cls.FUNDAMENTAL_FACTORS, 
                      cls.PRICE_VOLUME_FACTORS, cls.MACRO_FACTORS]
        
        for config in all_configs:
            for sub_cat in config.values():
                if factor_name in sub_cat:
                    return sub_cat[factor_name]
        
        return {}
    
    @classmethod
    def validate_factor_config(cls) -> bool:
        """验证因子配置的有效性"""
        # 检查必要的配置项
        required_configs = ['TECHNICAL_FACTORS', 'FUNDAMENTAL_FACTORS', 
                          'PRICE_VOLUME_FACTORS', 'PREPROCESSING_CONFIG']
        
        for config_name in required_configs:
            if not hasattr(cls, config_name):
                return False
        
        return True

# 创建因子配置实例
factor_config = FactorConfig()

# 验证配置
if not factor_config.validate_factor_config():
    raise ValueError("因子配置验证失败，请检查配置文件")
