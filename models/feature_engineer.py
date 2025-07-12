"""
特征工程模块
负责特征选择、特征变换和特征组合
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from utils.logger import LoggerManager
from utils.helpers import timer, save_pickle, load_pickle

logger = LoggerManager.get_model_logger()

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.config = Config()
        self.scalers = {}
        self.feature_selectors = {}
        self.selected_features = []
        
        logger.info("特征工程器初始化完成")
    
    @timer
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'importance') -> List[str]:
        """
        特征选择
        
        Args:
            X: 特征数据
            y: 目标变量
            method: 选择方法 'importance', 'correlation', 'mutual_info', 'rfe'
        
        Returns:
            选中的特征列表
        """
        logger.info(f"开始特征选择，使用方法: {method}")
        
        # 移除非数值列
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_columns]
        
        if method == 'importance':
            selected_features = self._select_by_importance(X_numeric, y)
        elif method == 'correlation':
            selected_features = self._select_by_correlation(X_numeric, y)
        elif method == 'mutual_info':
            selected_features = self._select_by_mutual_info(X_numeric, y)
        elif method == 'rfe':
            selected_features = self._select_by_rfe(X_numeric, y)
        else:
            logger.warning(f"未知的特征选择方法: {method}，使用默认方法")
            selected_features = self._select_by_importance(X_numeric, y)
        
        self.selected_features = selected_features
        logger.info(f"特征选择完成，选中 {len(selected_features)} 个特征")
        
        return selected_features
    
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """基于特征重要性选择"""
        # 使用随机森林计算特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 获取特征重要性
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 选择重要性大于阈值的特征
        min_importance = self.config.FEATURE_SELECTION['min_importance']
        max_features = self.config.FEATURE_SELECTION['max_features']
        
        selected = importance_df[
            importance_df['importance'] >= min_importance
        ]['feature'].head(max_features).tolist()
        
        logger.info(f"基于重要性选择了 {len(selected)} 个特征")
        return selected
    
    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """基于相关性选择"""
        # 计算与目标变量的相关性
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # 移除高度相关的特征
        correlation_threshold = self.config.FEATURE_SELECTION['correlation_threshold']
        max_features = self.config.FEATURE_SELECTION['max_features']
        
        selected = []
        correlation_matrix = X.corr().abs()
        
        for feature in correlations.index:
            if len(selected) >= max_features:
                break
                
            # 检查与已选特征的相关性
            if not selected:
                selected.append(feature)
            else:
                max_corr = correlation_matrix.loc[feature, selected].max()
                if max_corr < correlation_threshold:
                    selected.append(feature)
        
        logger.info(f"基于相关性选择了 {len(selected)} 个特征")
        return selected
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """基于互信息选择"""
        max_features = self.config.FEATURE_SELECTION['max_features']
        
        selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        logger.info(f"基于互信息选择了 {len(selected)} 个特征")
        return selected
    
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """基于递归特征消除选择"""
        max_features = self.config.FEATURE_SELECTION['max_features']
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator, n_features_to_select=max_features)
        selector.fit(X, y)
        
        selected = X.columns[selector.support_].tolist()
        
        logger.info(f"基于RFE选择了 {len(selected)} 个特征")
        return selected
    
    def create_interaction_features(self, X: pd.DataFrame, 
                                  max_interactions: int = 10) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            X: 特征数据
            max_interactions: 最大交互特征数
        
        Returns:
            包含交互特征的数据
        """
        logger.info("开始创建交互特征")
        
        enhanced_X = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            logger.warning("数值特征不足，无法创建交互特征")
            return enhanced_X
        
        # 选择最重要的特征进行交互
        if hasattr(self, 'selected_features') and self.selected_features:
            important_features = [f for f in self.selected_features if f in numeric_columns][:5]
        else:
            important_features = numeric_columns[:5]
        
        interaction_count = 0
        
        for i, feat1 in enumerate(important_features):
            for feat2 in important_features[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # 乘积交互
                enhanced_X[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
                
                # 比值交互（避免除零）
                denominator = X[feat2].replace(0, np.nan)
                enhanced_X[f'{feat1}_div_{feat2}'] = X[feat1] / denominator
                
                interaction_count += 2
        
        logger.info(f"创建了 {interaction_count} 个交互特征")
        return enhanced_X
    
    def create_polynomial_features(self, X: pd.DataFrame, 
                                 degree: int = 2, max_features: int = 20) -> pd.DataFrame:
        """
        创建多项式特征
        
        Args:
            X: 特征数据
            degree: 多项式度数
            max_features: 最大特征数
        
        Returns:
            包含多项式特征的数据
        """
        logger.info(f"开始创建 {degree} 次多项式特征")
        
        enhanced_X = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # 选择最重要的特征
        if hasattr(self, 'selected_features') and self.selected_features:
            important_features = [f for f in self.selected_features if f in numeric_columns][:max_features]
        else:
            important_features = numeric_columns[:max_features]
        
        for feature in important_features:
            for d in range(2, degree + 1):
                enhanced_X[f'{feature}_pow_{d}'] = X[feature] ** d
        
        logger.info(f"创建了多项式特征，总特征数: {enhanced_X.shape[1]}")
        return enhanced_X
    
    def scale_features(self, X: pd.DataFrame, method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        特征缩放
        
        Args:
            X: 特征数据
            method: 缩放方法 'standard', 'minmax', 'robust'
            fit: 是否拟合缩放器
        
        Returns:
            缩放后的特征数据
        """
        logger.info(f"开始特征缩放，使用方法: {method}")
        
        scaled_X = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if method not in self.scalers or fit:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"未知的缩放方法: {method}，使用标准缩放")
                scaler = StandardScaler()
            
            if fit:
                scaler.fit(X[numeric_columns])
                self.scalers[method] = scaler
        else:
            scaler = self.scalers[method]
        
        # 应用缩放
        scaled_X[numeric_columns] = scaler.transform(X[numeric_columns])
        
        logger.info("特征缩放完成")
        return scaled_X
    
    def reduce_dimensions(self, X: pd.DataFrame, method: str = 'pca',
                         n_components: int = 50, fit: bool = True) -> pd.DataFrame:
        """
        降维
        
        Args:
            X: 特征数据
            method: 降维方法 'pca'
            n_components: 主成分数量
            fit: 是否拟合降维器
        
        Returns:
            降维后的特征数据
        """
        logger.info(f"开始降维，使用方法: {method}, 目标维度: {n_components}")
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_columns]
        
        if method == 'pca':
            if 'pca' not in self.feature_selectors or fit:
                pca = PCA(n_components=min(n_components, X_numeric.shape[1]))
                if fit:
                    pca.fit(X_numeric)
                    self.feature_selectors['pca'] = pca
            else:
                pca = self.feature_selectors['pca']
            
            # 应用PCA
            X_reduced = pca.transform(X_numeric)
            
            # 创建新的DataFrame
            pca_columns = [f'pca_{i}' for i in range(X_reduced.shape[1])]
            reduced_df = pd.DataFrame(X_reduced, columns=pca_columns, index=X.index)
            
            # 保留非数值列
            non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_columns:
                for col in non_numeric_columns:
                    reduced_df[col] = X[col]
            
            logger.info(f"PCA降维完成，解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
            return reduced_df
        
        else:
            logger.warning(f"未知的降维方法: {method}")
            return X
    
    def engineer_features(self, X: pd.DataFrame, y: pd.Series = None,
                         feature_selection: bool = True,
                         interaction_features: bool = True,
                         polynomial_features: bool = False,
                         scaling: bool = True,
                         dimensionality_reduction: bool = False) -> pd.DataFrame:
        """
        完整的特征工程流程
        
        Args:
            X: 特征数据
            y: 目标变量
            feature_selection: 是否进行特征选择
            interaction_features: 是否创建交互特征
            polynomial_features: 是否创建多项式特征
            scaling: 是否进行特征缩放
            dimensionality_reduction: 是否进行降维
        
        Returns:
            工程化后的特征数据
        """
        logger.info("开始完整特征工程流程")
        
        engineered_X = X.copy()
        
        # 1. 特征选择
        if feature_selection and y is not None:
            selected_features = self.select_features(engineered_X, y)
            # 保留选中的特征和非数值特征
            non_numeric_cols = engineered_X.select_dtypes(exclude=[np.number]).columns.tolist()
            keep_cols = selected_features + non_numeric_cols
            engineered_X = engineered_X[keep_cols]
        
        # 2. 创建交互特征
        if interaction_features:
            engineered_X = self.create_interaction_features(engineered_X)
        
        # 3. 创建多项式特征
        if polynomial_features:
            engineered_X = self.create_polynomial_features(engineered_X)
        
        # 4. 特征缩放
        if scaling:
            engineered_X = self.scale_features(engineered_X)
        
        # 5. 降维
        if dimensionality_reduction:
            engineered_X = self.reduce_dimensions(engineered_X)
        
        logger.info(f"特征工程完成，最终特征数: {engineered_X.shape[1]}")
        return engineered_X
    
    def save_feature_engineer(self, filepath: str):
        """保存特征工程器"""
        save_data = {
            'scalers': self.scalers,
            'feature_selectors': self.feature_selectors,
            'selected_features': self.selected_features
        }
        save_pickle(save_data, filepath)
        logger.info(f"特征工程器已保存到: {filepath}")
    
    def load_feature_engineer(self, filepath: str):
        """加载特征工程器"""
        save_data = load_pickle(filepath)
        self.scalers = save_data.get('scalers', {})
        self.feature_selectors = save_data.get('feature_selectors', {})
        self.selected_features = save_data.get('selected_features', [])
        logger.info(f"特征工程器已从 {filepath} 加载")
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            X: 特征数据
            y: 目标变量
        
        Returns:
            特征重要性DataFrame
        """
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_columns]
        
        # 使用随机森林计算重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_numeric, y)
        
        importance_df = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
