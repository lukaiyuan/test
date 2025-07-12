"""
XGBoost模型模块
实现XGBoost模型的训练、预测和优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from config.settings import Config
from utils.logger import LoggerManager
from utils.helpers import timer, save_pickle, load_pickle

logger = LoggerManager.get_model_logger()

class XGBoostModel:
    """XGBoost模型封装"""
    
    def __init__(self, params: Dict = None):
        """
        初始化XGBoost模型
        
        Args:
            params: 模型参数字典
        """
        self.config = Config()
        self.params = params or self.config.XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_importance = None
        self.training_history = {}
        
        logger.info("XGBoost模型初始化完成")
    
    @timer
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            early_stopping_rounds: 早停轮数
        
        Returns:
            训练结果字典
        """
        logger.info("开始训练XGBoost模型")
        
        # 准备数据
        train_data = xgb.DMatrix(X_train, label=y_train)
        eval_sets = [(train_data, 'train')]
        
        if X_val is not None and y_val is not None:
            val_data = xgb.DMatrix(X_val, label=y_val)
            eval_sets.append((val_data, 'val'))
        
        # 训练参数
        train_params = self.params.copy()
        train_params.update({
            'eval_metric': 'rmse',
            'early_stopping_rounds': early_stopping_rounds,
            'verbose_eval': False
        })
        
        # 训练模型
        evals_result = {}
        self.model = xgb.train(
            params=train_params,
            dtrain=train_data,
            num_boost_round=train_params.get('n_estimators', 1000),
            evals=eval_sets,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        
        # 保存训练历史
        self.training_history = evals_result
        
        # 计算特征重要性
        self.feature_importance = self.model.get_score(importance_type='weight')
        
        # 评估模型
        train_pred = self.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)
        
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)
        
        # 记录训练结果
        training_result = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iteration': self.model.best_iteration,
            'feature_importance': self.feature_importance
        }
        
        logger.info(f"模型训练完成，最佳迭代: {self.model.best_iteration}")
        logger.log_model_metrics("XGBoost训练", train_metrics)
        if val_metrics:
            logger.log_model_metrics("XGBoost验证", val_metrics)
        
        return training_result
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
        
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        test_data = xgb.DMatrix(X)
        predictions = self.model.predict(test_data)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率（用于分类任务）
        
        Args:
            X: 特征数据
        
        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if self.params.get('objective') not in ['binary:logistic', 'multi:softprob']:
            logger.warning("当前模型不是分类模型，返回预测值")
            return self.predict(X)
        
        test_data = xgb.DMatrix(X)
        probabilities = self.model.predict(test_data)
        
        return probabilities
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # 计算方向准确率
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
        metrics['direction_accuracy'] = direction_accuracy
        
        return metrics
    
    @timer
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame = None, y_val: pd.Series = None,
                            method: str = 'grid', n_iter: int = 50) -> Dict[str, Any]:
        """
        超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            method: 优化方法 'grid', 'random'
            n_iter: 随机搜索迭代次数
        
        Returns:
            最佳参数和结果
        """
        logger.info(f"开始超参数优化，使用方法: {method}")
        
        # 定义参数搜索空间
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # 创建XGBoost回归器
        xgb_regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 选择搜索方法
        if method == 'grid':
            search = GridSearchCV(
                estimator=xgb_regressor,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                estimator=xgb_regressor,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError(f"未知的优化方法: {method}")
        
        # 执行搜索
        search.fit(X_train, y_train)
        
        # 更新模型参数
        self.params.update(search.best_params_)
        
        # 使用最佳参数重新训练
        best_result = self.train(X_train, y_train, X_val, y_val)
        
        optimization_result = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'training_result': best_result
        }
        
        logger.info(f"超参数优化完成，最佳分数: {search.best_score_:.4f}")
        logger.info(f"最佳参数: {search.best_params_}")
        
        return optimization_result
    
    def get_feature_importance(self, importance_type: str = 'weight') -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 'weight', 'gain', 'cover'
        
        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_importance(self, max_num_features: int = 20, importance_type: str = 'weight'):
        """绘制特征重要性图"""
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance(importance_type)
            top_features = importance_df.head(max_num_features)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel(f'Feature Importance ({importance_type})')
            plt.title('XGBoost Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘制重要性图")
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 保存XGBoost模型
        model_path = filepath.replace('.pkl', '_xgb.json')
        self.model.save_model(model_path)
        
        # 保存其他信息
        model_info = {
            'params': self.params,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        save_pickle(model_info, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        # 加载XGBoost模型
        model_path = filepath.replace('.pkl', '_xgb.json')
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # 加载其他信息
        model_info = load_pickle(filepath)
        self.params = model_info.get('params', {})
        self.feature_importance = model_info.get('feature_importance', {})
        self.training_history = model_info.get('training_history', {})
        
        logger.info(f"模型已从 {filepath} 加载")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 目标变量
            cv_folds: 交叉验证折数
        
        Returns:
            交叉验证结果
        """
        logger.info(f"开始 {cv_folds} 折交叉验证")
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"训练第 {fold + 1} 折")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # 创建临时模型
            temp_model = XGBoostModel(self.params.copy())
            temp_model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # 预测和评估
            val_pred = temp_model.predict(X_val_fold)
            fold_metrics = self._calculate_metrics(y_val_fold, val_pred)
            cv_scores.append(fold_metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for metric in cv_scores[0].keys():
            avg_metrics[f'mean_{metric}'] = np.mean([score[metric] for score in cv_scores])
            avg_metrics[f'std_{metric}'] = np.std([score[metric] for score in cv_scores])
        
        cv_result = {
            'cv_scores': cv_scores,
            'avg_metrics': avg_metrics,
            'n_folds': cv_folds
        }
        
        logger.info("交叉验证完成")
        logger.log_model_metrics("交叉验证平均", avg_metrics)
        
        return cv_result
