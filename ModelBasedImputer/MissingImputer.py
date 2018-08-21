import numpy as np 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import xgboost as xgb 
import lightgbm as lgb 


class MissingImputer(BaseEstimator, TransformerMixin):
	def __init__(self, max_iter = 10, ini_fill = True, ini_strategy_reg = 'mean', ini_strategy_clf = 'most_frequent', with_cat = False, cat_index = None, tol = 1e-3, model_reg = "knn", model_clf = "knn"):
		'''
		-max_iter:迭代次数
		-ini_fill：是否要进行简单填补(False仅对xgb和lgb有效)
		-ini_strategy_reg:连续变量简单填补规则, mean or median
		-ini_strategy_clf:离散变量简单填补规则, only most_frequent
		-cat_index:离散变量索引(int)
		-tol:阈值
		-model_reg:连续变量采用的预测缺失值模型, be xgboost,lightgbm, randomforest, knn
		-model_clf:离散变量采用的预测缺失值模型
		'''
		self.ini_fill = ini_fill
		self.max_iter = max_iter
		self.imputer_reg = Imputer(strategy = ini_strategy_reg)
		self.imputer_clf = Imputer(strategy = ini_strategy_clf)
		self.with_cat = with_cat
		self.cat_index = cat_index
		self.tol = tol
		self.model_reg = model_reg
		self.model_clf = model_clf
		if (not self.ini_fill) and (self.model_reg not in ('lightgbm', 'xgboost')) and (self.model_clf not in ('lightgbm', 'xgboost')):
			raise ValueError("ini_fill = False only work when model is lightgbm or xgboost")


	def fit(self, X, y = None, model_params = {'regressor':{}, 'classifier':{}}):
		'''
		-model_params:params for models,it should be a map
		'''
		X = check_array(X, dtype=np.float64, force_all_finite=False)

		if X.shape[1] == 1:
			raise ValueError("your X should have at least two features(predictiors)")

		#简单规则缺失值填补
		imputed_ini = X.copy() 
		if self.ini_fill:
			for i in np.arange(X.shape[1]):
				if self.with_cat and i in self.cat_index:
					imputed_ini[:, i:i+1] = self.imputer_clf.fit_transform(X[:, i].reshape(-1,1))
				else:
					imputed_ini[:, i:i+1] = self.imputer_reg.fit_transform(X[:, i].reshape(-1,1))

		#print('fit:imputed_ini')
		#print(imputed_ini)
		#将有缺失值的特征，按缺失值个数来先后预测
		X_nan = np.isnan(X)
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1]

		imputed_X = imputed_ini.copy()
		self.gamma_ = []
		#set model params
		if self.model_reg == 'xgboost' and self.model_clf == 'lightgbm':
			self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'xgboost' and self.model_clf == 'xgboost':
			self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'xgboost' and self.model_clf == 'randomforest':
			self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'xgboost' and self.model_clf == 'knn':
			self.estimators_ = [xgb.XGBClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]

		if self.model_reg == 'lightgbm' and self.model_clf == 'lightgbm':
			self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'lightgbm' and self.model_clf == 'xgboost':
			self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'lightgbm' and self.model_clf == 'randomforest':
			self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'lightgbm' and self.model_clf == 'knn':
			self.estimators_ = [lgb.LGBMClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]

		if self.model_reg == 'randomforest' and self.model_clf == 'lightgbm':
			self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'randomforest' and self.model_clf == 'xgboost':
			self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'randomforest' and self.model_clf == 'randomforest':
			self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'randomforest' and self.model_clf == 'knn':
			self.estimators_ = [RandomForestClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]

		if self.model_reg == 'knn' and self.model_clf == 'lightgbm':
			self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else lgb.LGBMRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'knn' and self.model_clf == 'xgboost':
			self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else xgb.XGBRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'knn' and self.model_clf == 'randomforest':
			self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.cat_index else RandomForestRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]
		if self.model_reg == 'knn' and self.model_clf == 'knn':
			self.estimators_ = [KNeighborsClassifier(**model_params['classifier']) if self.with_cat and i in self.scat_index else KNeighborsRegressor(**model_params['regressor']) for i in np.arange(X.shape[1])]

		#获取各列缺失值的bool值
		self.iter_ = 0
		self.estimators_ = self.estimators_ * self.max_iter
		for iter in np.arange(self.max_iter):
			for i in num_nan_desc:
				i_nan_index = X_nan[:, i]
				#查看该特征是否有缺失值
				if np.sum(i_nan_index) == 0:
					break

				X_1 = np.delete(imputed_X, i, 1)
				X_train = X_1[~i_nan_index]
				y_train = imputed_X[~i_nan_index, i]

				X_pre = X_1[i_nan_index]
				self.estimators_[iter*X.shape[1]+i].fit(X_train, y_train)

				imputed_X[i_nan_index, i] = self.estimators_[iter*X.shape[1]+i].predict(X_pre)

			self.iter_ += 1

			gamma = ((imputed_X-imputed_ini)**2/(1e-6+imputed_X.var(axis=0))).sum()/(1e-6+X_nan.sum())
			self.gamma_.append(gamma)
			if np.abs(np.diff(self.gamma_[-2:])) < self.tol:
				break
		#for test
		print(imputed_X)

		return self 

	def transform(self, X):
		X = check_array(X, dtype=np.float64, force_all_finite=False)

		if X.shape[1] == 1:
			raise ValueError("your X should have at least two features(predictiors)")

		
		#简单规则缺失值填补
		imputed_ini = X.copy() 
		if self.ini_fill:
			for i in np.arange(X.shape[1]):
				if self.with_cat and i in self.cat_index:
					imputed_ini[:, i:i+1] = self.imputer_clf.fit_transform(X[:, i].reshape(-1,1))
				else:
					imputed_ini[:, i:i+1] = self.imputer_reg.fit_transform(X[:, i].reshape(-1,1))

		#print('transform:imputed_ini')
		#print(imputed_ini)
		X_nan = np.isnan(X)
		num_nan_desc = X_nan.sum(axis=0).argsort()[::-1]

		for iter in np.arange(self.iter_):
			for i in num_nan_desc:
				i_nan_index = X_nan[:, i]
				if np.sum(i_nan_index) == 0:
					break

				X_1 = np.delete(imputed_ini, i, 1)
				X_pre = X_1[i_nan_index]

				imputed_ini[i_nan_index, i] = self.estimators_[iter*X.shape[1]+i].predict(X_pre)


		'''for i, estimators in enumerate(self.estimators_):
			i_nan_index = X_nan[:, i]
			if np.sum(i_nan_index) == 0:
				continue

			X_1 = np.delete(imputed_ini, i, 1)
			X_pre = X_1[i_nan_index]
			X[i_nan_index, i] = estimators.predict(X_pre)'''

		return imputed_ini




