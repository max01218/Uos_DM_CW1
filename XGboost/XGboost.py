# Cell 1
import pandas as pd
import numpy as np
# load the provided data
test_df = pd.read_csv('DATA/test_data.csv')
train_df = pd.read_csv('DATA/train_data_no_selection.csv')

# Cell 2
print(train_df.shape)
print(test_df.shape)

# Cell 3
# 2. 創造新特徵
# # (1) 季節特徵 (春:1, 夏:2, 秋:3, 冬:4)
# train_df["season"] = train_df["weekofyear"].apply(lambda x: (x % 52) // 13 + 1)
# test_df["season"] = test_df["weekofyear"].apply(lambda x: (x % 52) // 13 + 1)
# # (2) 植被指數的統計特徵
# ndvi_cols = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]
# train_df["ndvi_mean"] = train_df[ndvi_cols].mean(axis=1)
# train_df["ndvi_std"] = train_df[ndvi_cols].std(axis=1)
# test_df["ndvi_mean"] = test_df[ndvi_cols].mean(axis=1)
# test_df["ndvi_std"] = test_df[ndvi_cols].std(axis=1)
# # (3) 降水與溫度的交互特徵
# train_df["precip_temp_ratio"] = train_df["precipitation_amt_mm"] / (train_df["reanalysis_air_temp_k"] + 1)
# test_df["precip_temp_ratio"] = test_df["precipitation_amt_mm"] / (test_df["reanalysis_air_temp_k"] + 1)

# Cell 4
# 圣胡安数据分离
train_df_sj = train_df[train_df['city'] == 1]
train_df_iq = train_df[train_df['city'] == 0]
test_df_sj = test_df[test_df['city'] == 1]
test_df_iq = test_df[test_df['city'] == 0]

# Cell 5
print(train_df_sj.shape)
print(train_df_iq.shape)
print(test_df_sj.shape)
print(test_df_iq.shape)

# Cell 6
# 前移一周
train_df_sj['total_cases'] = train_df_sj['total_cases'].shift(-1)
train_df_iq['total_cases'] = train_df_iq['total_cases'].shift(-1)
# 刪除最後一個
train_df_sj = train_df_sj.iloc[:-1].reset_index(drop=True)
train_df_sj['total_cases'] = train_df_sj['total_cases'].astype(int)
train_df_iq = train_df_iq.iloc[:-1].reset_index(drop=True)
train_df_iq['total_cases'] = train_df_iq['total_cases'].astype(int)
# Remove `week_start_date` string.
train_df_sj.drop(['week_start_date','city'], axis=1, inplace=True)
train_df_iq.drop(['week_start_date','city'], axis=1, inplace=True)
test_df_sj.drop(['week_start_date','city'], axis=1, inplace=True)
test_df_iq.drop(['week_start_date','city'], axis=1, inplace=True)

# Cell 7
# 将时间序列数据分割为训练集和测试集
train_size = int(len(train_df_sj) * 0.8)

train_data_sj, test_data_sj = train_df_sj[:train_size], train_df_sj[train_size:]
print(train_data_sj.shape,test_data_sj.shape)

# Cell 9
target = ['total_cases']
X_train = train_data_sj.drop(columns=target)
y_train = train_data_sj[target]
X_test = test_data_sj.drop(columns=target)
y_test = test_data_sj[target]

# Cell 10
# 使用网格搜索进行超参数调优
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, HalvingGridSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
#分类器使用 xgboost
clf1 = xgb.XGBRegressor()
param_dist = {
    'n_estimators': [50, 100, 200, 500],  # 避免过大值
    'max_depth': [3, 5, 7, 9],  # 限制范围，防止过拟合
    'learning_rate': np.linspace(0.01, 0.3, 10),  # 避免梯度爆炸
    'subsample': np.linspace(0.7, 1.0, 5),  # 避免数据量过少
    'colsample_bytree': np.linspace(0.7, 1.0, 5),  # 控制特征采样
    'min_child_weight': [1, 3, 5, 7],  # 限制最小值，防止训练失败
    'gamma': np.linspace(0, 0.5, 5),  # 避免过度剪枝
    'reg_alpha': [0, 0.1, 1, 10],  # 限制 L1 正则化
    'reg_lambda': [0.1, 1, 10]  # 限制 L2 正则化
}

# 先使用 RandomizedSearchCV 进行快速搜索
random_search = RandomizedSearchCV(
    estimator=clf1, 
    param_distributions=param_dist,
    n_iter=50,  # 限制搜索次数，提高搜索效率
    cv=5, 
    scoring='neg_mean_squared_error',
    verbose=1, 
    n_jobs=-1,
    random_state=42
)

# 进行训练
random_search.fit(X_train, y_train)

# 获取最佳参数
best_params = random_search.best_params_
print("RandomizedSearch 最佳参数:", best_params)

# 在最佳参数附近进行精细搜索
fine_tune_param_dist = {
    'n_estimators': [best_params['n_estimators'] - 50, best_params['n_estimators'], best_params['n_estimators'] + 50],
    'max_depth': [max(best_params['max_depth'] - 1, 3), best_params['max_depth'], best_params['max_depth'] + 1],
    'learning_rate': np.linspace(max(best_params['learning_rate'] / 2, 0.01), min(best_params['learning_rate'] * 2, 1), 5),
    'subsample': np.linspace(best_params['subsample'] - 0.1, best_params['subsample'] + 0.1, 5),
    'colsample_bytree': np.linspace(best_params['colsample_bytree'] - 0.1, best_params['colsample_bytree'] + 0.1, 5),
    'min_child_weight': [max(best_params['min_child_weight'] - 1, 1), best_params['min_child_weight'], best_params['min_child_weight'] + 1],
    'gamma': [max(best_params['gamma'] - 0.1, 0), best_params['gamma'], best_params['gamma'] + 0.1],
    'reg_alpha': [best_params['reg_alpha'] / 2, best_params['reg_alpha'], best_params['reg_alpha'] * 2],
    'reg_lambda': [best_params['reg_lambda'] / 2, best_params['reg_lambda'], best_params['reg_lambda'] * 2]
}

halving_search = HalvingGridSearchCV(
    estimator=clf1, 
    param_grid=fine_tune_param_dist,
    cv=5, 
    scoring='neg_mean_squared_error',
    verbose=1, 
    n_jobs=-1,
    factor = 2 #每次縮小2倍範圍
)
halving_search.fit(X_train, y_train)
best_params = halving_search.best_params_
print("best_para:", best_params)

# Cell 11
from xgboost import XGBRegressor

xgb_model_sj = XGBRegressor(**best_params)

xgb_model_sj.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error

predictions = xgb_model_sj.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(mae)

import matplotlib.pyplot as plt
# 視覺化預測結果
plt.figure(figsize=(12, 6))
plt.plot(y_test.to_numpy(), label="Actual Cases", color='blue')
plt.plot(predictions, label="Predicted Cases", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Total Cases")
plt.title("Xgboost Prediction Results(sj)")
plt.legend()
plt.show()

# Cell 12
feature_importance = xgb_model_sj.feature_importances_
important_features_sj = X_train.columns[feature_importance > 0.05]
X_train_feature =  X_train[important_features_sj]
X_test_feature = X_test[important_features_sj]
print(feature_importance)
print(important_features_sj)

# Cell 13
# 重新訓練一遍
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBRegressor
xgb_model_sj_best = XGBRegressor(**best_params)

xgb_model_sj_best.fit(X_train_feature, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error

predictions = xgb_model_sj_best.predict(X_test_feature)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(mae)

import matplotlib.pyplot as plt
# 視覺化預測結果
plt.figure(figsize=(12, 6))
plt.plot(y_test.to_numpy(), label="Actual Cases", color='blue')
plt.plot(predictions, label="Predicted Cases", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Total Cases")
plt.title("Xgboost Prediction Results")
plt.legend()
plt.show()

# Cell 14
# 将时间序列数据分割为训练集和测试集
train_size = int(len(train_df_iq) * 0.75)

train_data_iq, test_data_iq = train_df_iq[:train_size], train_df_iq[train_size:]
print(train_data_iq.shape,test_data_iq.shape)

# Cell 15
target = ['total_cases']
X_train = train_data_iq.drop(columns=target)
y_train = train_data_iq[target]
X_test = test_data_iq.drop(columns=target)
y_test = test_data_iq[target]

# Cell 16
# 使用网格搜索进行超参数调优
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
#分类器使用 xgboost
clf2 = xgb.XGBRegressor()
param_dist = {
    'n_estimators': range(100, 180, 10),  # 步長從 4 增加到 10
    'max_depth': range(3, 12, 2),  # 步長從 1 增加到 2
    'learning_rate': np.linspace(0.05, 1.5, 10),  # 減少點的數量
    'subsample': np.linspace(0.75, 0.85, 10),  # 減少點的數量
    'colsample_bytree': np.linspace(0.6, 0.9, 5),  # 減少點的數量
    'min_child_weight': range(1, 6, 2)  # 步長從 1 增加到 2
}

grid_search = GridSearchCV(
    clf2, param_dist,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("best_para:",best_params)

# Cell 17
from xgboost import XGBRegressor

xgb_model_iq = XGBRegressor(**best_params)

xgb_model_iq.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error

predictions = xgb_model_iq.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(mae)

import matplotlib.pyplot as plt
# 視覺化預測結果
plt.figure(figsize=(12, 6))
plt.plot(y_test.to_numpy(), label="Actual Cases", color='blue')
plt.plot(predictions, label="Predicted Cases", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Total Cases")
plt.title("Xgboost Prediction Results(iq)")
plt.legend()
plt.show()

# Cell 18
feature_importance = xgb_model_iq.feature_importances_
important_features = X_train.columns[feature_importance > 0.05]
X_train =  X_train[important_features]
X_test = X_test[important_features]
print(feature_importance)

# Cell 19
# 重新訓練一遍
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBRegressor
xgb_model_iq_best = XGBRegressor(**best_params)

xgb_model_iq_best.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error

predictions = xgb_model_iq_best.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(mae)

import matplotlib.pyplot as plt
# 視覺化預測結果
plt.figure(figsize=(12, 6))
plt.plot(y_test.to_numpy(), label="Actual Cases", color='blue')
plt.plot(predictions, label="Predicted Cases", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Total Cases")
plt.title("Xgboost Prediction Results")
plt.legend()
plt.show()

