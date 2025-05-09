import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import VotingRegressor, StackingRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def safe_scale(x):
    """标准化数据，处理标准差为0的情况"""
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 確保目錄存在
os.makedirs('DATA', exist_ok=True)

def add_lag_features(df, columns, lags):
    """添加滞后特征"""
    lag_features = {}
    for col in columns:
        for lag in lags:
            lag_features[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)
            if lag > 1:
                lag_features[f'{col}_lag_{lag}_diff'] = df.groupby('city')[col].shift(lag) - df.groupby('city')[col].shift(lag-1)
                # 添加变化率，处理除零情况
                pct_change = df.groupby('city')[col].pct_change(periods=lag)
                lag_features[f'{col}_lag_{lag}_pct_change'] = np.clip(pct_change, -10, 10)  # 限制变化率范围
    
    result = pd.concat([df, pd.DataFrame(lag_features)], axis=1)
    return result

def add_rolling_features(df, columns, windows):
    """添加移动平均特征"""
    rolling_features = {}
    for col in columns:
        for window in windows:
            # 使用groupby来确保不跨城市计算
            group = df.groupby('city')[col]
            
            # 基本统计量
            rolling_features[f'{col}_rolling_mean_{window}'] = group.rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            rolling_features[f'{col}_rolling_std_{window}'] = group.rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
            rolling_features[f'{col}_rolling_min_{window}'] = group.rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
            rolling_features[f'{col}_rolling_max_{window}'] = group.rolling(window=window, min_periods=1).max().reset_index(0, drop=True)
            
            # 只在窗口足够大时计算高阶统计量
            if window >= 3:
                rolling_features[f'{col}_rolling_skew_{window}'] = group.rolling(window=window, min_periods=3).skew().reset_index(0, drop=True)
            if window >= 4:
                rolling_features[f'{col}_rolling_kurt_{window}'] = group.rolling(window=window, min_periods=4).kurt().reset_index(0, drop=True)
    
    result = pd.concat([df, pd.DataFrame(rolling_features)], axis=1)
    # 将无穷大和NaN替换为0
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)
    return result

def add_seasonal_features(df):
    """添加季节性特征"""
    # 基本时间特征
    df['year'] = df['week_start_date'].dt.year
    df['month'] = df['week_start_date'].dt.month
    df['day'] = df['week_start_date'].dt.day
    df['dayofweek'] = df['week_start_date'].dt.dayofweek
    df['quarter'] = df['week_start_date'].dt.quarter
    
    # 季节性特征 - 只在season列不存在时添加
    if 'season' not in df.columns:
        df['season'] = pd.cut(df['month'], 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['spring', 'summer', 'autumn', 'winter'])
        df = pd.get_dummies(df, columns=['season'])
    
    # 更细致的周期性特征
    period_mappings = {
        'weekofyear': ('weekofyear', 52),
        'month': ('month', 12),
        'day': ('day', 31),
        'quarter': ('quarter', 4)
    }
    
    for period_name, (col_name, max_val) in period_mappings.items():
        if f'{period_name}_sin' not in df.columns:
            df[f'{period_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_val)
            df[f'{period_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_val)
    
    return df

def add_interaction_features(df):
    """添加交互特征"""
    # 标准化数值以避免数值过大
    def safe_scale(x):
        std = np.std(x)
        if std == 0:
            return x - np.mean(x)
        return (x - np.mean(x)) / std
    
    # 对关键特征进行标准化
    temp_scaled = safe_scale(df['reanalysis_air_temp_k'])
    humidity_scaled = safe_scale(df['reanalysis_relative_humidity_percent'])
    precip_scaled = safe_scale(df['precipitation_amt_mm'])
    
    # 基本交互
    df['temp_humidity'] = temp_scaled * humidity_scaled
    df['temp_precip'] = temp_scaled * precip_scaled
    df['humidity_precip'] = humidity_scaled * precip_scaled
    
    # NDVI交互
    ndvi_cols = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']
    df['ndvi_mean'] = df[ndvi_cols].mean(axis=1)
    df['ndvi_std'] = df[ndvi_cols].std(axis=1)
    
    # 温度相关交互
    temp_range = df['reanalysis_max_air_temp_k'] - df['reanalysis_min_air_temp_k']
    temp_range_scaled = safe_scale(temp_range)
    df['temp_range'] = temp_range_scaled
    df['temp_humidity_interaction'] = temp_range_scaled * humidity_scaled
    
    return df

def add_time_features(df):
    """添加更细致的时间特征"""
    # 基本时间特征
    df['year'] = df['week_start_date'].dt.year
    df['month'] = df['week_start_date'].dt.month
    df['week'] = df['week_start_date'].dt.isocalendar().week
    df['day'] = df['week_start_date'].dt.day
    
    # 添加周期性特征
    for col in ['month', 'week', 'day']:
        if f'{col}_sin' not in df.columns:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
    
    # 添加年份相对特征
    df['years_from_start'] = (df['year'] - df['year'].min())
    
    return df

def feature_engineering(df, city):
    """增强的特征工程"""
    # 处理缺失值
    df = df.copy()
    
    # 保存日期列用于后续处理
    week_start_date = df['week_start_date']
    
    # 天气相关特征
    weather_columns = [
        'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw',
        'precipitation_amt_mm', 'reanalysis_air_temp_k',
        'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
        'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
        'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
        'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg',
        'reanalysis_tdtr_k', 'station_avg_temp_c', 'station_diur_temp_rng_c',
        'station_max_temp_c', 'station_min_temp_c', 'station_precip_mm'
    ]
    
    # 根据城市选择不同的滞后期
    if city == 'sj':
        lags = [1, 2, 4, 8, 12, 26, 52]  # San Juan使用更多的滞后期
        windows = [4, 8, 12, 26]  # 更关注中长期趋势
    else:
        lags = [1, 2, 4, 8, 26]  # Iquitos使用较少的滞后期
        windows = [4, 8, 12]  # 更关注短期趋势
    
    # 首先处理数值列的缺失值
    df[weather_columns] = df[weather_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 添加滞后特征
    for col in weather_columns:
        for lag in lags:
            if f'{col}_lag_{lag}' not in df.columns:
                df[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)
    
    # 添加滚动统计特征
    for col in weather_columns:
        for window in windows:
            # 基本统计量
            if f'{col}_rolling_mean_{window}' not in df.columns:
                df[f'{col}_rolling_mean_{window}'] = df.groupby('city')[col].rolling(
                    window=window, min_periods=1).mean().reset_index(0, drop=True)
            if f'{col}_rolling_std_{window}' not in df.columns:
                df[f'{col}_rolling_std_{window}'] = df.groupby('city')[col].rolling(
                    window=window, min_periods=1).std().reset_index(0, drop=True)
    
    # 增强周期性特征
    df['year'] = week_start_date.dt.year
    df['month'] = week_start_date.dt.month
    df['weekofyear'] = week_start_date.dt.isocalendar().week
    
    # 添加更细致的周期性特征
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
    
    # 添加季节性特征（使用独热编码）
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['spring', 'summer', 'autumn', 'winter'])
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    df = df.drop('season', axis=1)  # 删除原始season列
    
    # 为San Juan添加额外的特征
    if city == 'sj':
        # 添加温度和湿度的交互特征
        temp_cols = ['reanalysis_air_temp_k', 'reanalysis_avg_temp_k']
        humidity_cols = ['reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg']
        
        for temp_col in temp_cols:
            for humidity_col in humidity_cols:
                df[f'interact_{temp_col}_{humidity_col}'] = df[temp_col] * df[humidity_col]
        
        # 添加NDVI指数的组合特征
        df['ndvi_mean'] = df[['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']].mean(axis=1)
        df['ndvi_std'] = df[['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']].std(axis=1)
    
    # 删除日期列
    if 'week_start_date' in df.columns:
        df = df.drop(['week_start_date'], axis=1)
    
    # 处理剩余的缺失值
    df = df.fillna(0)
    
    # 处理无穷大
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def select_features(X, y, feature_names):
    """使用多个模型进行特征选择"""
    # 将城市列转换为数值类型
    X = X.copy()
    X['city'] = X['city'].map({'sj': 0, 'iq': 1})
    
    # 确保不包含目标变量
    if 'total_cases' in X.columns:
        X = X.drop(['total_cases'], axis=1)
        feature_names = np.array([f for f in feature_names if f != 'total_cases'])
    
    # 初始化特征选择器
    selectors = {
        'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # 存储每个特征被选中的次数
    feature_scores = np.zeros(len(feature_names))
    
    # 使用每个模型进行特征选择
    for name, model in selectors.items():
        selector = SelectFromModel(model, prefit=False)
        selector.fit(X, y)
        feature_scores += selector.get_support().astype(int)
    
    # 选择被至少两个模型选中的特征
    selected_features = feature_names[feature_scores >= 2]
    
    return selected_features, feature_scores

def analyze_feature_importance(X, y, feature_names, city_name):
    """分析并可视化特征重要性"""
    # 确保不包含目标变量
    if 'total_cases' in X.columns:
        X = X.drop(['total_cases'], axis=1)
        feature_names = np.array([f for f in feature_names if f != 'total_cases'])
    
    # 训练LightGBM模型
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 获取特征重要性
    importance = model.feature_importances_
    
    # 创建特征重要性DataFrame
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 绘制前20个最重要的特征
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), feature_imp['importance'][:20])
    plt.xticks(range(20), feature_imp['feature'][:20], rotation=45, ha='right')
    plt.title(f'{city_name} 特征重要性分析')
    plt.tight_layout()
    plt.savefig(f'{city_name.lower().replace(" ", "_")}_feature_importance.png')
    plt.close()
    
    return feature_imp

def train_city_model(city_train, city_test, city_name):
    """为每个城市训练单独的模型"""
    # 直接从数据中获取城市代码
    city = city_train['city'].iloc[0]
    
    # 特征工程
    X = feature_engineering(city_train.copy(), city)
    y = np.log1p(city_train['total_cases'])
    X_test = feature_engineering(city_test.copy(), city)
    
    # 将城市列转换为数值类型
    X = X.copy()
    X['city'] = X['city'].map({'sj': 0, 'iq': 1})
    X_test = X_test.copy()
    X_test['city'] = X_test['city'].map({'sj': 0, 'iq': 1})
    
    # 确保删除所有非数值列和目标变量
    non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
    if len(non_numeric_cols) > 0:
        X = X.drop(non_numeric_cols, axis=1)
        X_test = X_test.drop(non_numeric_cols, axis=1)
    
    if 'total_cases' in X.columns:
        X = X.drop(['total_cases'], axis=1)
    
    # 特征选择
    feature_names = np.array(X.columns)
    selected_features, feature_scores = select_features(X, y, feature_names)
    
    # 特征重要性分析
    feature_imp = analyze_feature_importance(X, y, feature_names, city_name)
    print(f"\n{city_name} 前10个最重要特征:")
    print(feature_imp.head(10))
    
    # 使用选定的特征
    X = X[selected_features]
    X_test = X_test[selected_features]
    
    # 获取模型参数
    lgb_params, xgb_params, gbm_params, cat_params = get_model_params(city)
    
    # 创建基础模型管道
    lgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', lgb.LGBMRegressor(**lgb_params))
    ])
    
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', xgb.XGBRegressor(**xgb_params))
    ])
    
    gbm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', GradientBoostingRegressor(**gbm_params))
    ])
    
    cat_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', cb.CatBoostRegressor(**cat_params))
    ])
    
    # 创建元学习器
    if city == 'sj':
        meta_lgb = lgb.LGBMRegressor(
            learning_rate=0.005,
            n_estimators=200,
            num_leaves=40,
            feature_fraction=0.85,
            random_state=42,
            n_jobs=-1
        )
    else:
        meta_lgb = lgb.LGBMRegressor(
            learning_rate=0.01,
            n_estimators=100,
            num_leaves=31,
            feature_fraction=0.9,
            random_state=42,
            n_jobs=-1
        )
    
    # 创建堆叠模型
    stack = StackingRegressor(
        estimators=[
            ('lgb', lgb_pipe),
            ('xgb', xgb_pipe),
            ('gbm', gbm_pipe),
            ('cat', cat_pipe)
        ],
        final_estimator=meta_lgb,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=True,
        n_jobs=-1
    )
    
    # 使用时序交叉验证
    if city == 'sj':
        tscv = TimeSeriesSplit(n_splits=5, test_size=52)  # 使用一年作为验证集
    else:
        tscv = TimeSeriesSplit(n_splits=5, test_size=26)  # 使用半年作为验证集
    
    cv_scores = []
    oof_predictions = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练前的数据标准化
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        stack.fit(X_train_fold, y_train_fold)
        val_pred = stack.predict(X_val_fold)
        oof_predictions[val_idx] = val_pred
        
        fold_mae = mean_absolute_error(np.expm1(y_val_fold), np.expm1(val_pred))
        cv_scores.append(fold_mae)
        print(f"{city_name} Fold {fold} MAE: {fold_mae:.2f}")
    
    print(f"{city_name} 交叉驗證MAE: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})")
    print(f"{city_name} OOF R2 分數: {r2_score(y, oof_predictions):.4f}")
    
    # 训练最终模型
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    stack.fit(X_scaled, y)
    predictions = stack.predict(X_test_scaled)
    return np.expm1(predictions)

def get_model_params(city):
    """优化的模型参数"""
    if city == 'sj':
        # San Juan使用更复杂的模型配置
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'learning_rate': 0.005,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5,
            'min_data_in_leaf': 30,
            'min_gain_to_split': 0.02,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 8,
            'learning_rate': 0.005,
            'n_estimators': 3000,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.01
        }
        
        gbm_params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.005,
            'subsample': 0.85,
            'min_samples_split': 20,
            'min_samples_leaf': 10
        }
        
        cat_params = {
            'iterations': 3000,
            'learning_rate': 0.005,
            'depth': 8,
            'l2_leaf_reg': 5,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.85,
            'random_seed': 42,
            'verbose': False
        }
    else:
        # Iquitos使用较简单的模型配置
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'min_data_in_leaf': 15,
            'min_gain_to_split': 0.01,
            'max_depth': 6,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'verbose': -1
        }
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 2000,
            'min_child_weight': 2,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'gamma': 0.01
        }
        
        gbm_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.9,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }
        
        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.01,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.9,
            'random_seed': 42,
            'verbose': False
        }
    
    return lgb_params, xgb_params, gbm_params, cat_params

# 读取数据
train_data = pd.read_csv('DATA/dengue_features_train.csv')
test_data = pd.read_csv('DATA/dengue_features_test.csv')

# 数据预处理
train_data['week_start_date'] = pd.to_datetime(train_data['week_start_date'])
test_data['week_start_date'] = pd.to_datetime(test_data['week_start_date'])

# 添加季节性特征
train_data = add_seasonal_features(train_data)
test_data = add_seasonal_features(test_data)

# 分别为每个城市训练模型并预测
predictions = []
for city in ['sj', 'iq']:
    city_train = train_data[train_data['city'] == city]
    city_test = test_data[test_data['city'] == city]
    city_predictions = train_city_model(city_train, city_test, 'San Juan' if city == 'sj' else 'Iquitos')
    predictions.extend(city_predictions)

# 创建提交文件
submission = pd.DataFrame({
    'city': test_data['city'],
    'year': test_data['year'],
    'weekofyear': test_data['weekofyear'],
    'total_cases': predictions
}).astype({'total_cases': int})

# 保存结果
submission.to_csv('DATA/benchmark.csv', index=False)

# 绘制预测结果折线图
plt.figure(figsize=(15, 6))

# 分别绘制两个城市的预测结果
for city in ['sj', 'iq']:
    city_results = submission[submission['city'] == city]
    plt.plot(range(len(city_results)), city_results['total_cases'], label=city)

plt.title('登革热病例预测结果')
plt.xlabel('时间')
plt.ylabel('预测病例数')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图表
plt.savefig('prediction_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("預測完成，結果已保存到 DATA/benchmark.csv")
print("預測結果圖表已保存到 prediction_plot.png") 