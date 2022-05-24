#################################
# Store Item Demand Forecasting
#################################

# İş Problemi: Bir mağaza zinciri, 10 farklı mağazası ve 50 farklı ürünü için 3 aylık bir talep tahmini istemektedir.

# Veri Seti Hikayesi: bu veri seti farklı zaman serisi teknikleri denemek için sunulmuştur. Bir mağaza zincirinin
# 5 yıllık verilerinden 10 farklı mağazası ve 50 farklı ürünün bilgileri yer almaktadır.

# 4 Değişken, 958023 Gözlem

# data: Satış verilerinin tarihi (Tatil efekti veya mağaza kapanışı yoktur.
# store: Mağaza ID'si (Her bir mağaza için eşsiz numara)
# Item: Ürün ID'si (Her bir ürün için eşsiz numara)
# Sales: Satılan ürün sayıları (Belirli bir tarihte belirli bir mağazadan satılan ürünlerin sayısı)

#####################
# Görev
#####################
# Aşağıdaki zaman serisi ve makine öğrenmesi tekniklerini kullanarak ilgili mağaza zinciri için 3 aylık bir talep
# tahmin modeli oluşturunuz.
# - Random Noise
# - Log/Shifted Features
# - Rolling Mean Features
# - Exponentially Weighted Mean Features
# - Custom Cost Function (SMAPE)
# - LightGBM ile Model Validation

import pandas as pd
import numpy as np
from helpers.functions import *
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns

import holidays
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#####################
# EDA
#####################

train = pd.read_csv("week10/Hw10/train.csv")
test = pd.read_csv("week10/Hw10/test.csv")
df = pd.concat([train, test], sort=False)

check_df(df)

df.date = pd.to_datetime(df.date)

#####################
# Feature Engineering
#####################


def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week.astype(int)  # Series.dt.week have been deprecated.
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df


df = create_date_features(df)


us_holidays = holidays.US()
df["is_holiday"] = [1 if x in us_holidays else 0 for x in df.date]


df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

#####################
# Random Noise
#####################


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))


########################
# Lag/Shifted Features
########################


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


########################
# Rolling Mean Features
########################


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])

########################
# Exponentially Weighted Mean Features
########################


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

########################
# One-Hot Encoding
########################

df = one_hot_encoder(df, categorical_cols=['store', 'item', 'day_of_week', 'month'])

########################
# Converting sales to log(1+sales)
########################

df['sales'] = np.log1p(df["sales"].values)

########################
# Custom Cost Function
########################


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################


train = df.loc[(df["date"] < "2017-01-01"), :]
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ["date", "id", "sales", "year"]]

Y_train = train["sales"]
X_train = train[cols]

Y_val = val["sales"]
X_val = val[cols]


########################
# LightGBM ile Zaman Serisi Modeli
########################

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 15000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

start = time.time()
model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)
print("Time taken: ", time.time() - start)

# [10500]	training's l2: 0.0264932	training's SMAPE: 12.7741	valid_1's l2: 0.0300687	valid_1's SMAPE: 13.5257
# [10600]	training's l2: 0.0264795	training's SMAPE: 12.7713	valid_1's l2: 0.0300679	valid_1's SMAPE: 13.5256
# [10700]	training's l2: 0.0264666	training's SMAPE: 12.7685	valid_1's l2: 0.0300666	valid_1's SMAPE: 13.5251
# [10800]	training's l2: 0.0264525	training's SMAPE: 12.7656	valid_1's l2: 0.0300645	valid_1's SMAPE: 13.5244
# [10900]	training's l2: 0.0264386	training's SMAPE: 12.7628	valid_1's l2: 0.0300642	valid_1's SMAPE: 13.524
# [11000]	training's l2: 0.0264245	training's SMAPE: 12.7599	valid_1's l2: 0.030064	valid_1's SMAPE: 13.5239
# Early stopping, best iteration is:
# [10882]	training's l2: 0.026441	training's SMAPE: 12.7632	valid_1's l2: 0.0300628	valid_1's SMAPE: 13.5238
# Time taken:  879.6191971302032

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))  # 13.52377

########################
# Değişken Önem Düzeyleri
########################


def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp


feat_imp = plot_lgb_importances(model, num=df.shape[1])

feat_imp["importance_score"] = feat_imp["gain"] * feat_imp["split"]

not_important = feat_imp[feat_imp["importance_score"] < 0.1]["feature"].values

imp_feats = [col for col in cols if col not in not_important]
len(imp_feats)

########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[imp_feats]

test = df.loc[df.sales.isna()]
X_test = test[imp_feats]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=imp_feats)

start = time.time()
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration, verbose_eval=100)
print("Time taken: ", time.time() - start)
# Time taken:  250.62407279014587

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


########################
# Submission File
########################

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand_Hw.csv", index=False)
# Kaggle Private Score: Private score: 12.87



