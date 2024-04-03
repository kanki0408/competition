#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Google Driveと接続を行います。これを行うことで、Driveにあるデータにアクセスできるようになります。
# 下記セルを実行すると、Googleアカウントのログインを求められますのでログインしてください。
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# 作業フォルダへの移動を行います。
# 人によって作業場所が異なるので、その場合作業場所を変更してください。
import os
os.chdir('/content/drive/MyDrive/コンペ/参加中コンペ') #ここを変更。


# In[ ]:


pip install feature-engine


# In[ ]:


pip install feature-engine


# In[ ]:


import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submit.csv',index_col=0, header=None)
train.head()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import fetch_california_housing
train.hist(bins=30,figsize=(12,12))
plt.show()


# In[ ]:


import seaborn as sns
sns.set(style="darkgrid")
from sklearn.datasets import fetch_california_housing

#図のサイズを表している
plt.figure(figsize=(3,6))
#箱ひげ図を作成すると書いてある
sns.boxplot(y=train["Glucose"])
plt.title("Boxplot")
plt.show()


# In[ ]:


def plot_boxplot_and_hist(data,variable):
  #2つのMatplotlib.Axesからなる図(ax_boxとax_hist)
  #plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}) は、高さの比率が (0.50, 0.85) である2つのサブプロットを持つFigureオブジェクトを作成します。一方のサブプロットは箱ひげ図（ax_box）であり、もう一方はヒストグラム（ax_hist）です。sharex=True は、x軸を共有することを意味します。つまり、両方のサブプロットが同じx軸を共有します。
  #f, (ax_box, ax_hist) = ... のようにして、戻り値を f という変数に代入しています。これにより、作成された図全体 (f) とそれぞれのサブプロット (ax_box と ax_hist) にアクセスできます。
  f,(ax_box,ax_hist)=plt.subplots(2,sharex=True,
  gridspec_kw={"height_ratios":(0.50,0.85)})

  #グラフをそれぞれの軸に割り当てる
  #ヒストグラムと箱ひげ図の2つを作成している。またax=ax_boxとすることによってどこに描画するのかを決めている。
  sns.boxplot(x=data[variable],ax=ax_box)
  sns.histplot(data=data,x=variable,ax=ax_hist)

  #箱ひげ図のx軸のラベルを削除する
  ax_box.set(xlabel="")
  plt.title(variable)
  plt.show()


# In[ ]:


def find_limits(df,variable,fold):
  IQR = df[variable].quantile(0.75)-df[variable].quantile(0.25)
  lower_limit = df[variable].quantile(0.25)-(IQR*fold)
  upper_limit = df[variable].quantile(0.75)+(IQR*fold)
  return lower_limit,upper_limit


# In[ ]:


lower_limit,upper_limit = find_limits(train , "Glucose",2)
print(lower_limit,upper_limit)


# In[ ]:


outliers = np.where((train["Glucose"]>upper_limit) | (train["Glucose"]<lower_limit),True,False,)
outliers.sum()


# In[ ]:


from feature_engine.outliers import Winsorizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[ ]:


breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)
y = breast_cancer.target


# In[ ]:


def diagnostic_plots(df,variable):
  #図のサイズ
  plt.figure(figsize=(15,6))

  #作成された図をどの位置に表示するのかということを意味する。
  #1行2列のグリッドにおいて、1行目の左側（つまり1番目の位置）にサブプロットを配置することを意味します。
  plt.subplot(1,2,1)
  df[variable].value_counts().sort_index().plot.bar()
  plt.title(f"Hitogram of {variable}")

  plt.subplot(1,2,2)
  #variableというデータ実際の図と、正規分布の場合の図をプロットしている
  stats.probplot(df[variable],dist="norm",plot=plt)
  plt.title(f"Q-Q plot of {variable}")

  plt.show


# In[ ]:


diagnostic_plots(train,"Pregnancies")


# In[ ]:


def diagnostic_plots(df,variable):
  #図のサイズ
  plt.figure(figsize=(15,6))

  #作成された図をどの位置に表示するのかということを意味する。
  #1行2列のグリッドにおいて、1行目の左側（つまり1番目の位置）にサブプロットを配置することを意味します。
  plt.subplot(1,2,1)
  df[variable].hist(bins=30)
  plt.title(f"Hitogram of {variable}")

  plt.subplot(1,2,2)
  #variableというデータ実際の図と、正規分布の場合の図をプロットしている
  stats.probplot(df[variable],dist="norm",plot=plt)
  plt.title(f"Q-Q plot of {variable}")

  plt.show


# In[ ]:


train_x=train.drop(["Outcome"],axis=1)
train_y=train["Outcome"]
test_x = test.copy()


# In[ ]:


#相関関係の確認
train.corrwith(train["Outcome"])


# In[ ]:


train_x = train_x.drop(["index"],axis=1)
test_x = test_x.drop(["index"],axis=1)


# In[ ]:


lower_limit,upper_limit = find_limits(train_x,"Glucose",2)
lower_limit,upper_limit


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
t_train_x = poly.fit_transform(train_x)
t_test_x = poly.fit_transform(test_x)
t_train_x


# In[ ]:


poly.get_feature_names_out()


# In[ ]:


t_train_x = pd.DataFrame(t_train_x,columns=poly.get_feature_names_out())
t_test_x = pd.DataFrame(t_test_x,columns=poly.get_feature_names_out())


# In[ ]:


check = t_train_x
t_train_x["Outcome"] = train["Outcome"]


# In[ ]:


t_train_x.corrwith(t_train_x["Outcome"])


# In[ ]:


high_correlation_columns = []
corr_series = t_train_x.corrwith(t_train_x["Outcome"])
#corr_series.items()によってカラム名と値を取り出している
for column, correlation in corr_series.items():
    if abs(correlation) > 0.2:  # 絶対値が0.2よりも大きい場合
        high_correlation_columns.append(column)
OK = [ 'Pregnancies Glucose',
 'Pregnancies BloodPressure',
 'Pregnancies BMI',
 'Pregnancies Age',
 'Glucose BMI',
 'Glucose Age',
 'BloodPressure BMI',
 'BloodPressure Age',
 'BMI Age']
OK


# In[ ]:


o_train_x = train_x.copy()
o_test_x = test_x.copy()
o2_train_x = train_x.copy()
o2_test_x = test_x.copy()


# In[ ]:


o_train_x["Glucose"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["Glucose"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["Glucose"].min(),o_train_x["Glucose"].max()


# In[ ]:


lower_limit,upper_limit = find_limits(train_x,"BloodPressure",2)
lower_limit,upper_limit


# In[ ]:


"""o_train_x["BloodPressure"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_test_x["BloodPressure"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
o_train_x["BloodPressure"].min(),o_train_x["BloodPressure"].max()"""


# In[ ]:


OK


# In[ ]:


ok_train_x = o_train_x.copy()
ok_test_x = o_test_x.copy()
ok_train_x['Pregnancies BMI'] = t_train_x['Pregnancies BMI']
ok_test_x['Pregnancies BMI'] = t_test_x['Pregnancies BMI']


# In[ ]:


lower_limit,upper_limit = find_limits(ok_train_x,"Pregnancies BMI",2)
lower_limit,upper_limit
ok_train_x["Pregnancies BMI"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
ok_test_x["Pregnancies BMI"].clip(lower=lower_limit,upper=upper_limit,inplace=True)
ok_train_x["Pregnancies BMI"].min(),ok_train_x["Pregnancies BMI"].max()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


"""tree_train_x = poly.fit_transform(train_x)
tree_test_x = poly.fit_transform(test_x)"""
"""
"max_depth"は、XGBoostなどの決定木ベースのモデルのハイパーパラメータの1つであり、木の深さを指定します。
木の深さが深いほどモデルの複雑性が増し、過学習の可能性が高まります。
"""
param_grid = {"max_depth":[2,3,4,None]}
tree_model = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    cv=5,
    scoring="accuracy",
    param_grid=param_grid,
)


# In[ ]:


ok_train_x.head()


# In[ ]:


train_y.head()


# In[ ]:


"""variables = ["Age","Glucose"]
variables = ["Pregnancies","SkinThickness"]"""
variables = ["Insulin","Pregnancies"]
tree_model.fit(ok_train_x[variables],train_y)


# In[ ]:


ok2_train_x=ok_train_x.copy()
ok2_test_x=ok_test_x.copy()


# In[ ]:


ok2_train_x["new_feat"] = tree_model.predict(ok_train_x[variables])
ok2_test_x["new_feat"] = tree_model.predict(ok_test_x[variables])


# In[ ]:


"""from sklearn.preprocessing import StandardScaler
# 学習データに基づいて複数列の標準化を定義
scaler = StandardScaler()
scaler.fit(train_x)
scaler.fit(test_x)
# 変換後のデータで各列を置換
st_train_x = scaler.transform(train_x)
st_test_x = scaler.transform(test_x)

st_train_x=pd.DataFrame(st_train_x, columns=train_x.columns, index=train_x.index)
st_test_x=pd.DataFrame(st_test_x, columns=test_x.columns, index=test_x.index)"""


# In[ ]:


"""test_x2=test_x.copy()
train_x2=train_x.copy()
train_x2["Pregnancies BMI"] = ok_train_x["Pregnancies BMI"]
test_x2["Pregnancies BMI"] = ok_test_x["Pregnancies BMI"]
# 変換後のデータで各列を置換
train_x2['SkinThickness'] = np.log1p(train_x2['SkinThickness'])
train_x2['Insulin'] = np.log1p(train_x2['Insulin'])
train_x2['Age'] = np.log1p(train_x2['Age'])
train_x2['DiabetesPedigreeFunction'] = np.log1p(train_x2['DiabetesPedigreeFunction'])
train_x2['Pregnancies BMI'] = np.log1p(train_x2['Pregnancies BMI'])
test_x2['SkinThickness'] = np.log1p(test_x2['SkinThickness'])
test_x2['Insulin'] = np.log1p(test_x2['Insulin'])
test_x2['Age'] = np.log1p(test_x2['Age'])
test_x2['DiabetesPedigreeFunction'] = np.log1p(test_x2['DiabetesPedigreeFunction'])
test_x2['Pregnancies BMI'] = np.log1p(test_x2['Pregnancies BMI'])"""


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import log_loss , accuracy_score
import xgboost as xgb
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import log_loss , accuracy_score

scores_accuracy = []
scores_logloss =[]
#クロスバリデーションを行う
#学習データを4分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf=KFold(n_splits=3 , shuffle=True , random_state = 71)
for tr_idx,va_idx in kf.split(o_train_x):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=o_train_x.iloc[tr_idx],o_train_x.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
  #特徴量と目的変数をxgboostのデータ構造に変換する
  dtrain = xgb.DMatrix(tr_x,label=tr_y)
  dvalid = xgb.DMatrix(va_x, label = va_y)
  dtest = xgb.DMatrix(o_test_x)
  #ハイパーパラメータの設定
  #silent:1によってが学習中のメッセージを抑制するようになっている
  #random_stateをせっていすることによって再現性を保つことが出来るようにしている。
  #ベースラインのパラメータ
  params = {
      "booster":"gbtree",
      "objective": "binary:logistic",
      "eta":0.1,
      "gamma":0.0,
      "alpha":0.0,
      "lambda":1.0,
      "silent":1,
      "random_state":71,
      "eta":0.1,
      "max_depth":5,
      "min_child_weight":1,
      "subsample":0.8,
      "colsample_btree":0.8,
      "random_state":71,
            }

#パラメータの探索範囲
  param_space = {
      "min_child_weight" = hp.loguniform("min_child_weight",np.log(0.1),np.log(10)),
      "max_depth": hp.quniform("max_depth", 3,9,1),
      "subsample":hp.quniform("subsample",0.6,0.95,0.05),
      "colsample_bytree":hp.quniform("subsamle",0.6,0.95,0.05),
      "gamma":hp.loguniform("gamma",np.log(1e-8),np.log(1.0)),
      #余裕があればalpha,lambdaも調整する

  }

  num_round = 1000;

  watchlist = [(dtrain,"train"),(dvalid,"eval")]
  model = xgb.train(params,dtrain,num_round,evals=watchlist)

  va_pred = model.predict(dvalid)
  #loglossはロジスティック損失を表しており、ロジスティック損失は、確率予測の正確さを測るための指標のひとつで、誤差が大きいほど損失が指数関数的に大きくなる特徴があります。
  score = log_loss(va_y,va_pred)
  accuracy = accuracy_score(va_y,va_pred>0.5)
  scores_logloss.append(score)
  scores_accuracy.append(accuracy)

print(f"logloss:{np.mean(scores_logloss):.4f}")
print(f"accuracy:{np.mean(scores_accuracy):.4f}")

pred = model.predict(dtest)
pred_label=np.where(pred>0.5,1,0)
"""
logloss:0.4685
accuracy:0.8027

"""


# In[ ]:


"""from sklearn.linear_model import LogisticRegression
scores_accuracy = []
scores_logloss =[]
#クロスバリデーションを行う
#学習データを4分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf=KFold(n_splits=4 , shuffle=True , random_state = 71)
for tr_idx,va_idx in kf.split(train_x2):
  #学習データを学習データとバリデーションデータに分ける
  tr_x,va_x=train_x2.iloc[tr_idx],train_x2.iloc[va_idx]
  tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
  model_lr=LogisticRegression(solver="lbfgs",max_iter=300)
  model_lr.fit(tr_x,tr_y)
  va_pred=model_lr.predict_proba(va_x)[:,1]

  #loglossはロジスティック損失を表しており、ロジスティック損失は、確率予測の正確さを測るための指標のひとつで、誤差が大きいほど損失が指数関数的に大きくなる特徴があります。
  score = log_loss(va_y,va_pred)
  accuracy = accuracy_score(va_y,va_pred>0.5)
  scores_logloss.append(score)
  scores_accuracy.append(accuracy)

print(f"logloss:{np.mean(scores_logloss):.4f}")
print(f"accuracy:{np.mean(scores_accuracy):.4f}")
#logloss:0.4807
#accuracy:0.7770"""


# In[ ]:


from xgboost import XGBClassifier

model_xgb=XGBClassifier(n_estimators=20,random_state=71)
model_xgb.fit(ok_train_x,train_y)
pred_xgb=model_xgb.predict_proba(ok_test_x)[:,1]

pred=pred_xgb
pred_label=np.where(pred>0.5,1,0)
sample[1] = pred_label
sample.to_csv("submit.csv", header=None)


# In[ ]:




