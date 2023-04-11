from sklearn import svm
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
#from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import auc
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
import csv
import random
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return


#TreeAndDv = pd.read_csv(u'./label.txt', header=None, low_memory=False, sep='\t')
#y = []
#for i in TreeAndDv[0]:
   # y.append(int(i))
#TreeAndDv_3 = pd.read_csv('E:\\1119\\3.lncRNA-disease_association_vector\\Intract\\35ep-128dis.csv', header=None, low_memory=False, sep=',')
#j = 0
#train_vect = np.zeros((len(TreeAndDv_3[0]), 128))
#while j < len(TreeAndDv_3[0]):
   # t_0 = 0
   # while t_0 < 128:
     #   train_vect[j][t_0] = float(TreeAndDv_3[t_0][j])
   #     t_0 += 1
   # j += 1
#print(len(train_vect[0]))
SampleFeature = []
feature=[]
ReadMyCsv(feature, "Q_AE.csv")
for i in range(len(feature)):
    c = []
    for j in range(len(feature[0])):
        c.append(float(feature[i][j]))
    SampleFeature .append(c)
# print(len(SampleFeature))
# print(len(SampleFeature[0]))

# SampleLabel
SampleLabel = []
counter = 0
while counter < len(SampleFeature) / 2:
    SampleLabel.append(1)
    counter = counter + 1
counter1 = 0
while counter1 < len(SampleFeature) / 2:
    SampleLabel.append(0)
    counter1 = counter1 + 1

# 打乱数据集顺序
counter = 0
R = []
while counter < len(SampleFeature):
    R.append(counter)
    counter = counter + 1
random.shuffle(R)

RSampleFeature = []
RSampleLabel = []
counter = 0
while counter < len(SampleFeature):
    RSampleFeature.append(SampleFeature[R[counter]])
    RSampleLabel.append(SampleLabel[R[counter]])
    counter = counter + 1

print('len(RSampleFeature)', len(RSampleFeature))
print('len(RSampleLabel)', len(RSampleLabel))
SampleFeature = []
SampleLabel = []
SampleFeature = RSampleFeature
SampleLabel = RSampleLabel
X = np.array(SampleFeature)
y = np.array(SampleLabel)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=112)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
tprs = []
aupr_svm = []
aucs_svm = []
acc_svm = []
F1score_svm = []
precision_svm = []
recall_svm = []
mean_fpr = np.linspace(0, 1, 100)

tprs_RF = []
aupr_RF = []
aucs_RF = []
acc_RF = []
F1score_RF = []
precision_RF = []
recall_RF = []
mean_fpr_RF = np.linspace(0, 1, 100)

tprs_lgb = []
aupr_lgb = []
aucs_lgb = []
acc_lgb = []
F1score_lgb = []
precision_lgb = []
recall_lgb = []
mean_fpr_xgb = np.linspace(0, 1, 100)

tprs_ada = []
aupr_ada = []
aucs_ada = []
acc_ada = []
F1score_ada = []
precision_ada = []
recall_ada = []
mean_fpr_ada = np.linspace(0, 1, 100)

tprs_gbdt = []
aupr_gbdt = []
aucs_gbdt = []
acc_gbdt = []
F1score_gbdt = []
precision_gbdt = []
recall_gbdt = []
mean_fpr_gbdt = np.linspace(0, 1, 100)

auc_all = []
aupr_all = []
acc_all = []
F1score_all = []
precision_all = []
recall_all = []
data = X
label = y
i = 0
for train_index, test_index in kf.split(data, label):
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]
    #     #  svm
    model = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.01, probability=True)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_proba)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = metrics.roc_auc_score(Y_test, y_pred_proba)
    precision_svm_aupr, recall_svm_aupr, _ = precision_recall_curve(Y_test, y_pred_proba)
    aupr_svm.append(auc(recall_svm_aupr, precision_svm_aupr))
    aucs_svm.append(roc_auc)
    acc_svm.append(metrics.accuracy_score(Y_test, y_pred))
    recall_svm.append(metrics.recall_score(Y_test, y_pred))
    F1score_svm.append(metrics.f1_score(Y_test, y_pred))
    precision_svm.append(metrics.precision_score(Y_test, y_pred))

    #  RF
    model_RF = RandomForestClassifier(n_estimators=100, max_depth=11, min_samples_split=80, min_samples_leaf=10, max_features=7, oob_score=True, random_state=10)
    model_RF.fit(X_train, Y_train)
    y_pred_RF = model_RF.predict(X_test)
    y_pred_proba_RF = model_RF.predict_proba(X_test)[:, 1]
    fpr_RF, tpr_RF, thresholds_RF = metrics.roc_curve(Y_test, y_pred_proba_RF)
    tprs_RF.append(interp(mean_fpr_RF, fpr_RF, tpr_RF))
    tprs_RF[-1][0] = 0.0
    precision_RF_aupr, recall_RF_aupr, _ = precision_recall_curve(Y_test, y_pred_proba_RF)
    aupr_RF.append(auc(recall_RF_aupr, precision_RF_aupr))
    roc_auc_RF = metrics.roc_auc_score(Y_test, y_pred_proba_RF)
    aucs_RF.append(roc_auc_RF)
    acc_RF.append(metrics.accuracy_score(Y_test, y_pred_RF))
    recall_RF.append(metrics.recall_score(Y_test, y_pred_RF))
    F1score_RF.append(metrics.f1_score(Y_test, y_pred_RF))
    precision_RF.append(metrics.precision_score(Y_test, y_pred_RF))

    #  gbdt
    model_gbdt =GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=1, min_samples_split=2, min_samples_leaf=1,  subsample=0.8, random_state=10)
    model_gbdt.fit(X_train, Y_train)
    y_pred_gbdt = model_gbdt.predict(X_test)
    y_pred_proba_gbdt = model_gbdt.predict_proba(X_test)[:, 1]
    fpr_gbdt, tpr_gbdt, thresholds_gbdt = metrics.roc_curve(Y_test, y_pred_proba_gbdt)
    tprs_gbdt.append(interp(mean_fpr_gbdt, fpr_gbdt, tpr_gbdt))
    tprs_gbdt[-1][0] = 0.0
    precision_gbdt_aupr, recall_gbdt_aupr, _ = precision_recall_curve(Y_test, y_pred_proba_gbdt)
    aupr_gbdt.append(auc(recall_gbdt_aupr, precision_gbdt_aupr))
    roc_auc_gbdt = metrics.roc_auc_score(Y_test, y_pred_proba_gbdt)
    aucs_gbdt.append(roc_auc_gbdt)
    acc_gbdt.append(metrics.accuracy_score(Y_test, y_pred_gbdt))
    recall_gbdt.append(metrics.recall_score(Y_test, y_pred_gbdt))
    F1score_gbdt.append(metrics.f1_score(Y_test, y_pred_gbdt))
    precision_gbdt.append(metrics.precision_score(Y_test, y_pred_gbdt))

    #  adaboost
    model_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME", n_estimators=800, learning_rate=0.2)
    model_ada.fit(X_train, Y_train)
    y_pred_ada = model_ada.predict(X_test)
    y_pred_proba_ada = model_ada.predict_proba(X_test)[:, 1]
    fpr_ada, tpr_ada, thresholds_ada = metrics.roc_curve(Y_test, y_pred_proba_ada)
    tprs_ada.append(interp(mean_fpr_ada, fpr_ada, tpr_ada))
    tprs_ada[-1][0] = 0.0
    precision_ada_aupr, recall_ada_aupr, _ = precision_recall_curve(Y_test, y_pred_proba_ada)
    aupr_ada.append(auc(recall_ada_aupr, precision_ada_aupr))
    roc_auc_ada = metrics.roc_auc_score(Y_test, y_pred_proba_ada)
    aucs_ada.append(roc_auc_ada)
    acc_ada.append(metrics.accuracy_score(Y_test, y_pred_ada))
    recall_ada.append(metrics.recall_score(Y_test, y_pred_ada))
    F1score_ada.append(metrics.f1_score(Y_test, y_pred_ada))
    precision_ada.append(metrics.precision_score(Y_test, y_pred_ada))

    # xgb
    model_lgb = lgb.LGBMClassifier(objective='binary', num_leaves=25,
                                learning_rate=0.1, n_estimators=383, max_depth=7,
                                bagging_fraction=0.7, feature_fraction=0.9, reg_lambda=0.2)
    model_lgb.fit(X_train, Y_train)
    y_pred_xgb = model_lgb.predict(X_test)
    y_pred_proba_xgb = model_lgb.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, thresholds_xgb = metrics.roc_curve(Y_test, y_pred_proba_xgb)
    tprs_lgb.append(interp(mean_fpr_xgb, fpr_xgb, tpr_xgb))
    tprs_lgb[-1][0] = 0.0
    precision_xgb_aupr, recall_xgb_aupr, _ = precision_recall_curve(Y_test, y_pred_proba_xgb)
    aupr_lgb.append(auc(recall_xgb_aupr, precision_xgb_aupr))
    roc_auc_xgb = metrics.roc_auc_score(Y_test, y_pred_proba_xgb)
    aucs_lgb.append(roc_auc_xgb)
    acc_lgb.append(metrics.accuracy_score(Y_test, y_pred_xgb))
    recall_lgb.append(metrics.recall_score(Y_test, y_pred_xgb))
    F1score_lgb.append(metrics.f1_score(Y_test, y_pred_xgb))
    precision_lgb.append(metrics.precision_score(Y_test, y_pred_xgb))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(tprs, axis=0)
print(mean_auc)

mean_tpr_RF = np.mean(tprs_RF, axis=0)
mean_tpr_RF[-1] = 1.0
mean_auc_RF = auc(mean_fpr_RF, mean_tpr_RF)
std_auc_RF = np.std(tprs_RF, axis=0)
print(mean_auc_RF)

mean_tpr_gbdt = np.mean(tprs_gbdt, axis=0)
mean_tpr_gbdt[-1] = 1.0
mean_auc_gbdt = auc(mean_fpr_gbdt, mean_tpr_gbdt)
std_auc_gbdt = np.std(tprs_gbdt, axis=0)
print(mean_auc_gbdt)

mean_tpr_ada = np.mean(tprs_ada, axis=0)
mean_tpr_ada[-1] = 1.0
mean_auc_ada = auc(mean_fpr_ada, mean_tpr_ada)
std_auc_ada = np.std(tprs_ada, axis=0)
print(mean_auc_ada)

mean_tpr_lgb = np.mean(tprs_lgb, axis=0)
mean_tpr_lgb[-1] = 1.0
mean_auc_xgb = auc(mean_fpr_xgb, mean_tpr_lgb)
std_auc_xgb = np.std(tprs_lgb, axis=0)
print(mean_auc_xgb)

auc_all.append(mean_auc)
auc_all.append(mean_auc_ada)
auc_all.append(mean_auc_gbdt)
auc_all.append(mean_auc_RF)
auc_all.append(mean_auc_xgb)

acc_all.append(np.mean(acc_svm))
acc_all.append(np.mean(acc_ada))
acc_all.append(np.mean(acc_gbdt))
acc_all.append(np.mean(acc_RF))
acc_all.append(np.mean(acc_lgb))

recall_all.append(np.mean(recall_svm))
recall_all.append(np.mean(recall_ada))
recall_all.append(np.mean(recall_gbdt))
recall_all.append(np.mean(recall_RF))
recall_all.append(np.mean(recall_lgb))

F1score_all.append(np.mean(F1score_svm))
F1score_all.append(np.mean(F1score_ada))
F1score_all.append(np.mean(F1score_gbdt))
F1score_all.append(np.mean(F1score_RF))
F1score_all.append(np.mean(F1score_lgb))

precision_all.append(np.mean(precision_svm))
precision_all.append(np.mean(precision_ada))
precision_all.append(np.mean(precision_gbdt))
precision_all.append(np.mean(precision_RF))
precision_all.append(np.mean(precision_lgb))

aupr_all.append(np.mean(aupr_svm))
aupr_all.append(np.mean(aupr_ada))
aupr_all.append(np.mean(aupr_gbdt))
aupr_all.append(np.mean(aupr_RF))
aupr_all.append(np.mean(aupr_lgb))

plt.plot(mean_fpr, mean_tpr, color='r', label=r'SVM (area=%0.4f)' % mean_auc, lw=2, alpha=.8)
plt.plot(mean_fpr_RF, mean_tpr_RF, color='g', label=r'RF (area=%0.4f)' % mean_auc_RF, lw=2, alpha=.8)
plt.plot(mean_fpr_gbdt, mean_tpr_gbdt, color='b', label=r'GBDT (area=%0.4f)' % mean_auc_gbdt, lw=2, alpha=.8)
plt.plot(mean_fpr_ada, mean_tpr_ada, color='m', label=r'Adaboost (area=%0.4f)' % mean_auc_ada, lw=2, alpha=.8)
plt.plot(mean_fpr_xgb, mean_tpr_lgb, color='y', label=r'LGBMClassifier (area=%0.4f)' % mean_auc_xgb, lw=2, alpha=.8)


std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
tprs_lower = np.maximum(mean_tpr-std_tpr, 0)

std_tpr_RF = np.std(tprs_RF, axis=0)
tprs_upper_RF = np.minimum(mean_tpr_RF+std_tpr_RF, 1)
tprs_lower_RF = np.maximum(mean_tpr_RF-std_tpr_RF, 0)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
#plt.savefig('./five_classifier_ROC1.tif')
plt.savefig('./five_classifier_ROC1.png' , dpi=400)
plt.show()

plt.figure(2)
plt.plot(mean_fpr_xgb, mean_tpr_lgb, color='y', label=r'LGBMClassifier (area=%0.4f)' % mean_auc_xgb, lw=2, alpha=.8)
plt.xlabel('FPR', fontsize=13)
plt.ylabel('TPR', fontsize=13)
# plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC 5-fold CV (AUC = %0.4f).png' % (mean_auc_xgb), dpi=400)
plt.show()




result_all = pd.DataFrame()
result_all['auc'] = auc_all
result_all['acc'] = acc_all
result_all['F1score'] = F1score_all
result_all['recall'] = recall_all
result_all['precision'] = precision_all
result_all['aupr'] = aupr_all
result_all.to_csv('five_classifier_score.csv', sep=',', index=False, header=True)
