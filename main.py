import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics

pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 12)

data = pd.read_csv('G:\IIT\Subject\DM & ML\CW2\pima-indians-diabetes.csv')

print(data.shape)
print(data.columns)
print(data.head())

# Identification and treatment of any Missing Values
print(data.isnull().values.any())
# No Missing Values

# Treatment of Outliers using Box plot
fig = plt.figure(figsize=(50, 20))
data.plot.box(title='Boxplot of Happiness Features', rot=90)
plt.show()

sns.boxplot(data["gnancies"])
plt.show()

sns.boxplot(data["glucose"])
plt.show()

sns.boxplot(data["diastolic"])
plt.show()

sns.boxplot(data["triceps"])
plt.show()

sns.boxplot(data["insulin"])
plt.show()

sns.boxplot(data["bmi"])
plt.show()

sns.boxplot(data["dpf"])
plt.show()

sns.boxplot(data["age"])
plt.show()

print(np.where(data['gnancies'] > 12.5))
print(np.where(data['glucose'] < 50))
print(np.where(data['diastolic'] > 100))
print(np.where(data['diastolic'] < 40))
print(np.where(data['insulin'] > 250))
print(np.where(data['bmi'] < 20))
print(np.where(data['bmi'] > 40))
print(np.where(data['dpf'] > 1))
print(np.where(data['triceps'] > 60))
print(np.where(data['age'] > 60))
print(np.where(data['glucose'] > 160))

data.drop([28, 72, 86, 88, 159, 274, 298, 323, 357, 455, 518, 635, 691, 744], inplace=True)
data.drop([62, 75, 182, 342, 349, 502], inplace=True)
data.drop([2, 8, 13, 22, 56, 154, 185, 186, 206, 209, 228, 236, 245,
           258, 260, 317, 319, 359, 360, 399, 408, 425, 427, 440, 489, 498,
           545, 546, 549, 561, 579, 595, 604, 606, 622, 661, 675, 715, 748,
           753, 759], inplace=True)

data.drop([43, 84, 106, 177, 207, 362, 369, 658, 662, 672], inplace=True)

data.drop([7, 15, 18, 49, 60, 78, 81, 125, 172, 193, 222, 261, 266,
           269, 300, 332, 336, 347, 426, 430, 435, 453, 468, 484, 494,
           522, 533, 535, 589, 597, 599, 601, 619, 643, 697, 703, 706], inplace=True)

data.drop([53, 54, 73, 111, 139, 144, 153, 162, 199,
           215, 220, 231, 247, 248, 254, 279, 286, 296, 335,
           364, 370, 375, 388, 392, 395, 409, 412, 415, 480, 486,
           487, 519, 574, 584, 608, 612, 645, 655, 679, 695, 707,
           710, 713], inplace=True)
data.drop([9, 33, 50, 68, 90, 145, 239, 316, 371, 418,
           438, 526, 607, 639, 684], inplace=True)
data.drop([4, 16, 41, 45, 57, 58, 59, 67,
           92, 99, 120, 126, 155, 173, 178,
           201, 211, 213, 229, 230, 235, 237, 270, 275,
           287, 292, 293, 303, 328, 350, 354, 378, 379, 387,
           391, 405, 420, 422, 424, 428, 445, 469, 470,
           485, 531, 532, 558, 577, 580, 590, 596,
           623, 638, 673, 681, 682, 689, 699, 712, 732, 740,
           746, 747, 761], inplace=True)
data.drop([12, 39, 100, 131, 147, 152, 187, 218,
           243, 259, 267, 308, 314, 330, 383,
           416, 434, 458, 464, 493, 534, 588,
           593, 618, 621, 657, 659, 750, 755], inplace=True)

data.drop([115, 123, 129, 148, 221, 223, 263, 294, 361, 363, 456,
           459, 479, 495, 509, 537, 552, 582, 666, 674,
           763], inplace=True)

data.drop([11, 14, 40, 110,
           130, 132, 175, 212,
           227, 238, 283,
           306, 327, 339, 355,
           404,
           506, 515, 548, 598,
           611, 646, 647, 660, 670, 696, 702,
           708, 716, 728, 749], inplace=True)


print(np.where(data['age'] > 55))
print(np.where(data['gnancies'] > 10))

sns.boxplot(data["gnancies"])
plt.show()

sns.boxplot(data["glucose"])
plt.show()

sns.boxplot(data["diastolic"])
plt.show()

sns.boxplot(data["triceps"])
plt.show()

sns.boxplot(data["insulin"])
plt.show()

sns.boxplot(data["bmi"])
plt.show()

sns.boxplot(data["dpf"])
plt.show()

sns.boxplot(data["age"])
plt.show()

# split data into inputs and targets
inputs = data.drop(columns=['diabetes'])
targets = data['diabetes']

diabetes_true_count = len(data.loc[data['diabetes'] == 1])
diabetes_false_count = len(data.loc[data['diabetes'] == 0])
print(diabetes_true_count)
print(diabetes_false_count)

# split data into train and test sets
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3,
                                                                          stratify=targets)
print(targets_test)

# Normalizing Dataset
scaler = preprocessing.StandardScaler()
inputs_train = scaler.fit_transform(inputs_train)
inputs_test = scaler.fit_transform(inputs_test)

# create new a knn model
knn = KNeighborsClassifier()
knn.fit(inputs_train, targets_train)

# create a new random forest classifier
rf = RandomForestClassifier()
rf.fit(inputs_train, targets_train)

# create a new logistic regression model
lr = LogisticRegression()
lr.fit(inputs_train, targets_train)

# create a  Support Vector Machines model
svm = SVC()
svm.fit(inputs_train, targets_train)

#
# test the models with the test data and print their accuracy scores
print('knn: {}'.format(knn.score(inputs_test, targets_test)))
print('rf: {}'.format(rf.score(inputs_test, targets_test)))
print('lr: {}'.format(lr.score(inputs_test, targets_test)))
print('svm: {}'.format(svm.score(inputs_test, targets_test)))

# Voting Classifier
estimators = [('knn', knn), ('rf', rf), ('lr', lr), ('svm', svm)]
ensemble = VotingClassifier(estimators)
ensemble.fit(inputs_train, targets_train)
ensemble_score = ensemble.score(inputs_test, targets_test)
print("Voting Classifier New")
print(ensemble_score)

predictions = ensemble.predict(inputs_test)
cm = confusion_matrix(targets_test, predictions, labels=ensemble.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble.classes_)
disp.plot()
plt.show()

# 2nd part
# GRID search

# svm
grid_search_svm = GridSearchCV(SVC(gamma='auto'),
                               {'C': [1, 10, 20, 30, 40, 50], 'kernel': ['rbf', 'linear', 'sigmoid', 'poly']}, cv=5,
                               return_train_score=False)
grid_search_svm.fit(inputs_train, targets_train)
grid_search_svm_df = pd.DataFrame(grid_search_svm.cv_results_)
# print(grid_search_svm_df[['param_C', 'param_kernel', 'mean_test_score']])
print('svm_new')
print(grid_search_svm.best_params_)
print(grid_search_svm.best_score_)

# LR

grid_search_lr = GridSearchCV(LogisticRegression(solver='liblinear', multi_class='auto'),
                              {'C': [1, 10, 20, 30, 40, 50]}, cv=5, return_train_score=False)
grid_search_lr.fit(inputs_train, targets_train)
grid_search_lr_df = pd.DataFrame(grid_search_svm.cv_results_)
print('lr_new')
print(grid_search_lr.best_params_)
print(grid_search_lr.best_score_)

# RF

grid_search_rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 20, 30, 40, 50, 100, 200]}, cv=5,
                              return_train_score=False)
grid_search_rf.fit(inputs_train, targets_train)
grid_search_rf_df = pd.DataFrame(grid_search_svm.cv_results_)
print('rf_new')
print(grid_search_rf.best_params_)
print(grid_search_rf.best_score_)

# knn

# KNeighborsClassifier().get_params().keys()
grid_search_knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': list(range(1, 31))}, cv=5,
                               return_train_score=False)
grid_search_knn.fit(inputs_train, targets_train)
grid_search_knn_df = pd.DataFrame(grid_search_svm.cv_results_)
print('knn_new')
print(grid_search_knn.best_params_)
print(grid_search_knn.best_score_)

# Voting Classifier After Grid Search
estimators = [('knn', grid_search_knn), ('rf', grid_search_rf), ('lr', grid_search_lr), ('svm', grid_search_svm)]
ensemble = VotingClassifier(estimators)
ensemble.fit(inputs_train, targets_train)
ensemble_score = ensemble.score(inputs_test, targets_test)
print("Voting Classifier")
print(ensemble_score)

#
print('Accuracy Score : ' + str(accuracy_score(targets_test, predictions)))
print('Precision Score : ' + str(precision_score(targets_test, predictions)))
print('Recall Score : ' + str(recall_score(targets_test, predictions)))
print('F1 Score : ' + str(f1_score(targets_test, predictions)))
print('AUROC : ' + str(roc_auc_score(targets_test, predictions)))

predictions = ensemble.predict(inputs_test)
fpr, tpr, thresholds = roc_curve(targets_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
disp.plot()
plt.show()
print('ROC Curve : ' + str(roc_curve(targets_test, predictions)))

predictions = ensemble.predict(inputs_test)
cm = confusion_matrix(targets_test, predictions, labels=ensemble.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble.classes_)
disp.plot()
plt.show()
print('Confusion Matrix : ' + str(confusion_matrix(targets_test, predictions)))
