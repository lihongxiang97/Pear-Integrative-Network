#!/storage/lihongxiang/soft/keras/bin/python3
#Writed by LHX on 2023/11/22
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import (ensemble, linear_model,
                     neighbors,  svm)
import tensorflow as tf
import keras
from keras.losses import binary_crossentropy
import joblib
from sklearn.metrics import (accuracy_score, auc, f1_score, make_scorer,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve,precision_recall_curve)
from keras.layers import *
#from imblearn.over_sampling import  ADASYN
import tqdm
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.impute import KNNImputer
#import  matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from plotnine import *
import warnings
import subprocess
import os
#warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser(description="Machine Learning Script for Gene Regulation Network Data")
parser.add_argument("--train", type=str, required=True, help="Path to the training data CSV file")
parser.add_argument("--test", type=str, required=True, help="Path to the test data CSV file")
parser.add_argument("--randomnum",type=int,default=20,help="random number of training, larger number means greater results but more time. default is 20 ")
parser.add_argument("--modeldir", type=str, default='./model', help="Directory to save the output files")
parser.add_argument("--dir", type=str, default='./result', help="Directory to save the predict result files")
parser.add_argument("--gpu", action='store_true', help="Use GPU for training if available")

args = parser.parse_args()

if args.gpu:
    # Set the environment variable to only use GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if not os.path.exists(args.modeldir):
    # 如果不存在，创建文件夹
    os.makedirs(args.modeldir)

if not os.path.exists(args.dir):
    # 如果不存在，创建文件夹
    os.makedirs(args.dir)

class machinelearning():
    def __init__(self):
        self.lr = linear_model.LogisticRegressionCV()
        self.svm = svm.SVC()
        self.bagging = ensemble.BaggingClassifier()
        self.xgboost = XGBClassifier()
        self.lr_param = {
            'fit_intercept': [True, False],  # default: True
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag',
                       'saga'],  # default: lbfgs
            'random_state': [0],
            'max_iter': [1000000]
        }

        self.svm_param = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [1, 2, 3, 4, 5],  # default=1.0
            'gamma': [.1, .25, .5, .75, 1.0],  # edfault: auto
            'decision_function_shape': ['ovo', 'ovr'],  # default:ovr
            'probability': [True],
            'random_state': [0]
        }

        self.bagging_param = {
            'n_estimators': [10, 50, 100, 300],  # default=10
            'max_samples': [.1, .25, .5, .75, 1.0],  # default=1.0
            'random_state': [0]
        }

        self.xgb_param = {
            'learning_rate': [.01, .03, .05, .1, .25],  # default: .3
            'max_depth': [1, 2, 4, 6, 8, 10],  # default 2
            'n_estimators': [10, 50, 100, 300],
            'seed': [0],
            'tree_method':['gpu_hist'] if args.gpu else ['auto']
        }

    def train(self, traindata, testdata, randomnum):
        '''
        :param data:Array data
        :param randomnum: Number of repetitions
        :return:
        '''
        auclr = np.zeros(randomnum)
        aucsvm = np.zeros(randomnum)
        aucbagging = np.zeros(randomnum)
        aucxgb = np.zeros(randomnum)
        aucnn = np.zeros(randomnum)

        f1lr = np.zeros(randomnum)
        f1svm = np.zeros(randomnum)
        f1bagging = np.zeros(randomnum)
        f1xgb = np.zeros(randomnum)
        f1nn = np.zeros(randomnum)

        TPlr = np.zeros(randomnum)
        TPsvm = np.zeros(randomnum)
        TPbagging = np.zeros(randomnum)
        TPxgb = np.zeros(randomnum)
        TPnn = np.zeros(randomnum)

        FPlr = np.zeros(randomnum)
        FPsvm = np.zeros(randomnum)
        FPbagging = np.zeros(randomnum)
        FPxgb = np.zeros(randomnum)
        FPnn = np.zeros(randomnum)

        TNlr = np.zeros(randomnum)
        TNsvm = np.zeros(randomnum)
        TNbagging = np.zeros(randomnum)
        TNxgb = np.zeros(randomnum)
        TNnn = np.zeros(randomnum)
        FNlr = np.zeros(randomnum)
        FNsvm = np.zeros(randomnum)
        FNbagging = np.zeros(randomnum)
        FNxgb = np.zeros(randomnum)
        FNnn = np.zeros(randomnum)

        Truesample = np.zeros(randomnum)
        Falsesample = np.zeros(randomnum)

        all_nntesty = np.zeros((int(testdata.shape[0]),))
        all_baggingtesty = np.zeros((int(testdata.shape[0]),))
        all_lrtesty= np.zeros((int(testdata.shape[0]),))
        all_svmtesty= np.zeros((int(testdata.shape[0]),))
        all_xgbtesty= np.zeros((int(testdata.shape[0]),))

        for i in tqdm.tqdm(range(randomnum)):
            x = traindata[:, 2:]
            y = traindata[:, 0]
            _y = y
            trainsd_x = x[:, :int(len(_y))]
            trainotherx = x[:, int(len(_y)):]

            testsd_x = testdata[:, 1:int(len(_y) + 1)]
            testotherx = testdata[:, int(len(_y) + 1):]

            index = np.arange(len(_y))
            np.random.shuffle(index)

            randomcol = index[:int(0.8*len(y))]
            valindex = index[int(0.8*len(y)):]
            trainsd0_x = trainsd_x[:, randomcol]
            testsd0_x = testsd_x[:, randomcol]

            newtrainx = np.concatenate((trainsd0_x, trainotherx), axis=1)
            newtestx = np.concatenate((testsd0_x, testotherx), axis=1)

            nanindert = KNNImputer()
            newtrainx = nanindert.fit_transform(newtrainx)
            newtestx = nanindert.fit_transform(newtestx)

            train_x = newtrainx[randomcol]
            train_y = _y[randomcol]
            train_y = train_y.astype(np.int32)

            val_x = newtrainx[valindex]
            val_y = _y[valindex]
            val_y = val_y.astype(np.int32)

            scalr = preprocessing.MinMaxScaler()
            train_x = scalr.fit_transform(train_x)
            val_x = scalr.transform(val_x)
            newtestx = scalr.transform(newtestx)

            sd_trainx = train_x[:, 0:int(0.8 * len(_y) + 4)]
            trans_trainx = train_x[:, int(0.8 * len(_y) + 4):int(0.8 * len(_y) + 107)]
            rna_trainx = train_x[:, int(0.8 * len(_y) + 107):int(0.8 * len(_y) + 130)]
            prot_trainx = train_x[:, int(0.8 * len(_y) + 130):int(0.8 * len(_y) + 136)]

            sd_valx = val_x[:, 0:int(0.8 * len(_y) + 4)]
            trans_valx = val_x[:, int(0.8 * len(_y) + 4):int(0.8 * len(_y) + 107)]
            rna_valx = val_x[:, int(0.8 * len(_y) + 107):int(0.8 * len(_y) + 130)]
            prot_valx = val_x[:, int(0.8 * len(_y) + 130):int(0.8 * len(_y) + 136)]

            sd_testx = newtestx[:, 0:int(0.8 * len(_y) + 4)]
            trans_testx = newtestx[:, int(0.8 * len(_y) + 4):int(0.8 * len(_y) + 107)]
            rna_testx = newtestx[:, int(0.8 * len(_y) + 107):int(0.8 * len(_y) + 130)]
            prot_testx = newtestx[:, int(0.8 * len(_y) + 130):int(0.8 * len(_y) + 136)]

            train_x = np.concatenate((sd_trainx, trans_trainx, prot_trainx),axis=1)
            val_x = np.concatenate((sd_valx, trans_valx, prot_valx),axis=1)
            newtestx = np.concatenate((sd_testx, trans_testx, prot_testx),axis=1)

            Truesample[i] = sum(val_y == 1)
            Falsesample[i] = sum(val_y == 0)

            inputs4 = Input(shape=(int(0.8 * len(_y) + 4),))
            inputs3 = Input(shape=(6,))
            #inputs2 = Input(shape=(23,))
            inputs1 = Input(shape=(103,))

            x1 = Dense(128, activation='relu')(inputs1)
            x1 = Dropout(0.1)(x1)
            #x2 = Dense(64, activation='relu')(inputs2)
            #x2 = Dropout(0.1)(x2)
            x3 = Dense(64, activation='relu')(inputs3)
            x3 = Dropout(0.1)(x3)
            x4 = Dense(128, activation='relu')(inputs4)
            x4 = Dropout(0.1)(x4)

            x = Concatenate()([x1, x3, x4])
            x = Dense(50, activation='relu')(x)

            out = Dense(1, activation='sigmoid')(x)
            model = keras.Model([inputs1, inputs3, inputs4], out)
            model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])
            from keras.callbacks import ModelCheckpoint
            filepath = args.modeldir+'/nn.h5'
            checkpoint = ModelCheckpoint(filepath,
                                         monitor='val_loss',
                                         verbose=0,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='auto',
                                         period=1
                                         )
            callbacks_list = [checkpoint]
            model.fit([trans_trainx, prot_trainx, sd_trainx], train_y, batch_size=64, epochs=100, verbose=1,
                      validation_data=([trans_valx, prot_valx, sd_valx], val_y), callbacks=callbacks_list)
            nn = keras.models.load_model(args.modeldir+'/nn.h5')

            predict = np.array(nn.predict([trans_valx, prot_valx, sd_valx]))
            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0
            nnauc = roc_auc_score(val_y, nn.predict([trans_valx, prot_valx, sd_valx]))
            nnf1 = f1_score(val_y, predict)
            matrix = confusion_matrix(val_y, predict)
            TPnn[i] = matrix[1, 1]
            FPnn[i] = matrix[0, 1]
            TNnn[i] = matrix[0, 0]
            FNnn[i] = matrix[1, 0]
            aucnn[i] = nnauc
            f1nn[i] = nnf1

            nntesty = nn.predict([trans_testx, prot_testx, sd_testx])
            nntesty = nntesty.reshape(len(nntesty), )
            all_nntesty += nntesty

            if nnauc > aucnn[i - 1]  and nnf1 > f1nn[i - 1] and i >= 1:
                subprocess.call('mv '+args.modeldir+'/nn.h5 '+args.modeldir+'/best_nn_model.h5',shell=True)
            elif i == 0:
                subprocess.call('mv '+args.modeldir+'/nn.h5 '+args.modeldir+'/best_nn_model.h5',shell=True)

            lr_gs = GridSearchCV(estimator=self.lr,
                                 param_grid=self.lr_param,
                                 cv=5,
                                 scoring='roc_auc',
                                 refit=True, n_jobs=-1)

            svm_gs = GridSearchCV(estimator=self.svm,
                                  param_grid=self.svm_param,
                                  cv=5,
                                  scoring='roc_auc',
                                  refit=True, n_jobs=-1)

            bagging_gs = GridSearchCV(estimator=self.bagging,
                                      param_grid=self.bagging_param,
                                      cv=5,
                                      scoring='roc_auc',
                                      refit=True, n_jobs=-1)
            xgb_gs = GridSearchCV(estimator=self.xgboost,
                                  param_grid=self.xgb_param,
                                  cv=5,
                                  scoring='roc_auc',
                                  refit=True, n_jobs=-1)


            lr_gs.fit(train_x, train_y)
            svm_gs.fit(train_x, train_y)
            bagging_gs.fit(train_x, train_y)
            xgb_gs.fit(train_x, train_y)

            Blr = lr_gs.best_estimator_
            Bsvm = svm_gs.best_estimator_
            Bbagging = bagging_gs.best_estimator_
            Bxgb = xgb_gs.best_estimator_

            baggingtesty = Bbagging.predict_proba(newtestx)[:, 1]
            lrtesty = Blr.predict_proba(newtestx)[:, 1]
            svmtesty = Bsvm.predict_proba(newtestx)[:, 1]
            xgbtesty = Bxgb.predict_proba(newtestx)[:, 1]

            all_baggingtesty += baggingtesty
            all_lrtesty += lrtesty
            all_svmtesty += svmtesty
            all_xgbtesty += xgbtesty

            test_y_prob = tf.one_hot(val_y, depth=2)

            lrauc = roc_auc_score(test_y_prob, Blr.predict_proba(val_x))
            svmauc = roc_auc_score(test_y_prob, Bsvm.predict_proba(val_x))
            baggingauc = roc_auc_score(test_y_prob, Bbagging.predict_proba(val_x))
            xgbauc = roc_auc_score(test_y_prob, Bxgb.predict_proba(val_x))

            lrf1 = f1_score(val_y, Blr.predict(val_x))
            svmf1 = f1_score(val_y, Bsvm.predict(val_x))
            baggingf1 = f1_score(val_y, Bbagging.predict(val_x))
            xgbf1 = f1_score(val_y, Bxgb.predict(val_x))

            auclr[i] = lrauc
            f1lr[i] = lrf1
            matrix = confusion_matrix(val_y, Blr.predict(val_x))
            TPlr[i] = matrix[1, 1]
            FPlr[i] = matrix[0, 1]
            TNlr[i] = matrix[0, 0]
            FNlr[i] = matrix[1, 0]

            if lrauc > auclr[i - 1] and lrf1 > f1lr[i - 1] and i >= 1:
                joblib.dump(Blr, args.modeldir+'/best_lr_model.pkl')
            elif i == 0:
                joblib.dump(Blr, args.modeldir+'/best_lr_model.pkl')

            aucsvm[i] = svmauc
            f1svm[i] = svmf1
            matrix = confusion_matrix(val_y, Bsvm.predict(val_x))
            TPsvm[i] = matrix[1, 1]
            FPsvm[i] = matrix[0, 1]
            TNsvm[i] = matrix[0, 0]
            FNsvm[i] = matrix[1, 0]

            if svmauc > aucsvm[i - 1] and svmf1 > f1svm[i - 1] and i >= 1:
                joblib.dump(Bsvm, args.modeldir+'/best_svm_model.pkl')
            elif i == 0:
                joblib.dump(Bsvm, args.modeldir+'/best_svm_model.pkl')

            aucbagging[i] = baggingauc
            f1bagging[i] = baggingf1
            matrix = confusion_matrix(val_y, Bbagging.predict(val_x))
            TPbagging[i] = matrix[1, 1]
            FPbagging[i] = matrix[0, 1]
            TNbagging[i] = matrix[0, 0]
            FNbagging[i] = matrix[1, 0]
            if baggingauc > aucbagging[i - 1] and baggingf1 > f1bagging[i - 1] and i >= 1:
                joblib.dump(Bbagging, args.modeldir+'/best_bagging_model.pkl')
            elif i == 0:
                joblib.dump(Bbagging, args.modeldir+'/best_bagging_model.pkl')
            aucxgb[i] = xgbauc
            f1xgb[i] = xgbf1
            matrix = confusion_matrix(val_y, Bxgb.predict(val_x))
            TPxgb[i] = matrix[1, 1]
            FPxgb[i] = matrix[0, 1]
            TNxgb[i] = matrix[0, 0]
            FNxgb[i] = matrix[1, 0]
            if xgbauc > aucxgb[i - 1] and xgbf1 > f1xgb[i - 1] and i >= 1:
                joblib.dump(Bxgb, args.modeldir+'/best_xgb_model.pkl')
            elif i == 0:
                joblib.dump(Bxgb, args.modeldir+'/best_xgb_model.pkl')
        nntesty0 = all_nntesty/randomnum
        baggingtesty0 = all_baggingtesty/randomnum
        lrtesty0 = all_lrtesty/randomnum
        svmtesty0 = all_svmtesty/randomnum
        xgbtesty0 = all_xgbtesty/randomnum
        return nntesty0, baggingtesty0, lrtesty0, svmtesty0, xgbtesty0, auclr, aucsvm, aucbagging, aucxgb, aucnn, f1lr, f1svm, f1bagging, f1xgb, f1nn, TPlr, TPsvm, TPbagging, TPxgb, TPnn, FPlr, FPsvm, FPbagging, FPxgb, FPnn, TNlr,  TNsvm,  TNbagging, TNxgb, TNnn, FNlr,  FNsvm,  FNbagging, FNxgb, FNnn, Truesample, Falsesample

traindata=np.array(pd.read_csv(args.train))
testdata=np.array((pd.read_csv(args.test)))
model=machinelearning()
nntesty0, baggingtesty0, lrtesty0, svmtesty0, xgbtesty0, auclr, aucsvm, aucbagging, aucxgb, aucnn, f1lr, f1svm, f1bagging, f1xgb, f1nn, TPlr, TPsvm, TPbagging, TPxgb, TPnn, FPlr, FPsvm, FPbagging, FPxgb, FPnn, TNlr, TNsvm, TNbagging, TNxgb, TNnn, FNlr, FNsvm,  FNbagging, FNxgb, FNnn, Truesample, Falsesample=model.train(traindata,testdata,args.randomnum)

aucdict = {'LR_AUC': auclr,
           'SVM_AUC': aucsvm,
           'Bagging_AUC': aucbagging,
           'XgBoost_AUC': aucxgb,
           'NeuralNet_AUC': aucnn,
           'LR_F1': f1lr,
           'SVM_F1': f1svm,
           'Bagging_F1': f1bagging,
           'XgBoost_F1': f1xgb,
           'NeuralNet_F1': f1nn,
           'LR_TP': TPlr,
           'SVM_TP': TPsvm,
           'Bagging_TP': TPbagging,
           'XgBoost_TP': TPxgb,
           'NeuralNet_TP': TPnn,
           'LR_FP': FPlr,
           'SVM_FP': FPsvm,
           'Bagging_FP': FPbagging,
           'XgBoost_FP': FPxgb,
           'NeuralNet_FP': FPnn,
           'LR_FN': FNlr,
           'SVM_FN': FNsvm,
           'Bagging_FN': FNbagging,
           'XgBoost_FN': FNxgb,
           'NeuralNet_FN': FNnn,
           'LR_TN': TNlr,
           'SVM_TN': TNsvm,
           'Bagging_TN': TNbagging,
           'XgBoost_TN': TNxgb,
           'NeuralNet_TN': TNnn,
           'Truesamples': Truesample,
           'Falsesamples': Falsesample
           }

allfeaturedict = pd.DataFrame(aucdict).to_excel(args.dir+'/train_auc_f1.xlsx',index=False)
pd.DataFrame(aucdict).to_csv(args.dir+'/train_auc_f1.csv',index=False)

dict={'Gene':testdata[:,0],
      'nnpredresult':nntesty0,
      'baggingpredresult':baggingtesty0,
      'lrpredresult':lrtesty0,
      'svmpredresult':svmtesty0,
      'xgbpredresult':xgbtesty0
      }
result=pd.DataFrame(dict).to_excel(args.dir+'/predresult.xlsx',index=False)
pd.DataFrame(dict).to_csv(args.dir+'/predresult.csv',index=False)

###export importance of features
data=pd.DataFrame(pd.read_csv(args.train)) ##input train data
featurename=list(data.columns)[2:]
x=np.array(data)[:,2:].astype(np.float64)
y=np.array(data)[:,0].astype(np.int32)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)
lr_param = {
            'fit_intercept': [True, False],  # default: True
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag',
                       'saga'],  # default: lbfgs
            'random_state': [0],
            'max_iter': [10000]}
lr=linear_model.LogisticRegressionCV()
lr_gs = GridSearchCV(estimator=lr,
                                 param_grid=lr_param,
                                 cv=5,
                                 scoring='roc_auc',
                                 refit=True)
nanindert = KNNImputer()
train_x = nanindert.fit_transform(train_x)
test_x = nanindert.fit_transform(test_x)

lr_gs.fit(train_x,train_y)
Blr = lr_gs.best_estimator_
weight=np.abs(Blr.coef_).reshape(len(featurename),)
datafram=pd.DataFrame(weight/sum(weight))
datafram['featurename']=featurename
datafram.columns=['featureimportance','featurename']
datafram.to_excel(args.dir+'/featureimportance.xlsx',index=False)
datafram.to_csv(args.dir+'/featureimportance.csv',index=False)