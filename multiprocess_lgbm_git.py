#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import roc_curve, auc
from scipy.stats import ks_2samp
import datetime
import time
import json
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import warnings
import sys
import os
import joblib
from sklearn.metrics import accuracy_score
from scipy import stats
import pickle
import random
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import math
from scipy.stats import chi2_contingency
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
import copy
import threading
import pathlib
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn import model_selection
#warnings.filterwarnings('ignore')
from multiprocessing import Process
from sklearn.model_selection import GridSearchCV


ROOT=str(pathlib.Path(os.path.abspath(__file__)).parent.parent)

df1=pd.read_csv(ROOT+'\\xxx.csv',encoding='gbk')
df1=df1.select_dtypes(exclude=object)
df1.dropna(inplace=True)
df2=pd.read_csv(ROOT+'\\yyy.csv',encoding='gbk')
df2=df2.select_dtypes(exclude=object)
df2.dropna(inplace=True)
y_train=df1['bad_flag']
x_train=df1.drop('bad_flag',axis=1)
y_test=df2['bad_flag']
x_test=df2.drop('bad_flag',axis=1)
##in train and test dataset ,there is only numeric variables kept
##############################

def write2txt(data,filename):
    f = open(filename,"w")
    f.write(str(data))
    f.close()


def make_one_param_grid(param, param_min , param_max, step):
    if type(step)==int:
        return {param: [i for i in np.arange(param_min, param_max, step)]}
    elif type(step)==float:
        digit=len(str(step).split(".")[1])
        return {param: [round(i,digit) for i in np.arange(param_min, param_max, step)]}

def make_all_param_grid(params, params_min,  params_max, steps ):
    '''
    :param params:list
    :param params_min: list
    :param params_max: list
    :param steps: list
    :return:dict
    '''
    if len(params)==len(params_min)==len(params_max)==len(steps):
        result_dict={}
        for i in range(len(params)):
            temp=make_one_param_grid(params[i], params_min[i],params_max[i],steps[i])
            result_dict.update(temp)
        return result_dict;
    else:
        print('something wrong with counts of param，please check');
##

##compare search results gotten from some processes
def V_candidate(class_name, train_x, train_y, candidatelist):
    #
    mean_score=[]
    for i in range(len(candidatelist)):#
        clf=class_name(**candidatelist[i])
        mean_score_temp=round(np.array(model_selection.cross_val_score(clf, train_x, train_y, scoring='roc_auc', cv=10)).mean(),4)
        mean_score.append(mean_score_temp)

    print(mean_score)
    return candidatelist[mean_score.index(np.array(mean_score).max())]
    ###return dict,final optimal solution


def base_model(model_select,train_x, train_y, param_grid):

    model=model_select();

    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, verbose=1,cv=10)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    '''
    for para, val in best_parameters.items():
        print ('para' , para,   'val' , val)
    '''
    return  dict_get_value(  param_grid,best_parameters)

def dict_get_value(dict1,dict2):

    dict2_value_temp=[]
    for i in  dict1.keys():
        globals()['dict2_value_'+i]=dict2.get(i)
        dict2_value_temp.append(globals()['dict2_value_'+i])
    return dict(zip(list(dict1.keys()),dict2_value_temp))



def FlexSearch(train_x, train_y, model_select, param_grid, filename, delay=0.5):
    bests = {}
    print('You need to find the best in this parameter dictionary list：',param_grid)
    while True:

        params=[]
        params_min=[]
        params_max=[]
        steps=[]
        mededge = base_model(model_select, train_x, train_y, param_grid)
        print('mededge optimal solution is',mededge)

        for i in mededge:
            #
            if len(param_grid[i])-1>0:
                common_diff=(np.array(param_grid[i]).max()-np.array(param_grid[i]).min())/(len(param_grid[i])-1)
                step=common_diff##
            else:
                step=0;

            print('!!!!!!!tolerance:',  i,  step,   common_diff)
            #

            if len(str(mededge[i]).split('.')) == 1:  #
                minedge = int(mededge[i] - step - step / 2)  ##
                maxedge = int(mededge[i] + step + step / 2)
                step=int(step * delay)#

                print(i, " Current optimal parameter value:", mededge[i])
                print(i, " Next iteration domain %d ~ %d" % (minedge, maxedge))
                print(i, " Next iteration step:", step)
            elif len(str(mededge[i]).split('.')) > 1:  ##
                minedge = round((mededge[i] - step - step / 2),4)  ##
                maxedge = round((mededge[i] + step + step / 2),4)
                step=round((step * delay),4)  ##

                print(i," Current optimal parameter value:", mededge[i] )
                print(i," Next iteration domain %f ~ %f" % (minedge,maxedge) )
                print(i," Next iteration step:", step )

            if step>=0.01 and minedge>0 and maxedge > 0 and minedge!=maxedge:#
                #
                params.append(i)
                params_min.append(minedge)
                params_max.append(maxedge)
                steps.append(step)
            else:
                bests.update({i:mededge[i]})
                print('Intermediate process results！！！！！！！！！！！！！',bests)
                #
        if len(params)>=1:
            print('params',params)
            print('params_min',params_min)
            print('params_max',params_max)
            print('steps',steps)
            param_grid=make_all_param_grid(params, params_min, params_max, steps);
            print(  'new round of search parameters are：' ,   param_grid)
            continue
        else:
            print("Optimal model parameters:", bests)
            write2txt(bests,filename);
            break;
###   FlexSearch  end

##############Assign the whole process to different processes
##Starting from a variable, divided into different processes
def multi_process( train_x, train_y, model_select, param_grid, process_counts=3):
    if __name__=='__main__':
        start_time=time.time()

        index_temp=[]
        param_counts_temp=[]

        for i in (param_grid):
            index_temp.append(i)
            param_counts_temp.append(len(param_grid[i]))#
        multi_process_param_name=pd.Series(param_counts_temp,index=index_temp).sort_values(ascending=False).index[0]

        gap=int(len(param_grid[multi_process_param_name])/process_counts)


        for i in range(process_counts):
            globals()[multi_process_param_name+'_'+str(i)]=\
                param_grid[multi_process_param_name][(i*gap):(i*gap+gap)]
            if i==(process_counts)-1:
                globals()[multi_process_param_name+'_'+str(i)]=\
                    param_grid[multi_process_param_name][(i * gap):]
        param_grid.pop(multi_process_param_name)

        for i in range(process_counts):
            globals()['param_grid'+'_'+str(i)]=\
                { multi_process_param_name : globals()[multi_process_param_name+'_'+str(i)]}

            globals()['param_grid' + '_' + str(i)].update(param_grid)


        subprocess = []  #
        for i in range(process_counts):
            p = Process(target=FlexSearch, args=(train_x, train_y, model_select, globals()['param_grid' + '_' + str(i)],  'FlexibleSearch_best_params_'+str(i)+'.txt',0.5 ))
            subprocess.append(p)
        print('生成的进程分别是：', subprocess)

        for k in subprocess:
            k.start()
        for l in subprocess:
            l.join()

        candidatelist = []
        for i in range(process_counts):
            with open("FlexibleSearch_best_params_%s.txt" % i) as f:
                candidatelist.append(f.readlines()[0].strip())

        candidatelist_temp_1=[]

        for i in range(len(candidatelist)):
            exec('candidatelist_temp_2=' + candidatelist[i])

            globals()['candidatelist_temp_2']=locals()['candidatelist_temp_2']

            candidatelist_temp_1.append(candidatelist_temp_2)

        print('candidatelist_temp_1',candidatelist_temp_1)

        candidatelist=candidatelist_temp_1

        print("candidatelist:", candidatelist)

        best_n_estimators = V_candidate(model_select, train_x, train_y, candidatelist)
        print("best_n_estimators最终值", best_n_estimators)

        ####calculate time elapsed
        end_time = time.time()
        print('time elapsed：', end_time - start_time, 'seconds')
    #####################################  END   ######################################

    ##
param_grid = make_all_param_grid(['num_leaves', 'max_depth', 'n_estimators', 'learning_rate'], [10, 2, 100, 0.1],
                                     [31, 6, 200, 0.5], [5, 1, 10, 0.1])
    ##
if __name__ == "__main__":
    multi_process(x_train, y_train, LGBMClassifier, param_grid, process_counts=3)
    #
    # Can only run as a file

































