#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:18:15 2020

@author: ehsan.mousavi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multi_drm import MULTI_DRM_Gradient
from schema import CONTINUOUS_COLS


class Multi_model():
    
    def __init__(self,dataset, **arg):
        self.model_paramters = arg
        self.train = (dataset['train']).copy(deep = True)
        self.test = (dataset['test']).copy(deep = True)
        
    def __initliziatin__(self):
        for  df in [self.train,self.test]:
            Multi_model.fill_na(df)
        self.weight_train = Multi_model.create_weight_columns(self.train)
        self.weight_test = Multi_model.create_weight_columns(self.test)

    def run(self):
        
        self.__initliziatin__()
        
        X_train = self.train[CONTINUOUS_COLS].to_numpy()
        y_train = self.train[['gb_0d','spend']].to_numpy()
        W_train = self.weight_train.to_numpy()
        
        X_test = self.test[CONTINUOUS_COLS].to_numpy()
        y_test= self.test[['gb_0d','ni_0d','spend']]
        W_test= self.weight_test
        control_column = self.weight_test.columns.get_loc('control')
        
        
        
        model =  MULTI_DRM_Gradient(**self.model_paramters)
        model.fit(X=X_train,
                  y=y_train,
                  cohort_weight = W_train,
                  control_column=control_column)
    
        score_test = model.predict(X_test)
        er = Multi_model.expected_metrics(score_test,
                                          y_test,
                                          W_test,
                                          self.model_paramters['lambda_saving'])
        score_test = pd.DataFrame(score_test,columns=self.weight_test.columns)
        
        return score_test,er

    @staticmethod
    def fill_na(df):
        df.fillna({k : 0 for k in CONTINUOUS_COLS}, inplace = True)
        df.fillna({'gb_0d':0,'spend': 0,'ni_0d' : 0},inplace = True)
    
    @staticmethod
    def create_weight_columns(df):
        df['cohort_structure'] = df['title']
        df.loc[df.cohort == 'control','cohort_structure'] = 'control'
        weight = pd.get_dummies(df.cohort_structure)
        Normlize = (weight.sum()/len(weight)).to_dict()
        for k,v in Normlize.iteritems():
            weight[k] = weight[k]/v
        return weight
    
    @staticmethod
    def expected_metrics(score,y,cohort_weight,lambda_saving ):
        
        result = {k : Multi_model.expected_increment(score,
                                      y[k].to_numpy(),
                                      cohort_weight) for k in y.columns}
    
        result['cpigb'] = -result['ni_0d']/result['gb_0d']
        result['ni_cost'] =  -result['ni_0d']
        result['saving'] = lambda_saving*result['gb_0d'] +result['ni_0d']
        return result
    
    @staticmethod
    def expected_increment(score,y,cohort_weight):  
        value_tr = np.mean(np.sum(score*cohort_weight,axis=1)*y)
        value_c = np.mean(cohort_weight['control']*y)
        inc_value =value_tr -value_c
        return inc_value
    
    


if __name__ == "__main__":

    if False:
        from load_data import  load_data
        df = load_data()
        
    model_paramters = {"number_structures" : 3, 
                                "target_loss" : 300, 
                                "verbose" : False,
                                "dim_hidden_lst": [],
                                "obj_rule" : 'lagrangian',
                                "shadow" : 0.7,
                                "lambda_saving":.7,
                                "epochs" :  300,
                                "reg_scale" :.5,
                                "reg_type" :"L1",
                                "standardization_input" : True}
    
    
    train, test = train_test_split(df,test_size= .3)
    dataset = {'train': train, 'test': test}
    avg_score = []
    metric = []
    for l in np.arange(0,5):
        print(l)
        
#        model_paramters['shadow'] =  l
        model = Multi_model(dataset,**model_paramters)
        score, mt = model.run()
        metric.append(mt) 
        avg_score.append([score.mean()])
#    pd.DataFrame(metric).plot.scatter(x="ni_cost", y = "gb_0d" )
#    avg_score = pd.DataFrame([a[0].to_dict() for a in avg_score])
#    avg_score['lambda'] = np.arange(0,1,.1)
#    for r in result:
#            print(.7*r[0]["gb_0d"]-r[0]["spend"])
#        mm = []
#        for r,l in zip(metric,np.arange(0,1,.1)):
#            r.update({'lambda':l })
#            mm.append(r)
#rr =  df[['gb_0d','ni_0d','spend',"cohort","title"]].groupby(["cohort","title"]).mean()
#(rr.loc['treatment'] -rr.loc['control'])['ni_0d']/(rr.loc['treatment'] -rr.loc['control'])['gb_0d']
#
#            

    

#ss = pd.concat([test,score],axis=1)
#a= ss[ss['Enjoy 30% off $50 or more']>.75][CONTINUOUS_COLS].mean()
#b= ss[ss['$10 off min $30 basket']>.75][CONTINUOUS_COLS].mean()
#c= ss[ss['control']>0.75][CONTINUOUS_COLS].mean()

#ax = metric.plot.scatter(x="ni_cost", y = "gb_0d")
#for i, point in metric.iterrows():
#    l = point['lambda']
#    ax.text(point['ni_cost']-.005, point['gb_0d']+.01*ff(), str(round(point['lambda'],1)))
#
#
#ff = lambda : np.random.randint(0,5)