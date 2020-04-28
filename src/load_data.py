#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:28:48 2020

@author: ehsan.mousavi
"""

import pandas as pd
import os 



def load_data():
    data_source = '/Users/ehsan.mousavi/Documents/luckycharm/data_sources/march2020'
    os.chdir(data_source) 

    dfs =[]
    names = ['march-13.csv', 'March-13-2.csv',    'March-13-3.csv'  ]
    for nm in names:
            dfs.append(pd.read_csv(nm,na_values ='\N'))
    df = pd.concat(dfs)
    return df 