#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:54:19 2020

@author: ehsan.mousavi
"""
import pandas as pd

def  metric_calculators(df,lambdaa = .7):
    rr =  df[['gb_0d','ni_0d','spend',"cohort","title"]].groupby(["cohort","title"]).mean()
    delta_rr = rr.loc['treatment'] -rr.loc['control']
    metric = {'CPIGB': -delta_rr['ni_0d']/delta_rr['gb_0d'],
            'saving' : delta_rr['ni_0d']+lambdaa*delta_rr["gb_0d"] }
    return pd.DataFrame(metric)


'''
lambda =.7
[[{'cpigb': 0.46635901729757506,
   'gb_0d': 0.4864389261030331,
   'ni_0d': -0.22685517955269824,
   'ni_cost': 0.22685517955269824,
   'spend': 0.33926901268488846},
  $10 off min $30 basket       0.161779
  Enjoy 30% off $50 or more    0.681705
  control                      0.156290
  dtype: float32],
 [{'cpigb': 0.46079737121277503,
   'gb_0d': 0.15805226625423163,
   'ni_0d': -0.07283006880417153,
   'ni_cost': 0.07283006880417153,
   'spend': 0.108739089021517},
  $10 off min $30 basket       0.108202
  Enjoy 30% off $50 or more    0.238568
  control                      0.652491
  dtype: float32],
 [{'cpigb': 0.4901656387793605,
   'gb_0d': 0.16732915041675245,
   'ni_0d': -0.08201899990043515,
   'ni_cost': 0.08201899990043515,
   'spend': 0.11987937470391587},
  $10 off min $30 basket       0.108002
  Enjoy 30% off $50 or more    0.217541
  control                      0.674359
  dtype: float32],
 [{'cpigb': 0.48672013390896324,
   'gb_0d': 0.41347930215165274,
   'ni_0d': -0.20124870131183709,
   'ni_cost': 0.20124870131183709,
   'spend': 0.29410087913036154},
  $10 off min $30 basket       0.373820
  Enjoy 30% off $50 or more    0.265728
  control                      0.359516
  dtype: float32],
 [{'cpigb': 0.46566897649694117,
   'gb_0d': 0.29577342545705054,
   'ni_0d': -0.13773250830757905,
   'ni_cost': 0.13773250830757905,
   'spend': 0.20632841617871805},
  $10 off min $30 basket       0.100276
  Enjoy 30% off $50 or more    0.454714
  control                      0.445119
  dtype: float32]]
 
 '''
 
 '''
 [{'cpigb': 0.4998592430298286,
  'gb_0d': 0.16629126807881445,
  'lambda': 0.0,
  'ni_0d': -0.08312222738434649,
  'ni_cost': 0.08312222738434649,
  'spend': 0.12059259477427553},
 {'cpigb': 0.5293994299385173,
  'gb_0d': 0.13785686255791596,
  'lambda': 0.1,
  'ni_0d': -0.07298134445127324,
  'ni_cost': 0.07298134445127324,
  'spend': 0.10417778544324732},
 {'cpigb': 0.49217263830052965,
  'gb_0d': 0.16527738059082875,
  'lambda': 0.2,
  'ni_0d': -0.08134500445678894,
  'ni_cost': 0.08134500445678894,
  'spend': 0.11888542309507591},
 {'cpigb': 0.4910134216562023,
  'gb_0d': 0.1630282167053405,
  'lambda': 0.30000000000000004,
  'ni_0d': -0.08004904251099809,
  'ni_cost': 0.08004904251099809,
  'spend': 0.11693922581115365},
 {'cpigb': 0.511652708213043,
  'gb_0d': 0.2891623958710685,
  'lambda': 0.4,
  'ni_0d': -0.14795072296080425,
  'ni_cost': 0.14795072296080425,
  'spend': 0.2156297770644344},
 {'cpigb': 0.4742070427746779,
  'gb_0d': 0.19464481409040602,
  'lambda': 0.5,
  'ni_0d': -0.0923019416812384,
  'ni_cost': 0.0923019416812384,
  'spend': 0.13641570584480792},
 {'cpigb': 0.5380491100742186,
  'gb_0d': 0.13577886238040549,
  'lambda': 0.6000000000000001,
  'ni_0d': -0.07305569607066698,
  'ni_cost': 0.07305569607066698,
  'spend': 0.10392521338422708},
 {'cpigb': 0.49429792044603366,
  'gb_0d': 0.2885700018296813,
  'lambda': 0.7000000000000001,
  'ni_0d': -0.14263955180751958,
  'ni_cost': 0.14263955180751958,
  'spend': 0.21044875762091425},
 {'cpigb': 0.5331273295118933,
  'gb_0d': 0.5637995240462486,
  'lambda': 0.8,
  'ni_0d': -0.30057693463485297,
  'ni_cost': 0.30057693463485297,
  'spend': 0.4286716280365276},
 {'cpigb': 0.5737612841412792,
  'gb_0d': 0.5636305531982133,
  'lambda': 0.9,
  'ni_0d': -0.3233893899842665,
  'ni_cost': 0.3233893899842665,
  'spend': 0.4482093286668948}]
 
 
 [[$10 off min $30 basket       0.195798
  Enjoy 30% off $50 or more    0.175880
  control                      0.628364
  dtype: float32], [$10 off min $30 basket       0.145111
  Enjoy 30% off $50 or more    0.119076
  control                      0.734737
  dtype: float32], [$10 off min $30 basket       0.105756
  Enjoy 30% off $50 or more    0.278881
  control                      0.615083
  dtype: float32], [$10 off min $30 basket       0.107104
  Enjoy 30% off $50 or more    0.234948
  control                      0.657847
  dtype: float32], [$10 off min $30 basket       0.146887
  Enjoy 30% off $50 or more    0.377228
  control                      0.476165
  dtype: float32], [$10 off min $30 basket       0.173454
  Enjoy 30% off $50 or more    0.306630
  control                      0.519971
  dtype: float32], [$10 off min $30 basket       0.142195
  Enjoy 30% off $50 or more    0.127500
  control                      0.729177
  dtype: float32], [$10 off min $30 basket       0.092825
  Enjoy 30% off $50 or more    0.398622
  control                      0.508626
  dtype: float32], [$10 off min $30 basket       0.583230
  Enjoy 30% off $50 or more    0.311876
  control                      0.104859
  dtype: float32], [$10 off min $30 basket       0.610720
  Enjoy 30% off $50 or more    0.241439
  control                      0.147963
  dtype: float32]]
 
 
 
arr = [{'cpigb': 0.46635901729757506,
   'gb_0d': 0.4864389261030331,
   'ni_0d': -0.22685517955269824,
   'ni_cost': 0.22685517955269824,
   'spend': 0.33926901268488846,
  '$10 off min $30 basket':       0.161779,
  'Enjoy 30% off $50 or more':    0.681705,
  'control' :                    0.156290},
 {'cpigb': 0.46079737121277503,
   'gb_0d': 0.15805226625423163,
   'ni_0d': -0.07283006880417153,
   'ni_cost': 0.07283006880417153,
   'spend': 0.108739089021517,
  '$10 off min $30 basket':      0.108202,
  'Enjoy 30% off $50 or more':  0.238568,
  'control' :                     0.652491},
 {'cpigb': 0.4901656387793605,
   'gb_0d': 0.16732915041675245,
   'ni_0d': -0.08201899990043515,
   'ni_cost': 0.08201899990043515,
   'spend': 0.11987937470391587,
  '$10 off min $30 basket':       0.108002,
  'Enjoy 30% off $50 or more':     0.217541,
  'control' :                         0.674359}
,
 {'cpigb': 0.48672013390896324,
   'gb_0d': 0.41347930215165274,
   'ni_0d': -0.20124870131183709,
   'ni_cost': 0.20124870131183709,
   'spend': 0.29410087913036154,
  '$10 off min $30 basket':       0.373820,
  'Enjoy 30% off $50 or more':    0.265728,
  'control' :                        0.359516}
,
 {'cpigb': 0.46566897649694117,
   'gb_0d': 0.29577342545705054,
   'ni_0d': -0.13773250830757905,
   'ni_cost': 0.13773250830757905,
   'spend': 0.20632841617871805,
  '$10 off min $30 basket':       0.100276,
 'Enjoy 30% off $50 or more':     0.454714,
  'control' :                        0.445119}
]

'''
 