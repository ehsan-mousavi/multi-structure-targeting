#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:31:52 2020

@author: ehsan.mousavi
"""
import tensorflow as tf
import numpy as np
import pandas as pd
#    X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
#    y = tf.placeholder(tf.float32, shape=[None, 2], name='y')
#    cohort_weight =  tf.placeholder(tf.float32, shape=[None, self.number_structures], name='cohort_weight')
#    
#    o = tf.nn.softmax(X)
#    sess = tf.Session()
#
#
#data=[[1.0,2.0,4.0,5.0],[0.0,6.0,7.0,8.0],[8.0,1.0,1.0,1.0]]
#X=tf.constant(data)
#matResult=tf.matmul(X, X, transpose_b=True)
#
#multiplyResult=tf.reduce_sum(tf.multiply(X,X),axis=1)
#with tf.Session() as sess:
#   print('matResult')
#   print(sess.run([matResult]))
#   print()
#   print('multiplyResult')
#   print(sess.run([multiplyResult]))
#   
#   
#   X_1 = tf.placeholder(tf.float32, name = "X_1")
#   X_2 = tf.placeholder(tf.float32, name = "X_2")
#   a   = tf.placeholder(tf.float32, shape=[1], name='control_cost')
#
#   o = tf.nn.softmax(X_1)
#   multiply = tf.multiply(X_1, X_2, name = "multiply")
#   u= tf.reduce_mean(multiply) - a
#
#   with tf.Session() as session:
#       
#        result = session.run([multiply ,o, u], feed_dict={X_1:[1,2,3], X_2:[4,5,6], a : [8]})
#        print(result)
#        
#        
#        
#        
#        
# ------------------------------------------------

class Simple_forward_model():
        
    def __init__(self,
                 number_structures,
                 keep_prob  = .5,
                 dim_inputs=None, 
                 dim_hidden_lst=[], 
                 obj_rule='cpiv',
                 shadow=None, 
                 est_individual=False,
                 learning_rate=1e-1,
                 reg_scale=1e-1, reg_type='L2', epochs=100, target_loss=None, print_every=10,
                 verbose=True, plot_losses=False, random_seed=None, **kwargs):

        # obj_rule = 'lagrangian'
        # shadow = .35


        epochs =  100
        self.number_structures = number_structures
        self.keep_prob = keep_prob
        self.dim_inputs = dim_inputs
        self.dim_hidden_lst = dim_hidden_lst

        self.obj_rule = obj_rule
        self.est_individual = est_individual
        self.shadow = shadow
        self.learning_rate = learning_rate
        self.reg_scale = reg_scale
        self.reg_type = reg_type
        self.epochs = epochs
        self.target_loss = target_loss
        self.print_every = print_every
        self.verbose = verbose
        self.plot_losses = plot_losses
        self.random_seed = random_seed

        # non-initialized session
        self.sess = None

        pass
    
    
    
    def _create_placeholders(self):
            """Create placeholders for input data."""
    
            with tf.name_scope("data"):
                self.X = tf.placeholder(tf.float32, shape=[None, self.dim_inputs], name='X')
                self.value = tf.placeholder(tf.float32, shape=[None], name='value')
                self.cost = tf.placeholder(tf.float32, shape=[None], name='cost')
        #        self.sample_weight = tf.placeholder(tf.float32, shape=[None, 1], name='sample_weight')
                self.cohort_weight =  tf.placeholder(tf.float32, shape=[None, self.number_structures], name='cohort_weight')
                self.control_value = tf.placeholder(tf.float32, shape=[1], name='control_value')
                self.control_cost = tf.placeholder(tf.float32, shape=[1], name='control_cost')
                
    def _create_variables(self):
          with tf.name_scope("variable"):
              self.W =  tf.placeholder(tf.float32, 
                                       shape  = [self.dim_inputs,  self.number_structures],
                                       name = 'weight')
#              self.b = tf.placeholder(tf.float32, shape  = [self.dim_inputs,  self.number_structures]
 #                                      'name' = 'weight')
              
              self.dim_lst = [self.dim_inputs] + self.dim_hidden_lst + [self.number_structures]
              self.W_lst = [self.W ]
        
                    
    def _create_prediction(self):
            """Create model predictions."""
    
    
            with tf.name_scope("prediction"):
                h = self.X
                for i in range(len(self.dim_lst) - 1):
                    # not output layer, has bias term
                    if i < len(self.dim_lst) - 2:
                        h = tf.matmul(h, self.W_lst[i]) + self.b_lst[i]
                        h = tf.nn.relu(h)
                        h = tf.nn.dropout(h, keep_prob=self.keep_prob)
    
                    # output layer
                    else:
                        h = tf.matmul(h, self.W_lst[i])
                        h = tf.nn.softmax(h)
                self.score = h

    def _create_loss(self):
        """Create loss based on true label and predictions."""

        with tf.name_scope("loss"):
            self.promo_prob=tf.reduce_sum(tf.multiply(self.score, self.cohort_weight),
                                          axis=1)
            inc_value  =  tf.reduce_mean(tf.multiply(self.promo_prob, self.value))- self.control_value
            inc_cost = tf.reduce_mean( tf.multiply(self.promo_prob, self.cost)) - self.control_cost
            self.objective  =  inc_cost/inc_value
            reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.objective +reg_loss

    def _create_optimizer(self):
        """Create optimizer to optimize loss."""

        with tf.name_scope("optimizer"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_graph(self):
        """Build the computation graph."""

        self.graph = tf.Graph()

        # set self.graph as default graph
        with self.graph.as_default():

            self._create_placeholders()
            self._create_variables()
            self._create_prediction()
            self._create_loss()
       #     self._create_optimizer()
            self._init = tf.global_variables_initializer()


        # create session
        self.sess = tf.Session(graph=self.graph)
        
    def calculte_control_cost_value(self,y,control_weight):
        N = y.shape[0]
        vc = np.dot(y[:,0],control_weight)/N
        cc = np.dot(y[:,1],control_weight)/N
        return vc,cc
        
        
    def fit(self, X=None,
            y=None,
            cohort_weight = None, 
            control_column=-1 ,
            sample_weight=None, W=None):
       
        self.dim_inputs = X.shape[1]
        self._build_graph()
        variables = [self.loss, self.objective, self.W_lst, self.score]
        
        
        #calculate the control cost and  value
        
        value_c,cost_c = self.calculte_control_cost_value(y,cohort_weight[:,control_column])
        # initialize variables
        self.sess.run(self._init)
        feed_dict = {
                self.X: X,
                self.value: y[:,0],
                self.cost: y[:,1],
                self.cohort_weight: cohort_weight,
                self.control_value: [value_c],
                self.control_cost: [cost_c],
                self.W: W,
            }
        loss, objective, train_step,score= self.sess.run(variables, feed_dict=feed_dict)
        return loss, objective, train_step 



def generate_fake_data(N,feature=30, s= 4):
    
    X  = np.random.rand(N,feature)
    y  = np.random.rand(N,2)
    prob  = [.85/(s-1) for  _ in range(s-1)]+[.15] 

    cohort = np.random.choice(4,p = prob, size = N)
    cohort_weight = np.zeros([N,s])
    for i,u in enumerate(cohort):
        cohort_weight[i,u] = prob[u]
        
    W  = np.random.rand(feature,s)
        
    return X,y,cohort_weight, W

def nomalization(A):
    Z=np.sum(A,axis = 1) 
    Z = Z.reshape(Z.shape[0],1)
    Z = np.repeat(Z,A.shape[1],axis = 1)*1.0
    return A/Z
    
    (A.shape[0])
    return A/np.reshape(np.sum(A,axis = 1),A.shape[1])

def forward_model(X,y,cohort_weight, W):
    h = np.dot(X,W)
    N = X.shape[0]
    score =nomalization(np.exp(h))
    E_gb_p = np.sum(score[:,:-1]*cohort_weight[:,:-1],axis =1)*y[:,0]
    E_S_p = np.sum(score[:,:-1]*cohort_weight[:,:-1],axis =1)*y[:,1]
    E_gb_np = (1-score[:,-1])*cohort_weight[:,-1]*y[:,0]
    E_S_np = (1-score[:,-1])*cohort_weight[:,-1]*y[:,1]
    delta_gb = sum(E_gb_p)/N-sum(E_gb_np)/N
    delta_spend = sum(E_S_p)/N-sum(E_S_np)/N
    
    v = np.sum(score*cohort_weight,axis=1)*y[:,0]
    vt = np.mean(np.sum(score*cohort_weight,axis=1)*y[:,0])
    ct = np.mean(np.sum(score*cohort_weight,axis=1)*y[:,1])
    vc = np.mean(cohort_weight[:,-1]*y[:,0])
    cc = np.mean(cohort_weight[:,-1]*y[:,1])
    
    return delta_spend/delta_gb,delta_gb,delta_spend,score, v, vt


def expected_metrics(score,y,cohort_weight):  
    value = np.sum(score*cohort_weight,axis=1)*y[:,0]
    value_tr = np.mean(np.sum(score*cohort_weight,axis=1)*y[:,0])
    cost_tr = np.mean(np.sum(score*cohort_weight,axis=1)*y[:,1])
    value_c = np.mean(cohort_weight[:,-1]*y[:,0])
    cost_c = np.mean(cohort_weight[:,-1]*y[:,1])
    inc_value =value_tr -value_c
    inc_cost = cost_tr- cost_c
    return inc_value,inc_cost
    
    

if __name__ == "__main__":

    N  = 1000
    X,y,cohort_weight, W = generate_fake_data(N,feature = 30,  s= 4)
    cpigb,inc_c,inc_v,score,v, vt = forward_model(X,y,cohort_weight, W)
    
    model = Simple_forward_model(4)
    loss, objective, train_step = model.fit( X=X,y=y,
                                                                    cohort_weight = cohort_weight, 
                                                                    control_column=-1 ,
                                                                    sample_weight=None, W=W)
    print('TF-loss: {}\tTF-objective\t{}\t'.format(loss, objective))
    print('cpigb:{}\t cost:{} \t  value:{} cost:{} \t  value:{}'.format( a,b,c,dv,dc))
    print(ex_gb, vt)   
    print(cpigb-objective[0])
