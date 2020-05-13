#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:42:08 2020

@author: ehsan.mousavi
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#ModelType.DRM_MODEL = "DRM"

from datetime import datetime 
get_dt_str =  lambda  :datetime.now().strftime(format = "%b %d %Y %H:%M:%S")

class BaseModel():
    def __init__(self):
        pass

class MULTI_DRM_Gradient(BaseModel):
    """Direct Ranking Model using Gradient Descent."""

    #model_type = ModelType.DRM_MODEL

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
                 verbose=True, plot_losses=False, random_seed=None,
                 standardization_input=True,**kwargs):
        """
        Set hyper-parameters and construct computation graph.
        :param number_structures: number of different  structures (including the control)
        :param keep_prob: keep probboblity in drop-out regularization
        :param dim_inputs: input feature dimensions, if None, then this value will be inferred during fit
        :param dim_hidden_lst: number of neurons for hidden layers ([] means linear model)
        :param obj_rule: objective to optimize during training, could be cpiv (inc cost per inc value),
        lagrangian (lambda expression), value, cost (maximize delta values) or ivc (inc value per inc cost)
        :param est_individual: if True, then estimate the individual incremental value or cost, which depends on
        parameter obj_rule (only for value or cost)
        :param shadow: shadow price (lambda) for Lagrange objective, if None, use CPIT
        :param learning_rate: learning rate for gradient descent
        :param reg_scale: regularization factor
        :param reg_type: type of regularization, 'L1' or 'L2'
        :param epochs: number of epochs for training
        :param target_loss: target loss for training
        :param print_every: print every n training epochs
        :param verbose: if verbose during training
        :param plot_losses: if plot losses during training
        :param random_seed: random seed used to control the graph, if None then not control for random seed
        :param standardization_input: transfer the input to mean zero variance 1
        """
       # super(MULTI_DRM_Gradient, self).__init__()

        plot_losses = True
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
        self.standardization_input = standardization_input

        # non-initialized session
        self.sess = None
        self.scaler = None
        

        # build graph here if dim_inputs is passed
        if self.dim_inputs is not None:
            self._build_graph()

        # dictionary hold training statistics
        self.train_stats = {}
    
    def get_params(self):
            """
            :return: dictionary of hyper-parameters of the model.
            """
            return {
                'dim_inputs': self.dim_inputs,
                'dim_hidden_lst': self.dim_hidden_lst,
                'obj_rule': self.obj_rule,
                'shadow': self.shadow,
                'est_individual': self.est_individual,
                'learning_rate': self.learning_rate,
                'reg_scale': self.reg_scale,
                'reg_type': self.reg_type,
                'epochs': self.epochs,
                'target_loss': self.target_loss,
                'print_every': self.print_every,
                'verbose': self.verbose,
                'plot_losses': self.plot_losses,
                'random_seed': self.random_seed,
            }
        
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
            """Create variables for the model."""

    
            with tf.name_scope("variable"):
                if self.reg_type == 'L2':
                    regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_scale)
                else:
                    regularizer = tf.contrib.layers.l1_regularizer(scale=self.reg_scale)
    
                self.dim_lst = [self.dim_inputs] + self.dim_hidden_lst + [self.number_structures]
                print(self.dim_lst)
    
                self.W_lst = []
                self.b_lst = []
                for i in range(len(self.dim_lst)-1):
                    self.W_lst.append(tf.get_variable(
                        "W{}".format(i+1),
                        shape=[self.dim_lst[i], self.dim_lst[i+1]],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        regularizer=regularizer)
                    )
                    # not output layer, has bias term
                    if i < len(self.dim_lst) - 2:
                        self.b_lst.append(tf.get_variable("b{}".format(i+1), shape=[self.dim_lst[i+1]]))
        
    def _create_prediction(self):
            """Create model predictions."""
            epsilon = 1e-3

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
                     #   batch_mean, batch_var = tf.nn.moments(h,[0])
                     #    scale = tf.Variable(tf.ones([self.dim_lst[-1]]))
                    #    beta = tf.Variable(tf.zeros([self.dim_lst[-1]]]))
                    #    BN = tf.nn.batch_normalization(h,
                     #                                   batch_mean,
                     #                                   batch_var,
                     #                                   beta,
                     #                                   scale,
                       #                                 epsilon)
                    #    h = tf.nn.softmax(BN)
                        
                     #   h = tf.nn.softmax(20*tf.nn.tanh(h))
                        h = tf.nn.softmax(20*h)
                self.score = h
 
    def _create_loss(self):
        """Create loss based on true label and predictions."""

        with tf.name_scope("loss"):
            
           # gini=(tf.nn.l2_loss(    self.score))/100000
            gini = tf.losses.softmax_cross_entropy(self.score, 0*self.score)
            
            promo_prob=tf.reduce_sum(tf.multiply(self.score, self.cohort_weight),
                                          axis=1)
            inc_value  =  tf.reduce_mean(tf.multiply(promo_prob, self.value))- self.control_value
            inc_cost = tf.reduce_mean( tf.multiply(promo_prob, self.cost)) - self.control_cost
            


            # determine loss function based on self.obj_rule
            if self.obj_rule == 'cpiv':
                self.objective = inc_cost / inc_value

            elif self.obj_rule == 'ivc':
                # maximize ivc
                self.objective = - inc_value / inc_cost

            elif self.obj_rule == 'lagrangian':
                assert self.shadow is not None, 'Need to pass in shadow value if use lagrangian as obj_rule.'
                self.objective = inc_cost - self.shadow * inc_value

            elif self.obj_rule == 'value':
                # maximize delta values
                self.objective = - inc_value

            # use only cost as objective
            elif self.obj_rule == 'cost':
                # maximize delta cost
                self.objective = - inc_cost

            else:
                raise Exception('Invalid obj_rule!')

            # regularization
            reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
          #   weights = tf.trainable_variables()  # all vars of your graph
          #   reg_loss = tf.norm( weights,ord=1)

            # final loss
            self.loss = self.objective +reg_loss+.1*gini

    def _create_optimizer(self):
        """Create optimizer to optimize loss."""

        with tf.name_scope("optimizer"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_graph(self):
        """Build the computation graph."""

        self.graph = tf.Graph()

        # set self.graph as default graph
        with self.graph.as_default():
            # # clear old variables
            # tf.reset_default_graph()

            # set random seed
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)

            self._create_placeholders()
            self._create_variables()

            self._create_prediction()

            self._create_loss()
            self._create_optimizer()

            self._init = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

        # create session
        self.sess = tf.Session(graph=self.graph)

    @staticmethod
    def _calculate_avg_inc_value_cost(y):
        """
        Calculate average incremental values and cost
        :param y: numpy array, [value, cost, is_treatment]
        :return: numpy array with 2 number, [avg inc value, avg inc cost]
        """
        is_treatment = y[:, -1].astype(bool)
        y_t = y[is_treatment, :2]
        y_c = y[~is_treatment, :2]
        return np.mean(y_t, axis=0) - np.mean(y_c, axis=0)

    def save(self, path=None, spark=None):
        """
        Derived from BaseModel class.
        Save the model: model.save_model(path='model_ckpts/model.ckpt')
        Saved files: checkpoint, model.ckpt.meta, model.ckpt.index, model.ckpt.data-00000-of-00001
        """
        print({
            'msg': 'DRM_Gradient.save start',
            'path': path,
        })

        create_dir_if_not_exist(path)
        self.saver.save(self.sess, path)

        model_info = {
            'model_type': self.model_type,
            'path': path,
            'params': self.get_params(),
        }
        print({
            'msg': 'DRM_Gradient.save finish',
            'model_info': model_info,
        })
        return model_info

    @classmethod
    def load(cls, model_info=None, abs_dir=None, spark=None):
        """
        Derived from BaseModel class.
        Load the model.
        """
        print({
            'msg': 'DRM_Gradient.load start',
            'model_info': model_info,
            'abs_dir': abs_dir,
        })

        model = DRM_Gradient(**model_info['params'])

        abs_path = get_abs_path(model_info['path'], abs_dir=abs_dir)
        model.saver.restore(model.sess, abs_path)
        print({
            'msg': 'DRM_Gradient.load finish',
        })
        return model
    
    def calculte_control_cost_value(self,y,control_weight):
        N = y.shape[0]
        vc = np.dot(y[:,0],control_weight)/N
        cc = np.dot(y[:,1],control_weight)/N
        return vc,cc
        

    def fit(self, X=None,
            y=None,
            cohort_weight = None, 
            control_column=-1 ,
            sample_weight=None, **kwargs):
        """
        Train the model
        :param X: input features
        :param y: label value, cost 
        :param cohort_weight 
        :param control_column: the column at which we have control
        :param sample_weight: array of weights assigned to individual samples. If not provided, then unit weight
        :return: None
        """


        assert self.target_loss is not None, "Must pass in target_loss!"
        assert y.shape[1] == 2, 'y should have 2 columns!'

    #    sample_weight = check_sample_weight(sample_weight, [y.shape[0], 1])

        # infer dim_inputs from X
        self.dim_inputs = X.shape[1]
        
        if self.standardization_input:
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)

        # TensorFlow initialization
        self._build_graph()

        # setting up variables we want to compute (and optimizing)
        variables = [self.loss, self.objective, self.train_step, self.W_lst]

        #calculate the control cost and  value
        
        value_c,cost_c = self.calculte_control_cost_value(y,cohort_weight[:,control_column])
        # initialize variables
        self.sess.run(self._init)

        # record losses history
        objective_lst = []
        loss_lst = []
        
        
        for e in range(self.epochs):

            # gradient descent using all data
            # create a feed dictionary for this batch
            feed_dict = {
                self.X: X,
                self.value: y[:,0],
                self.cost: y[:,1],
                self.cohort_weight: cohort_weight,
                self.control_value: [value_c],
                self.control_cost: [cost_c],
     #           self.sample_weight: sample_weight,
            }

            loss, objective, train_step, W_lst = self.sess.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            # convert to float so it can be serialized to JSON
            loss_lst.append(float(loss))
            objective_lst.append(float(objective))

            # print every now and then
            if ((e + 1) % self.print_every == 0 or e == 0) and self.verbose:
                print("Epoch {0}: with training loss = {1}".format(e + 1, loss[0]))

        final_loss = loss_lst[-1]

        if self.verbose:
            print({
           #     'ts': get_dt_str(),
                'msg': 'DRM_gradient.fit',
                'final_loss': final_loss,
                'target_loss': self.target_loss,
            })

        assert final_loss < self.target_loss, \
            'Final loss: {}, target loss {} not reached, terminated'.format(final_loss, self.target_loss)

        # calculate average incremental value and cost in training set
        self.avg_inc_value_cost = self._calculate_avg_inc_value_cost(y)

        if self.plot_losses:
            plt.plot(loss_lst)
            plt.plot(objective_lst)
            plt.grid(True)
            plt.title('Historical Loss')
            plt.xlabel('Epoch Number')
            plt.ylabel('Epoch Loss')
            plt.show()

        # ToDo modularize logging the training statistics
        self.train_stats['objective_lst'] = objective_lst
        self.train_stats['loss_lst'] = loss_lst

        if self.verbose:
            print({
                'ts': get_dt_str(),
                'msg': 'DRM_gradient.fit finish',
                'final_loss': final_loss,
                'target_loss': self.target_loss,
                # 'W_lst': W_lst,
                'W_lst[0].shape': W_lst[0].shape,
                'W_lst[0].type': type(W_lst[0]),
            })


    def predict(self, X, **kwargs):
        """
        Predict
        :param X: features in numpy array
        :return:
        """
        
        if self.standardization_input:
            assert self.scaler is not None, "Training is not standardized"
            X = self.scaler.transform(X)
            
        assert self.sess is not None, "Model has not been fitted yet!"
        score = self.sess.run(self.score, feed_dict={self.X: X})
        return score

    @property
    def weights_lst(self):
        """
        Return the list of weights (length of this list is the number of layers).
        :return: a list with weights in each layer
        """
        assert self.sess is not None, "Model has not been fitted yet!"
        return self.sess.run(self.W_lst)

    @property
    def coef_(self):
        """
        Estimated coefficients for the linear DRM
        :return: array, shape (n_features, )
        """
        assert self.sess is not None, "Model has not been fitted yet!"
        return self.sess.run(self.W_lst)[0]

    def get_metrics(self):
        """
        Get metrics of the model.
        :return: a list of json representing metrics.
        """
        f = Figure(title='DRM_Gradient Train Loss', x_axis_label='Epoch', y_axis_label='Value')
        f.line(color='blue',
               x=range(len(self.train_stats['loss_lst'])),
               y=self.train_stats['loss_lst'],
               legend='Loss')
        f.line(color='green',
               x=range(len(self.train_stats['objective_lst'])),
               y=self.train_stats['objective_lst'],
               legend='CPIT')
        return [f.draw()]


    
if __name__ == "__main__":
    print("main DRM function")

    