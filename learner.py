import os
import time
import math
import keras
import numpy as np 
import tensorflow as tf

from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras import regularizers

from keras.callbacks import Callback


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # omit tensorflow info and warning messages
starttime = time.time() # for debugging/performance monitoring

class Histories(Callback):
    def __init__(self):
        self.epochs_to_model_costs = {}
        
    def on_train_begin(self, logs={}):
        self.epochs_to_model_costs = {}

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_categorical_accuracy')
        self.epochs_to_model_costs[epoch] = (self.model, val_loss, val_acc)

    def on_train_end(self, logs={}):
        return self.epochs_to_model_costs


class Learner():

    def __init__(self, max_len, alphabet_len, num_categories, train_data, train_labels, valid_data, valid_labels, rnn_type):
        self.max_len = max_len
        self.alphabet_len = alphabet_len
        self.num_categories = num_categories

        self.train_data = train_data
        self.train_labels = train_labels
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.rnn_type = rnn_type
    
    def learn(self, rnn_type, config={}, num_epochs=400, seed=0):
        #activation function, weight initialization
    
        BATCHSIZE  = config.get("batch_size", 128)
        DROPOUT = config.get("dropout", 0.0)
        DROPOUT_RECURRENT = config.get("dropout_recurrent", 0.0)
        NEURONS_DENSE  = config.get("neurons", 100)
        NEURONS_RNN  = config.get("neurons", 100)
        REGULARIZE = config.get("regularize_rate", 0.0)
        LEARNRATE = config.get("learn_rate", 0.001)

        histories = Histories()
        
        if rnn_type == "SimpleRNN":
            model = Sequential()
            model.add(Dense(NEURONS_DENSE, input_shape=(self.max_len, self.alphabet_len), kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE)))
            model.add(SimpleRNN(NEURONS_RNN, input_shape=(NEURONS_DENSE,), return_sequences=False, return_state=False, dropout=DROPOUT, recurrent_dropout=DROPOUT_RECURRENT, kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE), recurrent_regularizer = regularizers.l2(REGULARIZE))) 
            model.add(Dense(self.num_categories, activation='softmax', kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE), activity_regularizer= regularizers.l2(REGULARIZE)))
            adam = Adam(lr=LEARNRATE, beta_1=0.9, beta_2=0.999, decay=1e-6)
            model.compile(loss='categorical_crossentropy',
                          optimizer=adam,
                          metrics=['categorical_accuracy'])
        
        if rnn_type == "LSTM":
            model = Sequential()
            model.add(Dense(NEURONS_DENSE, input_shape=(self.max_len, self.alphabet_len), kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE)))
            model.add(LSTM(NEURONS_RNN, input_shape=(NEURONS_DENSE,), return_sequences=False, return_state=False, dropout=DROPOUT, recurrent_dropout=DROPOUT_RECURRENT, kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE), recurrent_regularizer = regularizers.l2(REGULARIZE))) 
            model.add(Dense(self.num_categories, activation='softmax', kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE), activity_regularizer= regularizers.l2(REGULARIZE)))
            adam = Adam(lr=LEARNRATE, beta_1=0.9, beta_2=0.999, decay=1e-6)
            model.compile(loss='categorical_crossentropy',
                          optimizer=adam,
                          metrics=['categorical_accuracy'])
        
        if rnn_type == "Bidirectional_LSTM":
            model = Sequential()
            model.add(Dense(NEURONS_DENSE, input_shape=(self.max_len, self.alphabet_len), kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE)))
            model.add(Bidirectional(LSTM(NEURONS_RNN, input_shape=(NEURONS_DENSE,), return_sequences=False, return_state=False, dropout=DROPOUT, recurrent_dropout=DROPOUT_RECURRENT, kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE), recurrent_regularizer = regularizers.l2(REGULARIZE)))) 
            model.add(Dense(self.num_categories, activation='softmax', kernel_regularizer=regularizers.l2(REGULARIZE), bias_regularizer= regularizers.l2(REGULARIZE), activity_regularizer= regularizers.l2(REGULARIZE)))
            adam = Adam(lr=LEARNRATE, beta_1=0.9, beta_2=0.999, decay=1e-6)
            model.compile(loss='categorical_crossentropy',
                          optimizer=adam,
                          metrics=['categorical_accuracy'])
        

        model.fit(self.train_data, 
                  self.train_labels,
                  epochs=num_epochs,
                  batch_size=BATCHSIZE,
                  validation_data = (self.valid_data, self.valid_labels), 
                  callbacks=[histories])

        return histories.epochs_to_model_costs



  # The returned epochs_to_model_costs maps epoch numbers to tuples containing (model at epoch, validation loss, lest loss).
