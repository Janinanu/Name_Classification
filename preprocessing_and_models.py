#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:04:42 2018

@author: janinanu
"""


# Import `tensorflow`
import tensorflow as tf
import os
from os import listdir
import matplotlib.pyplot as plt 
import random
import unicodedata
import re
import numpy as np
from numpy import array
from numpy import argmax
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras import regularizers

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

#%%

def load_names_dict(data_directory):
    
    lang_names_dict = {}
    
    for f in listdir(data_directory):
        file = os.path.join(data_directory, f)
        file_reader = open(file, "r")
        names_raw = file_reader.readlines()
        names_sorted = sorted(names_raw)
        
        names_list = []
        
        for name in names_sorted:
            if name.strip("\n").lower() not in names_list:
                names_list.append(name.strip("\n").lower())
        lang_names_dict[f.split(".")[0]] = names_list
        
        file_reader.close()
        
    return lang_names_dict
#%%
lang_names_dict = load_names_dict("/home/janinanu/Language_Identification/names")

#%%

def draw_distribution(lang_names_dict):
    count_dict = {}
    for language, name_list in lang_names_dict.items():
        count_dict[language] = len(name_list)
    
    plt.bar(range(len(count_dict)), count_dict.values())
    plt.xticks(range(len(count_dict)), count_dict.keys(), rotation='vertical')
    plt.xlabel('Language')
    plt.ylabel('Name count')
    plt.show()    
    
#%%
draw_distribution(lang_names_dict)
#%% 
def undersample_names(max_count):
    for language, name_list in lang_names_dict.items():
        if len(name_list) > max_count:
            shuffled_names = random.sample(name_list, len(name_list))
            name_list_subsample = sorted(shuffled_names[:max_count])
            lang_names_dict[language] = name_list_subsample
    
    return None
#%%
undersample_names(300)
#%%
def remove_intruders():
    for language, name_list in lang_names_dict.items():
        for name in name_list:
            if name in ["To the first page", "Get'Man", "/B"]:
                del name
#%%
remove_intruders()
#%%
draw_distribution(lang_names_dict)
#%%
def normalize_clean_name(name_unicode):
    strip_accents = ''.join(char for char in unicodedata.normalize('NFD', name_unicode)
                              if unicodedata.category(char) != 'Mn')
    
    replace_l_ss = strip_accents.replace("ß", "ss").replace("ł", "l")
    
    name_normalized_cleaned = re.sub('[:/1,]', '', replace_l_ss)
        
    return name_normalized_cleaned

#%%

def normalize_clean_names_dict():
    for language, name_list in lang_names_dict.items():
        lang_names_dict[language] = [normalize_clean_name(name) for name in name_list]
#%%
normalize_clean_names_dict()
#%%
def create_alphabet():
    letters = []
    for language, name_list in lang_names_dict.items():
        for name in name_list:
            for letter in name:
                if letter not in letters:
                    letters.append(letter)
    letters_sorted = sorted(letters)
    alphabet = ''.join(letters_sorted)
    
    return alphabet
#%%
alphabet = create_alphabet()

#%%
def get_max_len():
    max_len = 0
    for language, name_list in lang_names_dict.items():
        for name in name_list:
            if len(name)>max_len:
                max_len = len(name)
    return max_len
#%%
max_len = get_max_len()

#%% 
def name_to_array(name, max_len):
    one_hot_name = to_categorical([alphabet.find(letter) for letter in name], num_classes=len(alphabet))
    num_pads = max_len-len(name)
    one_hot_name_padded = np.vstack((one_hot_name, np.zeros((num_pads,len(alphabet)))))
    
    return one_hot_name_padded
#%%
def array_to_name(name_array): #max_len x len(alphabet)
    name = ""
    for letter_array in name_array:
        for position in letter_array:
            if position == 1.0:
                index = np.where(letter_array==position)
                letter = alphabet[index[0][0]]
                name += letter
    return name 
  
#%%
def create_name_language_pair_list():
    pair_list = []
    for language, name_list in lang_names_dict.items():
        for name in name_list:
            name_language_pair = (name, language)
            pair_list.append(name_language_pair) 
    #pair_list_sorted = sorted(pair_list, key=lambda x: len(x[0]), reverse=True)
    return pair_list
#%%
pair_list =  create_name_language_pair_list()
#%%
def define_categories():
    categories = []
    for language, name_list in lang_names_dict.items(): 
        if language not in categories:
            categories.append(language)
    categories_sorted = sorted(categories)
    num_categories = len(categories_sorted)

    return categories_sorted, num_categories

#%%
categories_sorted, num_categories = define_categories()
#%%
def create_one_hot_labels():   
    one_hot_labels = to_categorical([l for l in range(num_categories)], num_classes=num_categories)
    return one_hot_labels

#%%
one_hot_labels = create_one_hot_labels()
#%%
def create_name_language_tensor_pair_list():
    tensor_pair_list = []

    for (name, language) in pair_list:
        
        one_hot_name_padded = name_to_array(name, max_len)
       
        one_hot_label_index = categories_sorted.index(language)
        
        tensor_pair_list.append((one_hot_name_padded, one_hot_labels[one_hot_label_index]))
        
    return tensor_pair_list
#%%
tensor_pair_list = create_name_language_tensor_pair_list()

#%%
def create_data_label_tensors():
    
    name_tensor_aux_list = []
    label_tensor_aux_list = []
    
    for (name_tensor, label_tensor) in tensor_pair_list:
        name_tensor_aux_list.append(name_tensor)
        label_tensor_aux_list.append(label_tensor)
    data = np.stack(name_tensor_aux_list, axis=0)
    labels = np.stack(label_tensor_aux_list, axis = 0)
    
    return data, labels
#%%
data, labels = create_data_label_tensors()
#%%

def shuffle_in_unison(data, labels):
    assert data.shape[0] == labels.shape[0]
    p = np.random.permutation(data.shape[0])
    data_shuffled = data[p]
    labels_shuffled = labels[p]

    return data_shuffled, labels_shuffled

#%%
def train_valid_test_split(data, labels):
    
    data_shuffled, labels_shuffled = shuffle_in_unison(data, labels)
    
    train_split = int(0.8*data.shape[0])
    valid_split = int(0.1*data.shape[0])
    
    train_data = data_shuffled[:train_split]
    train_labels = labels_shuffled[:train_split]
    valid_data = data_shuffled[train_split:train_split+valid_split]
    valid_labels = labels_shuffled[train_split:train_split+valid_split]
    test_data = data_shuffled[train_split+valid_split:]
    test_labels = labels_shuffled[train_split+valid_split:]
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels
#%%
train_data, train_labels, valid_data, valid_labels, test_data, test_labels = train_valid_test_split(data, labels)

#%%

#GRID SEARCH: neurons, lr, dropout, kernel_regularizer, initializers, batch_size

def create_model():
        model = Sequential()
        model.add(Dense(100, input_shape=(max_len, len(alphabet)), kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.85))
        model.add(LSTM(100, input_shape=(100,))) 
        model.add(Dropout(0.85))
        model.add(Dense(num_categories, activation='softmax'))
        
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])	
        
        return model

#%%
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)

batch_size = [10,20]
param_grid = dict(batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(train_data, train_labels)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#%%
model = Sequential()

model.add(Dense(200, input_shape=(max_len, len(alphabet)), kernel_regularizer=regularizers.l2(0.000001), bias_regularizer= regularizers.l2(0.000001)))

model.add(Dropout(0.9))

model.add(LSTM(200, input_shape=(200,), kernel_regularizer=regularizers.l2(0.000001), bias_regularizer= regularizers.l2(0.000001), recurrent_regularizer = regularizers.l2(0.000001))) 

model.add(Dropout(0.9))

model.add(Dense(num_categories, activation='softmax', kernel_regularizer=regularizers.l2(0.000001), bias_regularizer= regularizers.l2(0.000001), activity_regularizer= regularizers.l2(0.000001)))

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_3', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(train_data, train_labels,
          epochs=200,
          batch_size=100,
          validation_data = (valid_data, valid_labels), 
          callbacks=[tbCallBack])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()



#%%
score = model.evaluate(test_data, test_labels, batch_size=10)
print(score)

#%%
from keras.layers import Conv1D, MaxPooling1D, Flatten

model = Sequential()
model.add(Conv1D(20, kernel_size=(5),
                 activation='relu',
                 input_shape=(max_len, len(alphabet)), kernel_regularizer=regularizers.l2(0.00000001), bias_regularizer= regularizers.l2(0.00000001), activity_regularizer= regularizers.l2(0.00000001)))
model.add(MaxPooling1D(pool_size=(2)))
model.add(Dropout(0.8))
#model.add(Conv1D(256,kernel_size=(5), activation='relu', kernel_regularizer=regularizers.l2(0.00001), bias_regularizer= regularizers.l2(0.00001), activity_regularizer= regularizers.l2(0.00001)))
#model.add(MaxPooling1D(pool_size=(2)))
#model.add(Dropout(0.75))
model.add(Flatten())
model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.00000001), bias_regularizer= regularizers.l2(0.00000001), activity_regularizer= regularizers.l2(0.00000001)))
model.add(Dropout(0.8))
model.add(Dense(num_categories, activation='softmax'))

adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph_cnn_24', histogram_freq=0, write_graph=True, write_images=True)

history = model.fit(train_data, train_labels,
          epochs=4000,
          batch_size=100,
          validation_data = (valid_data, valid_labels), 
          callbacks=[tbCallBack])


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


#%%
score = model.evaluate(test_data, test_labels, batch_size=10)
print(score)










