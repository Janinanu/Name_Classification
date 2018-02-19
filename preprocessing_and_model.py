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
            if name.strip("\n") not in names_list:
                names_list.append(name.strip("\n"))
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
def split_data():
    train_split = 0.8
    valid_split = 0.1
    #test_split = 0.1
    
    names_train_dict = {}
    names_valid_dict = {}
    names_test_dict = {}
    
    for language, name_list in lang_names_dict.items():
        shuffled_names = random.sample(name_list, len(name_list))
        names_train_dict[language] = sorted(shuffled_names[:int(len(shuffled_names)*train_split)])
        names_valid_dict[language] = sorted(shuffled_names[int(len(shuffled_names)*train_split):int(len(shuffled_names)*train_split) + int(len(shuffled_names)*valid_split)])
        names_test_dict[language] = sorted(shuffled_names[int(len(shuffled_names)*train_split) + int(len(shuffled_names)*valid_split):])
    return names_train_dict, names_valid_dict, names_test_dict
#%%
names_train_dict, names_valid_dict, names_test_dict = split_data()
#letter to index - letter to tensor - name to tensor

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
def name_to_array(name):
    one_hot_name = to_categorical([alphabet.find(letter) for letter in name], num_classes=len(alphabet))
    num_pads = max_len-len(name)
    one_hot_name_padded = np.vstack((one_hot_name, np.zeros((num_pads,len(alphabet)))))
    return one_hot_name_padded
  
        
#%%

def create_name_array_dict(names_dict): #names_train_dict, names_valid_dict, names_test_dict
    names_array_dict = {}
    for language, name_list in sorted(names_dict.items()):
            for name in name_list:  
                names_array_dict[language] = [name_to_array(str(name)) for name in name_list]
   
#    in_dict = str(names_dict)
#    format_add = in_dict.split("_")[0]
#    out_dict = "name_vecs_%s_dict" % (format_add) 
#    
#    out_dict = names_vector_dict

    return names_array_dict
    #return name_vecs_%s_dict (% names_dict.split("_")[1])
#%%
name_array_train_dict = create_name_array_dict(names_train_dict) 
name_array_valid_dict = create_name_array_dict(names_valid_dict) 
name_array_test_dict = create_name_array_dict(names_test_dict) 
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
def create_name_tensor_dict(names_array_dict):
    name_tensor_dict = {}
    for language, vec_list in sorted(names_array_dict.items()):
        name_tensor_dict[language] = np.concatenate(vec_list, axis = 0).reshape(-1, max_len, len(alphabet))
    return name_tensor_dict
#%%

name_tensor_train_dict = create_name_tensor_dict(name_array_train_dict)
name_tensor_valid_dict = create_name_tensor_dict(name_array_valid_dict)
name_tensor_test_dict = create_name_tensor_dict(name_array_test_dict)


#%%
def create_label_tensor_dict(name_tensor_dict):
    
    one_hot_labels = to_categorical([l for l in range(len(name_tensor_dict))], num_classes=len(name_tensor_dict))
    
    label_tensor_dict = {} #num_examples_per_lang x num_langs
    
    for i, (language, name_tensor) in enumerate(sorted(name_tensor_dict.items())): #108 x 18
        label_tensor_dict[language] = np.zeros((name_tensor_dict[language].shape[0], one_hot_labels[i].shape[0]))
        label_tensor_dict[language][:] = one_hot_labels[i]
    
    return label_tensor_dict
        
#%%%
label_tensor_train_dict = create_label_tensor_dict(name_tensor_train_dict)
label_tensor_valid_dict = create_label_tensor_dict(name_tensor_valid_dict)
label_tensor_test_dict = create_label_tensor_dict(name_tensor_test_dict)
#%%
def create_data(name_tensor_dict):
    #count_examples = 0
    stack_data = np.ones((1, max_len, len(alphabet)))
    for language, tensor in sorted(name_tensor_dict.items()):
        stack_data = np.concatenate((stack_data, tensor), axis = 0)
        #count_examples += tensor.shape[0]
    data = stack_data[1:]
    return data
#%%
train_data = create_data(name_tensor_train_dict)
valid_data = create_data(name_tensor_valid_dict)
test_data = create_data(name_tensor_test_dict)

#%%
def create_labels(label_tensor_dict):
    num_classes = len(label_tensor_dict)
    stack_labels = np.ones((1, num_classes))
    for language, tensor in sorted(label_tensor_dict.items()):
        stack_labels = np.concatenate((stack_labels, tensor), axis = 0)
    labels = stack_labels[1:]
    return labels
#%%
train_labels = create_labels(label_tensor_train_dict)
valid_labels = create_labels(label_tensor_valid_dict)
test_labels = create_labels(label_tensor_test_dict)

#%%
num_train_examples = train_data.shape[0]
num_output_classes = train_labels.shape[1]
#%%


#GRID SEARCH: neurons, lr, dropout, kernel_regularizer, initializers, batch_size
#
#def create_model():
#        model = Sequential()
#        model.add(Dense(100, input_shape=(max_len, len(alphabet)), kernel_regularizer=regularizers.l2(0.001)))
#        model.add(Dropout(0.85))
#        model.add(LSTM(100, input_shape=(100,))) 
#        model.add(Dropout(0.85))
#        model.add(Dense(num_output_classes, activation='softmax'))
#    
#        model.compile(loss='categorical_crossentropy',
#                  optimizer=adam,
#                  metrics=['accuracy'])	
#        
#        return model
#
##%%
#seed = 7
#np.random.seed(seed)
#
#model = KerasClassifier(build_fn=create_model, epochs=1, verbose=0)
#
#batch_size = [100,200]
#param_grid = dict(batch_size=batch_size)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(train_data, train_labels)
#
##history = grid.fit(train_data, train_labels, validation_data = (valid_data, valid_labels))
#    
##plt.plot(history.history['acc'])
##plt.plot(history.history['val_acc'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'valid'], loc='upper left')
##plt.show()
### summarize history for loss
##plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'valid'], loc='upper left')
##plt.show()
#
#
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
#%%+
model = Sequential()

model.add(Dense(200, input_shape=(max_len, len(alphabet)), kernel_regularizer=regularizers.l2(0.0001)))

model.add(Dropout(0.85))

model.add(LSTM(200, input_shape=(200,))) 

model.add(Dropout(0.8))

model.add(Dense(num_output_classes, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
          epochs=2000,
          batch_size=100,
          validation_data = (valid_data, valid_labels))

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


#score = model.evaluate(test_data, test_labels, batch_size=128)
