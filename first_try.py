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
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
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
def name_to_tensor(name):
    one_hot_name = to_categorical([alphabet.find(letter) for letter in name], num_classes=len(alphabet))
    num_pads = max_len-len(name)
    one_hot_name_padded = np.vstack((one_hot_name, np.zeros((num_pads,len(alphabet)))))
    return one_hot_name_padded


#%%

def create_name_vector_dict(names_dict): #names_train_dict, names_valid_dict, names_test_dict
    names_vector_dict = {}
    for language, name_list in names_dict.items():
            for name in name_list:  
                names_vector_dict[language] = [name_to_tensor(str(name)) for name in name_list]
   
#    in_dict = str(names_dict)
#    format_add = in_dict.split("_")[0]
#    out_dict = "name_vecs_%s_dict" % (format_add) 
#    
#    out_dict = names_vector_dict

    return names_vector_dict
    #return name_vecs_%s_dict (% names_dict.split("_")[1])
#%%
name_vector_train_dict = create_name_vector_dict(names_train_dict) 
name_vector_valid_dict = create_name_vector_dict(names_valid_dict) 
name_vector_test_dict = create_name_vector_dict(names_test_dict) 
    
#%%
def create_name_tensor_dict(names_vector_dict):
    name_tensor_dict = {}
    for language, vec_list in names_vector_dict.items():
        name_tensor_dict[language] = np.concatenate(vec_list, axis = 0).reshape(-1, max_len, len(alphabet))
    return name_tensor_dict
#%%

name_tensor_train_dict = create_name_tensor_dict(name_vector_train_dict)
name_tensor_valid_dict = create_name_tensor_dict(name_vector_valid_dict)
name_tensor_test_dict = create_name_tensor_dict(name_vector_test_dict)


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





#
#data = tf.placeholder(tf.float32, [1, len(name_tensor), len(alphabet)]) #Batch Size, Sequence Length, Input Dimension
#target = tf.placeholder(tf.float32, [None, len(lang_names_dict)])
##%%
#num_hidden = 10
#cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
#val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
##%%
#val = tf.transpose(val, [1, 0, 2])
#last = tf.gather(val, int(val.get_shape()[0]) - 1)
#
#weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
#bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
##%%
#
#prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
#
#cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
#
##%%
#optimizer = tf.train.AdamOptimizer()
#minimize = optimizer.minimize(cross_entropy)
##%%
#init_op = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init_op)
##%%
#batch_size = 1
#no_of_batches = int(len(data)/batch_size)
#epoch = 5000
#for i in range(epoch):
#    ptr = 0
#    for j in range(no_of_batches):
#        inp, out = data[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
#        ptr+=batch_size
#        sess.run(minimize,{data: inp, target: out})
#    print("Epoch - ",str(i))
#incorrect = sess.run(error,{data: test_input, target: test_output})
#print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
#sess.close()
#
#
##
