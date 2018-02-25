import os
import time
import numpy as np 
from os import listdir
import matplotlib.pyplot as plt 
import random
import unicodedata
import re
import keras
from keras import Sequential
from keras.models import load_model
from keras.layers import RNN, SimpleRNN, LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import to_categorical

from keras import losses

from learner import Learner
import hyperopt

import seaborn as sn
import pandas  as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # omit tensorflow info and warning messages
starttime = time.time() # for debugging/performance monitoring
#%%
#new_model = False # True -> a new model is trained. False -> a previously trained model is loaded and used.
#optimize_hyperparameters = False # True -> activate hyperopt.py. False -> use the given parameters.
#save_new_model = False
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
alphabet_len = len(alphabet)

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
#BATCH_SIZE  = 128 
#DROPOUT = 0.0 
#DROPOUT_RECURRENT = 0.0
#NEURONS_RNN   = 100 
#NEURONS_DENSE = 100
#REGULARIZE  = 0.0000000
#LEARNRATE  = 0.001 
#%%
RNN_TYPE = "Bidirectional_LSTM"

#%%
def train_with_hyperopt(rnn_type):
    
    paramranges = {}
    paramranges["num_epochs"] = 400
    paramranges["batch_size"] = [1, 16, 32, 64, 128, 256, 512]
    paramranges["dropout"] = [0.0,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8]
    paramranges["dropout_recurrent"] = [0.0,0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7,0.8]
    paramranges["neurons_dense"] = [50, 75, 100, 125, 150, 175, 200]
    paramranges["neurons_rnn"] = [50, 75, 100, 125, 150, 175, 200]
    paramranges["regularize_rate"] = [0.0, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001]
    paramranges["learn_rate"] = [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    
    learner = Learner(max_len, alphabet_len, num_categories, train_data, train_labels, valid_data, valid_labels, rnn_type)
    _, best_config = hyperopt.random_search(learner, paramranges, rnn_type=rnn_type)
    
    return best_config
#%%
best_config = train_with_hyperopt(RNN_TYPE) 
#%%
def train_with_best_config(best_config,rnn_type):
    
        num_epochs = best_config['num_epochs']+10
        batch_size = best_config['batch_size']
        dropout = best_config['dropout']
        dropout_recurrent = best_config['dropout_recurrent']
        neurons_dense = best_config['neurons_dense']
        neurons_rnn = best_config['neurons_rnn']
        regularize_rate = best_config['regularize_rate']
        learn_rate = best_config['learn_rate']

        model = Sequential()
        
        model.add(Dense(neurons_dense, input_shape=(max_len, alphabet_len), kernel_regularizer=regularizers.l2(regularize_rate), bias_regularizer= regularizers.l2(regularize_rate)))
       
        if rnn_type == "SimpleRNN":
            model.add(SimpleRNN(neurons_rnn, input_shape=(neurons_dense,), return_sequences=False, return_state=False, dropout=dropout, recurrent_dropout=dropout_recurrent, kernel_regularizer=regularizers.l2(regularize_rate), bias_regularizer= regularizers.l2(regularize_rate), recurrent_regularizer = regularizers.l2(regularize_rate))) 
        if rnn_type == "LSTM":
            model.add(LSTM(neurons_rnn, input_shape=(neurons_dense,), return_sequences=False, return_state=False, dropout=dropout, recurrent_dropout=dropout_recurrent, kernel_regularizer=regularizers.l2(regularize_rate), bias_regularizer= regularizers.l2(regularize_rate), recurrent_regularizer = regularizers.l2(regularize_rate))) 
        if rnn_type == "Bidirectional_LSTM":
            model.add(Bidirectional(LSTM(neurons_rnn, input_shape=(neurons_dense,), return_sequences=False, return_state=False, dropout=dropout, recurrent_dropout=dropout_recurrent, kernel_regularizer=regularizers.l2(regularize_rate), bias_regularizer= regularizers.l2(regularize_rate), recurrent_regularizer = regularizers.l2(regularize_rate)))) 

        model.add(Dense(num_categories, activation='softmax', kernel_regularizer=regularizers.l2(regularize_rate), bias_regularizer= regularizers.l2(regularize_rate), activity_regularizer= regularizers.l2(regularize_rate)))
        adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['categorical_accuracy'])
        
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./GRAPH', histogram_freq=0, write_graph=True, write_images=True)

        history = model.fit(train_data, 
                  train_labels,
                  epochs=num_epochs,
                  batch_size=batch_size,
                  validation_data = (valid_data, valid_labels), 
                  callbacks=[tbCallBack])

        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
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
        plt.savefig('plot_hyperopt_%s.png' %(rnn_type))

        model.save("saved_model_%s" %rnn_type)

#%%
train_with_best_config(best_config, RNN_TYPE)

#%%

model = load_model("/home/janinanu/Language_Identification/saved_model_%s" %(RNN_TYPE))

#%%
def test_scores(rnn_type):

    score = model.evaluate(test_data, test_labels, batch_size=32)
    print("Test loss for %s: %f" %(rnn_type, score[0]))
    print("Test accuracy for %s: %f" %(rnn_type, score[1]))
#%%
test_scores(RNN_TYPE)
#%%
def predict_top_3(rnn_type, input_name):
    
    name = name_to_array(input_name, max_len).reshape(-1, max_len, len(alphabet))
    y_prob = model.predict(name, batch_size=1) 
    top_3_classes = y_prob[0].argsort()[-3:][::-1]    
    print("Top 3 predicted language with %s for %s:" %(rnn_type, input_name))  
    for pred in top_3_classes:
        print("%s (%f)" %(categories_sorted[pred], y_prob[0][pred]))
        
#%%
predict_top_3(RNN_TYPE, "Wirth")     

#%%
def predict_class(test_data):
    
    pred_class = model.predict(test_data, verbose=2)
    pred_ids = np.argmax(pred_class, axis=1)
    
    return pred_class, pred_ids
#%%
pred_class, pred_ids = predict_class(test_data)
            
#%%
def create_confusion_matrix(rnn_type):
    
    confusion_matrix = np.zeros((num_categories, num_categories))
    pred_class, pred_ids = predict_class(test_data)
    true_ids = np.argmax(test_labels,axis=1)
    
    for example in range(len(pred_ids)):
        confusion_matrix[true_ids[example]][pred_ids[example]] +=1
    
    print("Confusion matrix for %s:" %(rnn_type))
    df_cm = pd.DataFrame(confusion_matrix, columns = categories_sorted, index =categories_sorted)
    fig = plt.figure(figsize = (10,7))
    cm = sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})

    return cm
#%%
confusion_matrix = create_confusion_matrix(RNN_TYPE)





#if(optimize_hyperparameters): 
#		learner = Learner(max_len, alphabet_len, num_categories, train_data, train_labels, valid_data, valid_labels)
#		model = hyperopt.random_search(learner, paramranges)[0]
#else: 
#		model.fit(train_set_padded, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dev_set_padded, dev_labels))
#if(save_new_model): model.save(trained_model)
#
#else:
#	model = load_model(trained_model)
#	print("loaded model from {}".format(trained_model))
#
#

