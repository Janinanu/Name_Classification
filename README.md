# Person Name Classification | Keras | RNNs

<p align="center">
<img src="https://github.com/Janinanu/Name_Classification/blob/master/src/cm_more_lstm.png" width="600" height="480" />
</p>

The goal of this project was to develop two RNN models (Simple RNN & LSTM) in Keras to identify the language of origin of person names and compare the performance of both models.
This task becomes relevant in text-to-speech synthesis applications. Only if the system can identify the language of a name, it can choose the proper pronunciation for it. 
E.g. a Russian name appearing within a Spanish language application might then be identified and pronounced in the correct Russian way.

I took inspiration from the following github repo: https://github.com/spro/practical-pytorch 
However, I transfered the model from PyTorch to Keras and modify it in many parts.

Originally, the data set used was taken from: https://github.com/spro/practical-pytorch/tree/master/data/names.
After examining the training curves and testing results on this rather small data set, however, it became clear that the model was stuck in a heavy overfitting situation despite using common regularization techniques such as dropout, L2 weight decay and reducing the model size. In another attempt to regularize the model, more names were collected manually from online pages to form a much larger data set which can be found in the following GDrive folder: https://drive.google.com/drive/folders/1kHCz8UrdprEHddxoc0gqWt5aehPpQssl?usp=sharing

For more details on the theoretical foundations of RNNs and on the models, feel free to take a look into the paper I wrote for this project: https://drive.google.com/file/d/1bix25CE1ox72mwtcm6VFIo4OEDJMwdbn/view?usp=sharing

Certainly one of the most difficult and time-consuming parts while developing neural networks is searching for the optimal hyperparameter configuration. For this project, the optimization process was automated with two additional scripts (“hyperopt.py” & “learner.py”). For each hyperparameter, it takes a range of manually defined candidate values and then runs a multitude of training sessions, randomly trying different combinations of the candidate values and storing the best configuration, i.e. the one that minimizes the loss. 


