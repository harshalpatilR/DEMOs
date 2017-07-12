
#    
#    This is a demo of text classification using the "bag of words" technique.
#    The classifier used is logistic but the same can be implemented with Support Vector Machine.
#    The code utilizes TensorFlow and will be run on a CPU and a GPU. The resulting training time will be recorded.
#    We would expect a much better traning performance on GPU.
#    @author: Harshal Patil
#    

#     load packages
import os 
import zipfile
import urllib
import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import time 
from matplotlib import pyplot as plt


#     load the data
#     the original file is at https://archive.ics.uci.edu/ml/machine-learning-databases/00228/
#     it is downloaded and saved locally
#     the location is the current working directory which is not changed in this code
cwd = os.getcwd()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
urllib.urlretrieve(url, "smsspamcollection.zip")
dnldf = os.path.join(cwd,"smsspamcollection.zip")
zipfile.ZipFile(dnldf).extractall()

#     open the text file
filelocation = os.path.join(cwd,"smsspamcollection")

#    parse the file
#    first column has label - Spam or Ham and second column has the message
#     read into a pandas dataframe
df = pd.read_csv(filelocation,sep="\t",names=["class","sms"])
df["label"] = np.vectorize(lambda x: 1 if(x=="spam") else 0)(df.loc[:,"class"])

#    check proper coding of label from class
print "1. check counts and percentages \n"
print pd.crosstab(df["label"],df["class"],margins=True)
print "\n"

#    check balance between positive and negative cases
#    check the percentage of positive cases 
print pd.crosstab(df["label"],df["class"],margins=True).apply(lambda x: x/len(df), axis=1)
print "\n"

#    preprocess strings to normalize and improve model performance
#    conver to lowercase, remove punctuations, numbers and extra spaces, tabs etc.
#    No stopwords removal - although it can be tried
message = np.vectorize(lambda x: x.lower())(df["sms"])
message = np.vectorize(lambda x: re.sub("["+string.punctuation+"]"," ",x))(message)
message = np.vectorize(lambda x: re.sub("[0123456789]"," ",x))(message)
message = np.vectorize(lambda x: " ".join(x.split()))(message)


#    we will create a documentxterm matrix which will be passed to learner
#    The counts will be using TF-IDF to differentiate unique terms and add higher weight to them
#    you may need to run nltk.download()
tfidfconvert = TfidfVectorizer(tokenizer=nltk.word_tokenize)
termdoc = tfidfconvert.fit_transform(message)

print "2. Document x Term Matrix size : "+ str(termdoc.shape[0])+" rows by "+ str(termdoc.shape[1])+" columns \n"
print "columns indicate the specific words and as suspected there are too many"
print "so we would reduce the size a bit arbitrarily to top 3000 words"

tfidfconvert = TfidfVectorizer(tokenizer=nltk.word_tokenize,max_features=3000)
termdoc = tfidfconvert.fit_transform(message)
print "3. Document x Term Matrix size : "+ str(termdoc.shape[0])+" rows by "+ str(termdoc.shape[1])+" columns \n"

nrow = termdoc.shape[0]
ncol = termdoc.shape[1]

#    divide in training and test set 
train_set = np.random.choice(nrow,int(round(0.8*nrow)),replace=False)
test_set = np.delete(range(nrow),train_set)

print "4. Training Set Size : " + str(len(train_set))
print "5. Test Set Size : " + str(len(test_set))
print "\n Establish TensorFlow graph and Train the model \n"

# actual training and test data
matrix_train = termdoc[train_set] 
matrix_test = termdoc[test_set] 
target_train = np.transpose([df["label"][train_set]])
target_test = np.transpose([df["label"][test_set]])

#matrix_train.shape
#matrix_test.shape


#    build tensorflow graph
#    We will use logistic model with TensorFlow
#    z=WX+b where W is a weight vector and X is input data matrix and b is the bias 
#    probability = exp(z)/(1+exp(z))
#    we will initialize the weight vector and bias to random normal values

#specify device - we will change this to GPU to check difference

#with tf.device('/gpu:0'):
#with tf.device('/cpu:0'):
W = tf.Variable(tf.random_normal(shape=[ncol,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

#    placeholder for data input - shape is none where we dont know in advance
x_data = tf.placeholder(shape=[None, ncol], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#    Model output and loss function
model_output = tf.add(tf.matmul(x_data, W), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

#    we want high probability for the class=1
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

#    declare which optimizer to use and minimize loss - learning rate is provided
my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)

#    Intitialize Variables
#    session is created to log deployment details

# when GPU is present
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#    training the model - we go through a number of random batches to train
batch=200
rounds = 10000

train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
runtime = []
i=0
time_start = time.time()
for i in range(rounds):
    rand_index = np.random.choice(matrix_train.shape[0], size=batch)
    rand_x = matrix_train[rand_index].todense() 
    #rand_y = np.transpose([target_train[rand_index]])
    rand_y = target_train[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    # Only record loss and accuracy every 100 generations
    if (i+1)%100==0:
        i_data.append(i+1)
        t1 = time.time()
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        t2=time.time()
        train_loss.append(train_loss_temp)
        runtime.append(t2-t1)
        
        t1 = time.time()
        test_loss_temp = sess.run(loss, feed_dict={x_data: matrix_test.todense(), y_target: target_test})
        t2=time.time()
        test_loss.append(test_loss_temp)
        runtime.append(t2-t1)
         
        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)
    
        test_acc_temp = sess.run(accuracy, feed_dict={x_data: matrix_test.todense(), y_target: target_test})
        test_acc.append(test_acc_temp)
    if (i+1)%500==0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x,2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

time_end = time.time()

print "\n 6. Training time in seconds (total) " + str(time_end - time_start)
print "7. Training time in seconds (only for running Loss node in TensorFlow graph) " + str(sum(runtime))

print "\n 8. Loss Function"
plt.plot(i_data,train_loss,label="Training Set")
plt.plot(i_data,test_loss,label="Test Set")
plt.legend()
plt.show()

print "9. Accuracy"
plt.plot(i_data,train_acc,label="Training Set")
plt.plot(i_data,test_acc,label="Test Set")
plt.legend()
plt.show()
