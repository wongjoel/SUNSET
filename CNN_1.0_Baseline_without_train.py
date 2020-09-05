
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import datetime
from modelutility_v2 import *


# define folder location
model_name = 'CNN_1.0_Baseline'
dir_path = "/flush5/won10v/SUNSET"
data_folder = os.path.join(dir_path, "data_pred_expanded","frequency_1")
output_folder = os.path.join('models',model_name)

#define file location
image_log_trainval_path = os.path.join(data_folder,'image_log_trainval.npy')
pv_log_trainval_path = os.path.join(data_folder,'pv_log_trainval.npy')
pv_pred_trainval_path = os.path.join(data_folder,'pv_pred_trainval.npy')
times_trainval_path = os.path.join(data_folder,'times_trainval.npy')

image_log_test_path = os.path.join(data_folder,'image_log_test.npy')
pv_log_test_path = os.path.join(data_folder,'pv_log_test.npy')
pv_pred_test_path = os.path.join(data_folder,'pv_pred_test.npy')
times_test_path = os.path.join(data_folder,'times_test.npy')

first_cloudy_test = datetime.datetime(2017,3,15)

# define model characteristics
width = 24
filter_size = [3,3]
dense_size = 1024
drop_rate = 0.4

# define training time parameters
num_epochs = 100
num_rep = 10
plotting = True
learning_rate = 3e-6
batch_size = 256


# In[2]:


# The model
def cnn_33_model(X,X2, y, is_training):
    # CBP sandwich 1
    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=width,
        kernel_size=filter_size,
        padding="same",
        activation=tf.nn.relu)
    bn1 = tf.layers.batch_normalization(inputs=conv1, axis= -1, training = is_training)
    pool1 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2, 2], strides=2)

    # CBP sandwich 2
    conv3 = tf.layers.conv2d(
        inputs=pool1,
        filters=width*2,
        kernel_size=filter_size,
        padding="same",
        activation=tf.nn.relu)
    bn3 = tf.layers.batch_normalization(inputs=conv3, axis= -1,training = is_training)
    pool2 = tf.layers.max_pooling2d(inputs=bn3, pool_size=[2, 2], strides=2)

    down_rate = 4
        
    # Two fully connected nets
    pool2_flat = tf.reshape(pool2, [-1, int(side_len/down_rate*side_len/down_rate*width*2)])
    pool2_flat_aug = tf.concat([pool2_flat,X2],axis=1)
    
    dense1 = tf.layers.dense(inputs=pool2_flat_aug, units=dense_size, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=drop_rate, training = is_training)
    dense2 = tf.layers.dense(inputs=dropout1, units=dense_size, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=drop_rate, training = is_training)
    
    regression = tf.layers.dense(inputs=dropout2, units=1)
    regression = tf.reshape(regression, [-1])
    return regression


# In[3]:


# load PV output and images for the trainval set
pv_log_trainval = np.load(pv_log_trainval_path)
images_trainval = np.load(image_log_trainval_path)
pv_pred_trainval = np.load(pv_pred_trainval_path)
times_trainval = np.load(times_trainval_path)

# stack up the history and colors into a unified channel
images_trainval = images_trainval.transpose((0,2,3,4,1))
images_trainval = images_trainval.reshape((images_trainval.shape[0],
                                           images_trainval.shape[1],images_trainval.shape[2],-1))

# shuffling of the data by day
shuffled_indices = day_block_shuffle(times_trainval)
pv_log_trainval = pv_log_trainval[shuffled_indices]
pv_pred_trainval = pv_pred_trainval[shuffled_indices]
times_trainval = times_trainval[shuffled_indices]
images_trainval = images_trainval[shuffled_indices]

# Input dimension is used to construct the model
side_len = images_trainval.shape[1]
image_input_dim = [side_len,side_len,images_trainval.shape[3]]
print(f"side_len = {side_len}")
print(f"image_input_dim = {image_input_dim}")

# In[4]:


# Build computational graph
tf.reset_default_graph()  # Reset computational graph

# x,y, pred_y 
x_var = tf.placeholder(tf.float32, [None, side_len, side_len, images_trainval.shape[3]])  # x variable
x2_var = tf.placeholder(tf.float32, [None, pv_log_trainval.shape[1]])
y_var = tf.placeholder(tf.float32, [None])  # y variable
is_training = tf.placeholder(tf.bool) # a flag

pred_y_var = cnn_33_model(x_var,x2_var, y_var, is_training)  # model in use

# loss and other accuracy indicator
loss_var = tf.losses.mean_squared_error(y_var, pred_y_var)  # loss in use
y_var_std = tf.reduce_mean(tf.square(y_var-tf.reduce_mean(y_var)))
rel_err_var = tf.sqrt(tf.divide(loss_var, y_var_std))

# Define optimizer and optimize session parameter
optimizer = tf.train.AdamOptimizer(learning_rate)

# batch normalization in tensorflow requires this extra dependency
# update_ops consist of calculating running average
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# make sure every time during training the moving average is updated
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(loss_var)
pass


# # In[ ]:


# # Create a list of variables and a list of data. They must have strict correspondence
# Xy_var = [x_var, x2_var, y_var]
# data_trainval = [images_trainval,pv_log_trainval,pv_pred_trainval]

# # Run training process
# save_path,training_history = run_training(num_rep, num_epochs, plotting, output_folder, model_name,"/gpu:0",
#                                           Xy_var, is_training, loss_var, rel_err_var, train_step, data_trainval, 
#                                           batch_size = batch_size)
                                          

# # unpack the training history cache    
# train_loss_hist,train_error_hist,val_loss_hist,val_error_hist = training_history

# np.save(os.path.join(output_folder, "train_loss.npy"), train_loss_hist)
# np.save(os.path.join(output_folder, "train_error.npy"), train_error_hist)
# np.save(os.path.join(output_folder, "val_loss.npy"), val_loss_hist)
# np.save(os.path.join(output_folder, "val_error.npy"), val_error_hist)


# ## Comparing mean validation loss

# In[ ]:


def find_best_loss(output_folder):
    # Restore error history
    train_loss = np.load(os.path.join(output_folder,"train_loss.npy"))
    val_loss = np.load(os.path.join(output_folder,"val_loss.npy"))
    
    num_rep = val_loss.shape[0]
    
    # Only obtain results for best model
    best_idx = np.zeros(num_rep,dtype =int)
    for i in range(num_rep):
        best_idx[i] = np.argmin(val_loss[i][val_loss[i]>0])

    train_best = train_loss[np.arange(num_rep),best_idx] 
    val_best = val_loss[np.arange(num_rep),best_idx]
    
    return [train_best,val_best]


# In[ ]:

# 2020-09-04 Joel Wong - This cell seems to be related to outputting clear sky index instead of PV power. I think it's irrelevant to what I currently need.

# ## Find the persistence MSE
# from Relative_op_func2 import Relative_output

# # calculate the kt of the current time
# kt_trainval,pv_log_trainval_theo = Relative_output(times_trainval, pv_log_trainval[:,0]) 
# # kt = pv_log_test[:,0]/pv_log_test_theo

# # calculate the P_theo in the future time
# _,pv_pred_trainval_theo = Relative_output(times_trainval + datetime.timedelta(minutes = 15), pv_pred_trainval)

# # forecast with kt_persistence
# pv_pred_trainval_ktpers = kt_trainval*pv_pred_trainval_theo

# # Find the MSE of the persistence model on trainval set
# ktpers_mse_trainval = np.mean(np.square(pv_pred_trainval - pv_pred_trainval_ktpers))
# print('kt-Persistence MSE of the trainval set: {0:.2f}'.format(ktpers_mse_trainval))


# In[ ]:


# Compare the training and validation loss boxchart with baseline model

baseline_model_name = 'CNN_1.0_Baseline'
output_folder_baseline = os.path.join('models',baseline_model_name)

best_loss_baseline = find_best_loss(output_folder_baseline)
best_loss_curr = find_best_loss(output_folder)


# # Compare the rRMSE between this model and baseline model
# f,axarr = plt.subplots(1,2,sharey = True)
# set_name = ['Training','Validation']
# model_label = ['Baseline', model_name]

# for i in range(len(axarr)):
#     best_loss = np.vstack((best_loss_baseline[i],best_loss_curr[i]))
#     axarr[i].boxplot(best_loss.T, labels = model_label,widths = 0.7)
    
#     axarr[i].set_title(set_name[i])
#     axarr[i].grid(False)
#     axarr[i].axhline(y=ktpers_mse_trainval,color= 'r',linewidth =1.25, alpha = 0.9, label = 'Persistence')
    
#     # Plot the mean loss
#     mean_loss = np.mean(best_loss,axis = 1)
#     axarr[i].scatter(np.arange(1,len(model_label)+1), mean_loss, marker = 'D', label = 'Mean Loss')
#     axarr[i].legend(loc = 'upper right')

# axarr[0].set_ylabel('MSE $(\mathrm{kW}^2)$')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

# plt.ylim(0,10)
# plt.savefig(os.path.join(output_folder,'compare_to_baseline.pdf'),bbox_inches = 'tight')
# plt.show()


# ## Deploying the model to the test set

# In[ ]:


# load PV output and images for the test
pv_log_test = np.load(pv_log_test_path)
images_test = np.load(image_log_test_path)
pv_pred_test = np.load(pv_pred_test_path)
times_test = np.load(os.path.join(times_test_path))

# stack up the log and color channels into a unified dimension
images_test = images_test.transpose((0,2,3,4,1))
images_test = images_test.reshape((images_test.shape[0],images_test.shape[1],images_test.shape[2],-1))

# Input dimension is used to construct the model
side_len = images_test.shape[1]
image_input_dim = [side_len,side_len,images_test.shape[3]]


# In[ ]:


# define variable and test data list [note the list here include training flag]
Xy_var_test = [x_var, x2_var, y_var]
data_test = [images_test,pv_log_test,pv_pred_test]

tic = time.process_time()

pred_y_value = inference_multirep(num_rep, model_name,output_folder,'/cpu:0', 
                                  Xy_var_test, is_training, pred_y_var, data_test)

toc = time.process_time() - tic

# Save the inference result
np.save(os.path.join('models',model_name,'pv_pred_test_modeled.npy'),np.mean(pred_y_value,axis = 0))

#Find the index of the first rainy day.
first_cloudy_idx = find_idx_with_dates(times_test, [first_cloudy_test])[0]

# Calculate bucket and single model MSE
bucket_mse_sunny = np.mean(np.square(np.mean(pred_y_value,axis=0)-pv_pred_test)[:first_cloudy_idx])
bucket_mse_cloudy = np.mean(np.square(np.mean(pred_y_value,axis=0)-pv_pred_test)[first_cloudy_idx:])
single_mse_sunny = np.mean(np.square(pred_y_value - pv_pred_test)[:first_cloudy_idx],axis=1)
single_mse_cloudy = np.mean(np.square(pred_y_value - pv_pred_test)[first_cloudy_idx:],axis=1)

# Print out the inference time
print('inference time per sample: {0:.4f}'.format(toc/10/images_test.shape[0]))

# Print out the sunny and cloudy test set forecast skills
print('test set sunny MSE is {0:.2f}, while cloudy MSE if {1:.2f}'.format(bucket_mse_sunny,bucket_mse_cloudy))

