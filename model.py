import numpy as np
import cv2
import tensorflow as tf
import vgg19
import cnnm
import time
import os

vdir = "data/"
optical_flow_dir = vdir+"optical-flow/"
res_dir = vdir+"resized/"

trainlist = "trainlist01.txt"
if not os.path.isfile(trainlist):
    print("Training set description not found!")
    exit()
trainlist = open(trainlist, "r").readlines()
training_set_length = len(trainlist)
training_set_offset = 0

def get_video(file, color=True):
    cap = cv2.VideoCapture(file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if color:
        frames = np.zeros((num_frames, 224, 224, 3))
    else:
        frames = np.zeros((num_frames, 224, 224))

    # load video into numpy array in the following format:
    # ar = [num_frames, 224, 224, 3]
    k=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames[k] = frame
        k = k+1

    return num_frames, frames

def get_data(L):
    global training_set_offset, training_set_length

    file, label = trainlist[training_set_offset].split()

    # if the file listed in training set doesn't exist, remove it from training set and continue on to next one
    # mostly irrelevant for full-blown training but helpful for training on small subset of training set
    while not (os.path.isfile(res_dir+file) and os.path.isfile(optical_flow_dir+file)):
        del trainlist[training_set_offset]
        training_set_length = training_set_length - 1
        if training_set_offset >=- training_set_length:
            training_set_offset = 0
        file, label = trainlist[training_set_offset].split()

    training_set_offset += 1
    if training_set_offset >= training_set_length :
        training_set_offset = 0

    # read spatial video
    _, spatial_frames = get_video(res_dir+file, color=True)
    
    # read optical flow video
    num_motion_frames, motion_frames = get_video(optical_flow_dir+file, color=False)

    # stack optical flow frames efficiently using numpy's stride_tricks method
    sizeof_int32 = np.dtype(np.int32).itemsize

    # stride by 1 frame with each time window and stride by 1 frame within frames in each time window
    # let A, B, C... be consequent frames and L=3
    # transformation can be represented as [A, B, C, ...] --> [[A, B, C], [B, C, D], [C, D, E], ...]
    stacked_motion_frames = np.lib.stride_tricks.as_strided(motion_frames, (num_motion_frames-L+1, L, 224, 224),
                                        (sizeof_int32*224*224, sizeof_int32*224*224, sizeof_int32*224, sizeof_int32))
    stacked_motion_frames = np.reshape(stacked_motion_frames, (num_motion_frames-L+1, 224, 224, L))

    return spatial_frames, stacked_motion_frames, label

def initialize_fc(name, shape, mean=0.0, dev=1e-3, scope=None):
    assert len(shape) == 2
    weight = tf.get_variable(name+"_weights", shape, dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
    bias = tf.get_variable(name+"_biases", shape[-1], dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
    return weight, bias

def fc_layer(name, prev, shape, gate="relu", mean=0.0, dev=1e-3):
    with tf.variable_scope(name):
        weights, biases = initialize_fc(name, shape, mean, dev)
        output = tf.matmul(prev, weights)+biases

        if gate == "relu":
            output = tf.nn.relu(output)
        elif gate == "tanh":
            output = tf.tanh(output)
        elif gate == "sigmoid":
            output = tf.sigmoid(output)

        return output

# L : height of stacked optical flows as input to temporal learning CNNs
L = 10
# C : number of classes
C = 101
# learning rates for optimizers
lr_proximal_gradient = 0.001
lr_gradient = 0.001
# lmbd1, lmbd2 : l1, l2 norm reg. constants for proximal gradient respectively
lmbd1 = lmbd2 = 1e-5
# number of steps to train
num_steps = 100000

# create and train the model
with tf.Session() as sess:
    spatial_video = tf.placeholder(tf.float32, [None, 224, 224, 3])
    stacked_flow = tf.placeholder(tf.float32, [None, 224, 224, L])
    labels = tf.placeholder(tf.int32, [1])

    num_spatial_frames = tf.shape(spatial_video)[0]
    num_flow_stacks = tf.shape(stacked_flow)[0]

    # build VGG19
    vgg = vgg19.Vgg19("vgg19.npy")
    vgg.build(spatial_video)
    vgg_fc = tf.reshape(vgg.relu6, [1, num_spatial_frames, 4096])

    # build spatial LSTM network
    with tf.variable_scope("spatial_lstm"):
        lstm_stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(1024, state_is_tuple=True),
                                                  tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)], state_is_tuple=True)
        spatial_lstm = tf.nn.dynamic_rnn(lstm_stack, vgg_fc, dtype=tf.float32, time_major=False)
        # method 1: aggregate all time frames
        # spatial_features = tf.reduce_sum(spatial_lstm[0][0], axis=0, keep_dims=True)
        # method 2: use the last time frame features as video level features
        spatial_features = tf.reshape(spatial_lstm[0][0][-1], [1, 512])

    # build CnnM
    cnnm2048 = cnnm.CnnM(L)
    cnnm2048.build(stacked_flow)
    cnnm_fc = cnnm2048.relu1
    cnnm_fc = tf.reshape(cnnm_fc, [1, num_flow_stacks, 4096])

    # build motion LSTM network
    with tf.variable_scope("motion_lstm"):
        lstm_stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(1024, state_is_tuple=True),
                                                  tf.contrib.rnn.BasicLSTMCell(512, state_is_tuple=True)], state_is_tuple=True)
        motion_lstm = tf.nn.dynamic_rnn(lstm_stack, cnnm_fc, dtype=tf.float32, time_major=False)
        #motion_features = tf.reduce_sum(motion_lstm[0][0], axis=0, keep_dims=True)
        motion_features = tf.reshape(motion_lstm[0][0][-1], [1, 512])

    # aggregate frame level CNN outputs to get video level CNN features
    with tf.variable_scope("regularized_fusion_network"):
        spatial_video_level_features = tf.reduce_mean(vgg.relu6, axis=0, keep_dims=True)
        motion_video_level_features = tf.reduce_mean(cnnm2048.relu1, axis=0, keep_dims=True)

    fc_spatial = fc_layer("fc_spatial", spatial_video_level_features, (4096, 200), "relu", 0.0, 0.001)
    fc_motion = fc_layer("fc_motion", motion_video_level_features, (4096, 200), "relu", 0.0, 0.001)

    with tf.variable_scope("regularized_fusion_network"):
        fusion_layer_input = fc_spatial + fc_motion
    # regularized fusion layer
    fusion_layer = fc_layer("fusion_layer", fusion_layer_input, (200, 200), "relu", 0.0, 0.001)

    ############################################

    # we can calculate prediction scores of temporal model and regularized fusion network separetely
    # and then combine them, but in order to work easier with tensorflow,
    # I'm going to concatenate all feature vectors and have one softmax layer
    # note that, since my proposed method is a generalization of the first method, 
    # (when some of the weights are zero)
    # it will converge to a good solution, altough, there may be a performance hit,
    # due to the computational cost of training one large layer compared to two smaller ones
    
    ############################################

    # spatial_lstm_pred = fc_layer("fusion", spatial_features, (512, C), "linear", 0.0, 0.001)
    # motion_lstm_pred = fc_layer("fusion", motion_features, (512, C), "linear", 0.0, 0.001)
    # fusion_layer_pred = fc_layer("softmax", fusion, (200, C), "linear", 0.0, 0.001)
    # to follow the paper verbatim, feed these three layers into softmax function 
    # and linearly combine their outputs to get the final prediction scores

    logits = tf.concat([spatial_features, motion_features, fusion_layer], axis=1)
    logits = fc_layer("softmax", logits, (1224, C), "linear", 0.0, 0.001)

    # define loss function
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    # gradient optimizer on global variables
    gradient_optimizer = tf.train.GradientDescentOptimizer(lr_gradient)
    gd_opt = gradient_optimizer.minimize(loss)

    # proximal gradient optimizer on fusion layer weights as defined in original paper
    proximal_gradient_optimizer = tf.train.ProximalGradientDescentOptimizer(lr_proximal_gradient, lmbd1, lmbd2)
    with tf.variable_scope("", reuse=True):
        fusion_layer_weights = tf.get_variable("fusion_layer/fusion_layer_weights")
        fusion_layer_biases = tf.get_variable("fusion_layer/fusion_layer_biases")
        p_grads = proximal_gradient_optimizer.compute_gradients(loss, var_list=[fusion_layer_weights, fusion_layer_biases])
        pgd_opt = proximal_gradient_optimizer.apply_gradients(p_grads, global_step=None)

    # to conserve memeory, pre-trained Vgg19 weights are fed into the model using placeholders
    # This reduces the amount of memory needed in this step to about a third
    feed_dict = {}
    feed_dict.update(vgg.var_dict)
    sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
    
    feed_dict[spatial_video] = feed_dict[stacked_flow] = feed_dict[labels] = None

    for i in range(num_steps):
        spatial_frames, stacked_motion_frames, label = get_data(L)
        feed_dict[spatial_video] = spatial_frames
        feed_dict[stacked_flow] = stacked_motion_frames
        feed_dict[labels] = np.array([label])
        _, _, l = sess.run([gd_opt, pgd_opt, loss], feed_dict=feed_dict)
        print(l)
