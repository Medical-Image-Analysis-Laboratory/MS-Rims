from config import *

import tensorflow as tf

from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d


def get_model_graph(patch_size, *, n_channels = 2, weight_ce=[1., 1.]):
    width, height, depth = patch_size

    with tf.name_scope('inputs'):
        # x: place holder for the input image.
        # y: place holder for the labels.
        # lr : place holder for learning rate. to change the learning rate as we move forward.
        x = tf.placeholder(tf.float32, [None, width, height, depth, n_channels])
        y = tf.placeholder(tf.float32, [None, N_CLASSES])
        lr = tf.placeholder(tf.float32)
        
    main, attention = tf.split(x, [1, 1], 4)
    
    # Flair_masked layers
    main_conv1 = conv_3d(main, 32, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    main_conv1 = conv_3d(main_conv1, 32, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    main_pool1 = max_pool_3d(main_conv1, 2)
    # Layer #2
    main_conv2 = conv_3d(main_pool1, 64, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    main_conv2 = conv_3d(main_conv2, 64, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    main_pool2 = max_pool_3d(main_conv2, 2)
    # Layer #3
    main_conv3 = conv_3d(main_pool2, 128, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    main_conv3 = conv_3d(main_conv3, 128, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    main_pool3 = max_pool_3d(main_conv3, 2)
    # Dense layer
    main_pred_conv = conv_3d(main_pool3, 1, 1, activation='linear', padding='valid')
        
    # T2 layers
    # Layer #1
    att_conv1 = conv_3d(attention, 32, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    att_conv1 = conv_3d(att_conv1, 32, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    att_pool1 = max_pool_3d(att_conv1, 2)
    # Layer #2
    att_conv2 = conv_3d(att_pool1, 64, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    att_conv2 = conv_3d(att_conv2, 64, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    att_pool2 = max_pool_3d(att_conv2, 2)
    # Layer #3
    att_conv3 = conv_3d(att_pool2, 128, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    att_conv3 = conv_3d(att_conv3, 128, 3, activation='tanh', weights_init='xavier', padding='same', regularizer="L2")
    att_pool3 = max_pool_3d(att_conv3, 2)
    # Dense layer
    att_pred_conv = conv_3d(att_pool3, 1, 1, activation='linear', padding='valid')

    concat = tf.concat([main_pred_conv, att_pred_conv], 4)
    fully1 = fully_connected(concat, 256,
                             activation='linear', weights_init='truncated_normal')
    fully2 = fully_connected(fully1, 128,
                             activation='linear', weights_init='truncated_normal')
    fully3 = fully_connected(fully2, 64,
                             activation='linear', weights_init='truncated_normal')

    pred = fully_connected(fully3, 2,
                           activation='linear')


    # We establish the cost function
    with tf.name_scope('cost'):
        # your class weights
        class_weights = tf.constant([weight_ce])
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * y, axis=1)
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        cost = tf.reduce_mean(weighted_losses)

        #error = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
        #error = tf.nn.weighted_cross_entropy_with_logits(targets = y , logits = pred, pos_weight =
                                                                 #tf.constant(weight_ce))
        #cost = tf.reduce_mean(error)
        #tf.summary.scalar('cost', cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, -1), tf.argmax(y, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    
    eval_dict = {
        "accuracy": accuracy,
    }
    
    pred = tf.nn.softmax(pred)
    return x, y, lr, eval_dict, cost, optimizer, pred