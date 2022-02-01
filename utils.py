import numpy as np
import math as m
import os
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from functools import reduce
import pandas as pd
import random
from scipy.signal import butter,filtfilt

def read_file(file):
    return pd.read_csv(file, delimiter='\t', skiprows=[0])

def get_paths(file_dir):
    res = []
    
    def helper(path):
        list_dir = os.listdir(path)
        for f in list_dir:
            f_path = os.path.join(path, f)
            if os.path.isdir(f_path):
                helper(f_path)
            else:
                res.append(f_path)
    helper(file_dir)
                
    return res

def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r     tant^(-1)(y/x)
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian 
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates    [x, y, z]
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def load_data(data_file, classification=True):
    """                                               
    Loads the data from MAT file. MAT file should contain two
    variables. 'featMat' which contains the feature matrix in the
    shape of [samples, features] and 'labels' which contains the output
    labels as a vector. Label numbers are assumed to start from 1.
    Parameters
    ----------
    data_file: str
                        # load data from .mat [samples, (features:labels)]
    Returns 
    -------
    data: array_like
    """
    print("Loading data from %s" % (data_file))
    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)
    print("Data loading complete. Shape is %r" % (dataMat['features'].shape,))
    if classification:
        return dataMat['features'][:, :-1], dataMat['features'][:, -1] - 1
    else:
        return dataMat['features'][:, :-1], dataMat['features'][:, -1]


def reformatInput(data, labels, indices):
    """
    Receives the indices for train and test datasets.
    param indices: tuple of (train, test) index numbers
    Outputs the train, validation, and test data and label datasets.
    """
    np.random.shuffle(indices[0])
    np.random.shuffle(indices[0])
    trainIndices = indices[0][len(indices[1]):]
    validIndices = indices[0][:len(indices[1])]
    testIndices = indices[1]

    if data.ndim == 4:
        return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Iterates over the samples returing batches of size batchsize.
    :param inputs: input data array. It should be a 4D numpy array for images [n_samples, n_colors, W, H] and 5D numpy
                    array if working with sequence of images [n_timewindows, n_samples, n_colors, W, H].
    :param targets: vector of target labels.
    :param batchsize: Batch size
    :param shuffle: Flag whether to shuffle the samples before iterating or not.
    :return: images and labels for a batch
    """
    if inputs.ndim == 4:
        input_len = inputs.shape[0]
    elif inputs.ndim == 5:
        input_len = inputs.shape[1]
    assert input_len == len(targets)

    if shuffle:
        indices = np.arange(input_len)  
        np.random.shuffle(indices) 
    for start_idx in range(0, input_len, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if inputs.ndim == 4:
            yield inputs[excerpt], targets[excerpt]
        elif inputs.ndim == 5:
            yield inputs[:, excerpt], targets[excerpt]


def gen_images(locs, features, n_gridpoints=32, normalize=True, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])  # features.shape为[samples, 3*nElectrodes]

    nSamples = features.shape[0]    # sample number 2670
    # Interpolate the values        # print(np.mgrid[-1:1:5j]) get [-1.  -0.5  0.   0.5  1. ]
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))

    
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),    # cubic
                                    method='cubic', fill_value=np.nan)
    
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        
        temp_interp[c] = np.nan_to_num(temp_interp[c])
        
    temp_interp = np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H] # WH xy
    temp_interp = np.swapaxes(temp_interp, 1, 2)
    temp_interp = np.swapaxes(temp_interp, 2, 3)    # [samples, W, H，colors]
    return temp_interp



def load_or_generate_images(file_path, average_image=3):
    """
    Generates EEG images
    :param average_image: average_image 1 for CNN model only, 2 for multi-frame model 
                        sucn as lstm, 3 for both.
    :return:            Tensor of size [window_size, samples, W, H, channel] containing generated
                        images.
    """
    print('-'*100)
    print('Loading original data...')
    locs = scipy.io.loadmat('../SampleData/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    # Class labels should start from 0
    feats, labels = load_data('../SampleData/FeatureMat_timeWin.mat')   # 2670*1344 和 2670*1
    

    if average_image == 1:   # for CNN only
        if os.path.exists(file_path + 'images_average.mat'):
            images_average = scipy.io.loadmat(file_path + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp  = feats[:, i*192:(i+1)*192]    # each window contains 64*3=192 data
                else:
                    temp += feats[:, i*192:(i+1)*192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
            scipy.io.savemat( file_path+'images_average.mat', {'images_average':images_average})
            print('Saving images_average done!')
        
        del feats
        images_average = images_average[np.newaxis,:]
        print('The shape of images_average.shape', images_average.shape)
        return images_average, labels
    
    elif average_image == 2:    # for mulit-frame model such as LSTM
        if os.path.exists(file_path + 'images_timewin.mat'):
            images_timewin = scipy.io.loadmat(file_path + 'images_timewin.mat')['images_timewin']
            print('\n')    
            print('Load images_timewin done!')
        else:
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(
                    np.array(locs_2d),
                    feats[:, i*192:(i+1)*192], 32, normalize=False) for i in range(feats.shape[1]//192)
                ])
            scipy.io.savemat(file_path + 'images_timewin.mat', {'images_timewin':images_timewin})
            print('Saving images for all time windows done!')
        
        del feats
        print('The shape of images_timewin is', images_timewin.shape)   # (7, 2670, 32, 32, 3)
        return images_timewin, labels
    
    else:
        if os.path.exists(file_path + 'images_average.mat'):
            images_average = scipy.io.loadmat(file_path + 'images_average.mat')['images_average']
            print('\n')
            print('Load images_average done!')
        else:
            print('\n')
            print('Generating average images over time windows...')
            # Find the average response over time windows
            for i in range(7):
                if i == 0:
                    temp = feats[:, i*192:(i+1)*192]
                else:
                    temp += feats[:, i*192:(i+1)*192]
            av_feats = temp / 7
            images_average = gen_images(np.array(locs_2d), av_feats, 32, normalize=False)
            scipy.io.savemat( file_path+'images_average.mat', {'images_average':images_average})
            print('Saving images_average done!')

        if os.path.exists(file_path + 'images_timewin.mat'):
            images_timewin = scipy.io.loadmat(file_path + 'images_timewin.mat')['images_timewin']
            print('\n')    
            print('Load images_timewin done!')
        else:
            print('\n')
            print('Generating images for all time windows...')
            images_timewin = np.array([
                gen_images(
                    np.array(locs_2d),
                    feats[:, i*192:(i+1)*192], 32, normalize=False) for i in range(feats.shape[1]//192)
                ])
            scipy.io.savemat(file_path + 'images_timewin.mat', {'images_timewin':images_timewin})
            print('Saving images for all time windows done!')

        del feats
        images_average = images_average[np.newaxis,:]
        print('The shape of labels.shape', labels.shape)
        print('The shape of images_average.shape', images_average.shape)    # (1, 2670, 32, 32, 3)
        print('The shape of images_timewin is', images_timewin.shape)   # (7, 2670, 32, 32, 3)
        return images_average, images_timewin, labels
    
    
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy
# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts

def get_params(sess):
    variables = tf.trainable_variables()
    params = {}
    for i in range(len(variables)):
        name = variables[i].name
        params[name] = sess.run(variables[i])
    return params
    
    
def to_one_hot(x, N = -1):
    x = x.astype('int32')
    if np.min(x) !=0 and N == -1:
        x = x - np.min(x)
    x = x.reshape(-1)
    if N == -1:
        N = np.max(x) + 1
    label = np.zeros((x.shape[0],N))
    idx = range(x.shape[0])
    label[idx,x] = 1
    return label.astype('float32')
    
def image_mean(x):
    x_mean = x.mean((0, 1, 2))
    return x_mean

def shape(tensor):
    """
    Get the shape of a tensor. This is a compile-time operation,
    meaning that it runs when building the graph, not running it.
    This means that it cannot know the shape of any placeholders
    or variables with shape determined by feed_dict.
    """
    return tuple([d for d in tensor.get_shape()])


def fully_connected_layer(in_tensor, out_units, w_name, b_name):
    """
    Add a fully connected layer to the default graph, taking as input `in_tensor`, and
    creating a hidden layer of `out_units` neurons. This should be done in a new variable
    scope. Creates variables W and b, and computes activation_function(in * W + b).
    """
    _, num_features = shape(in_tensor)
    weights = tf.get_variable(name = w_name, shape = [num_features, out_units], initializer = tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable( name = b_name, shape = [out_units], initializer=tf.constant_initializer(0.1))
    return tf.matmul(in_tensor, weights) + biases


def conv2d(in_tensor, filter_shape, out_channels):
    """
    Creates a conv2d layer. The input image (whish should already be shaped like an image,
    a 4D tensor [N, W, H, C]) is convolved with `out_channels` filters, each with shape
    `filter_shape` (a width and height). The ReLU activation function is used on the
    output of the convolution.
    """
    _, _, _, channels = shape(in_tensor)
    W_shape = filter_shape + [channels, out_channels]

    # create variables
    weights = tf.get_variable(name = "weights", shape = W_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable(name = "biases", shape = [out_channels], initializer= tf.constant_initializer(0.1))
    conv = tf.nn.conv2d( in_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
    h_conv = conv + biases
    return h_conv


#def conv1d(in_tensor, filter_shape, out_channels):
#    _, _, channels = shape(in_tensor)
#    W_shape = [filter_shape, channels, out_channels]
#    
#    W = tf.truncated_normal(W_shape, dtype = tf.float32, stddev = 0.1)
#    weights = tf.Variable(W, name = "weights")
#    b = tf.truncated_normal([out_channels], dtype = tf.float32, stddev = 0.1)
#    biases = tf.Variable(b, name = "biases")
#    conv = tf.nn.conv1d(in_tensor, weights, stride=1, padding='SAME')
#    h_conv = conv + biases
#    return h_conv

def vars_from_scopes(scopes):
    """
    Returns list of all variables from all listed scopes. Operates within the current scope,
    so if current scope is "scope1", then passing in ["weights", "biases"] will find
    all variables in scopes "scope1/weights" and "scope1/biases".
    """
    current_scope = tf.get_variable_scope().name
    #print(current_scope)
    if current_scope != '':
        scopes = [current_scope + '/' + scope for scope in scopes]
    var = []
    for scope in scopes:
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            var.append(v)
    return var

def tfvar2str(tf_vars):
    names = []
    for i in range(len(tf_vars)):
        names.append(tf_vars[i].name)
    return names


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
        
    
    

def predictor_accuracy(predictions, labels):
    """
    Returns a number in [0, 1] indicating the percentage of `labels` predicted
    correctly (i.e., assigned max logit) by `predictions`.
    """
    return tf.reduce_mean(tf.cast(predictions>0.5, tf.float32)==labels)

def dic2list(sources, targets):
    names_dic = {}
    for key in sources:
        names_dic[sources[key]] = key
    for key in targets:
        names_dic[targets[key]] = key
    names = []
    for i in range(len(names_dic)):
        names.append(names_dic[i])
    return names

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def norm_matrix(X, l):
    Y = np.zeros(X.shape);
    for i in range(X.shape[0]):
        Y[i] = X[i]/np.linalg.norm(X[i],l)
    return Y


def description(sources, targets):
    source_names = sources.keys()
    target_names = targets.keys()
    N = min(len(source_names), 4)
    description = source_names[0]   
    for i in range(1,N):
        description = description  + '_' + source_names[i]
    description = description + '-' + target_names[0]
    return description

def channel_dropout(X, p):
    if p == 0:
        return X
    mask = tf.random_uniform(shape = [tf.shape(X)[0], tf.shape(X)[2]])
    mask = mask + 1 - p
    mask = tf.floor(mask)
    dropout = tf.expand_dims(mask,axis = 1) * X/(1-p)
    return dropout
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_data_np(data_dir, time=600):
    samples = []
    group = []
    
    for path in os.listdir(data_dir):
        data = np.loadtxt(os.path.join(data_dir, path))
        control = "C" in path[1:path.rfind(' (')]
        samples.append(data)
        group.append(0 if control else 1)
    return samples, group

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs*0.5
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y

def val_split(X, Y, inds, split=0.66, time=800):
    x_train, y_train, x_val, y_val, grouped_X, grouped_Y = [], [], [], [], [], []
    for i in inds:
        x, y = to_train_samples(X[i:i+1], Y[i:i+1], time)
        ind = int(x.shape[0]*split)
        x_train += [x[i] for i in range(ind)]
        y_train += [y[i] for i in range(ind)]
        x_val += [x[i] for i in range(ind, x.shape[0])]
        y_val += [y[i] for i in range(ind, x.shape[0])]
        grouped_X.append(np.asarray([x[i] for i in range(ind, x.shape[0])]))
        grouped_Y.append(np.asarray([y[i] for i in range(ind, x.shape[0])]))
    
    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val), grouped_X, grouped_Y

def read_file(file):
    return pd.read_csv(file, delimiter='\t', skiprows=[0])

def resample(df, time, current, desired):
    resampled = []
    df = np.asarray(df)
    duration = time[df.shape[0]-1] - time[0]
    x_current = time - time[0]
    x_desired = np.arange(0, duration+1/desired, 1/desired)
    #print(x_current.shape, x_desired.shape, df[0].shape)
    for i in range(df.shape[1]):
        resampled.append(np.interp(x_desired, x_current, df[:, i]))
    return np.asarray(resampled).transpose()

def get_freq(df):
    time = df['Time']
    fs = int(time.shape[0] / (time[time.shape[0]-1]-time[0]))
    return fs


def integrate(X, Y):
    res = 0
    for i in range(1, X.shape[0]):
        x1, x2, y1, y2 = X[i-1], X[i], Y[i-1], Y[i]
        
        rect = (x2-x1)*y1
        triangle = (x2-x1)*(y2-y1)/2
        
        res += rect + triangle
    return res*-1

def rates(true, pred, points=50):
    thresholds = np.linspace(0, 1, num=points, endpoint=True)
    tprs = []
    fprs = []
    for i in range(thresholds.shape[0]):
        t = thresholds[i]
        t_pred = pred > t
        tp = np.logical_and(t_pred==1, true==1).sum()
        fp = np.logical_and(t_pred==1, true==0).sum()
        tn = np.logical_and(t_pred==0, true==0).sum()
        fn = np.logical_and(t_pred==0, true==1).sum()
        
        tprs.append(tp/(tp+fn))
        fprs.append(fp/(tn+fp))

    return np.asarray(fprs), np.asarray(tprs), thresholds

def update_freqs(freqs, counts, path):
    files = os.listdir(path)
    for p in files:
        df = read_file(os.path.join(path, p))
        for col in df.columns:
            if col.lower() == "sample" or col.lower() == "time":
                continue
            if "A" not in col and "G" not in col:
                counts[col] += 1
        freqs[get_freq(df)] += 1
    return
        
        
def get_paths(file_dir):
    res = []
    
    def helper(path):
        list_dir = os.listdir(path)
        for f in list_dir:
            f_path = os.path.join(path, f)
            if os.path.isdir(f_path):
                helper(f_path)
            else:
                res.append(f_path)
    helper(file_dir)
                
    return res


def group(X, Y):
    x, y = [], []
    start, end = 0, 0
    curr = Y[0]
    for i in range(Y.shape[0]):
        if curr != Y[i]:
            x.append(X[start:end])
            y.append(Y[start:end])
            start = end
            curr = Y[i]
        end += 1
    return x, y

def CI(X):
    res = []
    n_root = X.shape[0]**.5

    for i in range(X.shape[1]):
        mean = X[:, i].mean()
        std = X[:, i].std()
        error = 1.96*std/n_root
        res.append([mean-error, mean+error])
    return np.asarray(res)

def my_auc(true, pred):
    fpr, tpr, _ = my_roc(true, pred)
    return integrate(fpr, tpr)

def pick_balanced(Y):
    s = []
    c = []
    
    for i in range(len(Y)):
        if Y[i] == 1:
            s.append(i)
        else:
            c.append(i)
            
    return s + random.sample(c, k=len(s))


def my_roc(true, pred, points=50):
    true = np.asarray(true)
    pred = np.asarray(pred)
    thresholds = np.linspace(0, 1, num=points, endpoint=True)
    tprs = []
    fprs = []
    for i in range(thresholds.shape[0]):
        t = thresholds[i]
        t_pred = pred > t
        tp = np.logical_and(t_pred==1, true==1).sum()
        fp = np.logical_and(t_pred==1, true==0).sum()
        tn = np.logical_and(t_pred==0, true==0).sum()
        fn = np.logical_and(t_pred==0, true==1).sum()
        
        tprs.append(tp/(tp+fn))
        fprs.append(fp/(tn+fp))

    return np.asarray(fprs), np.asarray(tprs), thresholds


def moving_average(lst, window=5):
    return sum(lst[max(0, len(lst)-window):])/min(window, len(lst))

def min_ind(l):
    res = float('inf')
    curr = None
    for i in range(len(l)):
        if l[i] < res:
            res = l[i]
            curr = i
    return curr