from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    # https://keras.io/layers/normalization/ 
    # "# Note that we can name any layer by passing it a "name" argument." from https://keras.io/getting-started/functional-api-guide/
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    # https://keras.io/layers/wrappers/
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    # https://keras.io/layers/normalization/ 
    # "# Note that we can name any layer by passing it a "name" argument." from https://keras.io/getting-started/functional-api-guide/
    bn_rnn = BatchNormalization(name='bn_rnn_conv_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    # initialize input data variable
    var_input_data = input_data
    # initialize rnn array 
    deep_simp_rnn = []
    # initialize batch normalization rnn array
    deep_bn_rnn = [] 
    # for each recurrent layer add simple rnn and add batch normalization to reduce training time
    for rl in range(recur_layers):
        # add recurrent layer
        simp_rnn = GRU(units, activation='relu',
                       return_sequences=True, implementation=2, name='rnn_'+str(rl))(var_input_data)
        # add batch normalization
        bn_rnn = BatchNormalization(name='bn_rnn_'+str(rl))(simp_rnn)
        # append actual rnn to rnn array
        deep_simp_rnn.append(simp_rnn)
        # append actual batch normalization rnn to batch normalization rnn array
        deep_bn_rnn.append(bn_rnn)
        # the output sequence of the first recurrent layer is used as input for the next recurrent layer
        var_input_data = deep_bn_rnn[rl]

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(var_input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    # https://keras.io/layers/wrappers/
    # When specifying the Bidirectional wrapper, use merge_mode='concat'.
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2), 
                                    merge_mode='concat', name='bidir_rnn')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    
    # So far, you have experimented with a single bidirectional RNN layer. 
    # Consider stacking the bidirectional layers, to produce a deep bidirectional RNN! (https://www.cs.toronto.edu/~graves/asru_2013.pdf)
    
    # initialize input data variable
    var_input_data = input_data
    # initialize deep bidirectional RNN 
    deep_bidir_rnn = []
    # initialize batch normalization deep bidirectional RNN
    deep_bn_bidir_rnn = [] 
    # for each recurrent layer add bidirectional recurrent layer 
    # with dropout to avoid overfitting and add batch normalization to reduce training time
    for rl in range(recur_layers):
        # Add bidirectional recurrent layer with dropout
        bidir_rnn = Bidirectional(LSTM(units, activation='tanh',
                                    return_sequences=True, dropout=0.1, recurrent_dropout=0.1, implementation=2), 
                                    merge_mode='concat', name='bidir_rnn_'+str(rl))(input_data)
        # Add batch normalization
        bn_bidir_rnn = BatchNormalization(name='bn_bidir_rnn_'+str(rl))(bidir_rnn)
        
        # append actual bidirectional RNN to bidirectional RNN array
        deep_bidir_rnn.append(bidir_rnn)
        # append actual batch normalization bidirectional RNN to batch normalization bidirectional RNN array
        deep_bn_bidir_rnn.append(bn_bidir_rnn)
        # the output sequence of the first bidirectional recurrent layer is used as input for the next bidirectional recurrent layer
        var_input_data = deep_bn_bidir_rnn[rl]      
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(var_input_data)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    print(model.summary())
    return model