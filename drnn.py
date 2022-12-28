# -*- coding: utf-8 -*-
import copy
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn

def dRNN(cell, inputs, rate, scope='default'):
    """
    This function constructs a layer of dilated RNN.
    Inputs:
        cell -- the dilation operations is implemented independent of the RNN cell.
            In theory, any valid tensorflow rnn cell should work.
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
        rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper. 
        scope -- variable scope.
    Outputs:
        outputs -- the outputs from the RNN.
    """
    n_steps = len(inputs)# the input for the RNN.   inputs:list:286
    if rate < 0 or rate >= n_steps:#rate:1
        raise ValueError('The \'rate\' variable needs to be adjusted.')#inputs=n_steps*(batch_size*input_dims)
    print("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (scope, n_steps, rate, inputs[0].get_shape()[1]))#inputs[0]==n_steps

    # make the length of inputs divide 'rate', by using zero-padding
    EVEN = (n_steps % rate) == 0#EVEN=true
    if not EVEN:
        # Create a tensor in shape (batch_size, input_dims), which all elements are zero.  
        # This is used for zero padding
        zero_tensor = tf.zeros_like(inputs[0])
        dialated_n_steps = n_steps // rate + 1
        print("=====> %d time points need to be padded. " % (dialated_n_steps * rate - n_steps))#要填充的0的个数
        print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))
        for i_pad in range(dialated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)#把0填充进去
    else:
        dialated_n_steps = n_steps // rate
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate],
                                axis=0) for i in range(dialated_n_steps)]#[x1; x2],dilated_inputs=list:286----tensor(28,1)
    # building a dialated RNN with reformated (dilated) inputs
    dilated_outputs, _ = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs,
        dtype=tf.float32, scope=scope)#dilated_outputs=list:286----------tensor(28,100)
    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to 
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
	#tf.split(output, rate, axis=0)
    splitted_outputs = [tf.split(output, rate, axis=0)
                        for output in dilated_outputs]#splitted_outputs=list:286---list[tensor(28,100)]
    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]#unrolled_outputs:list:286-----tensor:(28,100)
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]
    return outputs


def multi_dRNN_with_dilations(cells, inputs, dilations):
    """
    This function constucts a multi-layer dilated RNN. 
    Inputs:
        cells -- A list of RNN cells.
        inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
    Outputs:
        x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
    """
    assert (len(cells) == len(dilations))
    outputs = []        
    output = []         
    x = copy.copy(inputs)#一个是导入copy模块。copy.copy(数据),inputs:list:286---tensor(28,1)
    i = 0 
    for cell, dilation in zip(cells, dilations):#dilation ：每一层
        scope_name = "multi_dRNN_dilation_%d" % i
        i +=1
        x= dRNN(cell, x, dilation, scope=scope_name) #x:list:286------tensor(28,100) ,x:list:50-----tensor(28,50)
        outputs.append(x)#list:3----list:286(28,100),list:286(28,50),list:286(28,50)
        x_trans = tf.stack(x,axis=0)#tf.stack是在新的维度上拼接，拼接后维度加1,,,(286，28，100)，，，（286，28，50）
        x_trans = tf.transpose(x_trans, [1,0,2])  
        output.append(x_trans)#output:list:3--(28,286,100),(28,286,50),(28,286,50)
    return outputs,output

	
def _contruct_cells(hidden_structs, cell_type):
    """
    This function contructs a list of cells.
    """
    # error checking
    if cell_type not in ["RNN", "LSTM", "GRU"]:
        raise ValueError("The cell type is not currently supported.")

    # define cells
    cells = []
    for hidden_dims in hidden_structs:
        if cell_type == "RNN":
            cell = tf.contrib.rnn.BasicRNNCell(hidden_dims)
        elif cell_type == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_dims)
        elif cell_type == "GRU":
            cell = tf.contrib.rnn.GRUCell(hidden_dims)
        cells.append(cell)

    return cells#list:3---GRUCell

def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take. 
    
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])#x=28,286,1,x_286,28,1
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])#x-=(8008,1)
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)#x_reformat=list:286---tensor(28,1)
    return x_reformat
    
def drnn_layer_final(x,
                        hidden_structs,
                        dilations,
                        n_steps,
                        input_dims,
                        cell_type):
    """
    This function construct a multilayer dilated RNN for classifiction.  
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        n_steps -- the length of the sequence.
        n_classes -- the number of classes for the classification.
        input_dims -- the input dimension.
        cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].
    
    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' does not pass any output activation functions.
    """
    # error checking
    assert (len(hidden_structs) == len(dilations))#hidden_structs=[100,50,50]
    
    # reshape inputs
    x_reformat = _rnn_reformat(x, input_dims, n_steps)#x=(28,286,1)   x_reformat=list:286---tensor(28,1)

    # construct a list of cells
    cells = _contruct_cells(hidden_structs, cell_type)#LIST:3----GRUCell

    # define dRNN structures
    outputs, output = multi_dRNN_with_dilations(cells, x_reformat, dilations)#output:[(28,286,100),(28,286,50),(28,286,50)]

    return outputs, output#outputs:[286][286][286]---286[tensor(28,100)],286[(28,50)],286[(28,50)]


