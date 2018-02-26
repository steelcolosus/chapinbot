
# coding: utf-8

# # Chapin style converstaion generator

# In[1]:

import pandas as pd


# ## Transforming data into a script like format

# In[2]:

data_path = "data/velmaxdata.csv"
threads = pd.read_csv(data_path)


# In[3]:

threads.head()


# In[4]:

fields_to_drop = ['Unnamed: 0','date']
threads = threads.drop(fields_to_drop,axis=1)
threads.head()


# ### Building script blocks

# In[5]:

thread_id = threads[0:1]['threadid'][0]
print(thread_id)


# In[6]:

with open('./data/script.txt','w') as out_file:
    
    for index, thread in threads.iterrows():
        
        if thread['threadid'] != thread_id:
            out_file.write('\n\n')

        line = "{}: {}".format(thread['author'],thread['text']).strip()
        if not line.endswith('.'):
            line+='.'

        out_file.write(line+'\n')
        
        thread_id = thread['threadid']


# ## Checkpoint 
# 
# After processing the data you can start here loading the transforming data directly. The preprocessed data has been saved to disk.

# In[1]:

import helper
data_dir = './data/script.txt'


# In[2]:

text = helper.load_data(data_dir)
index = len(text)//100
text = text[:index]


# ## Explore the data

# In[4]:

view_sentence_range = (0, 10)

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# ## Implementing Preprocessing Functions

# In order to preprocess the dataset we are going to implement the following preprocessing functions:
# * Lookup Table
# * Tokenize Punctuation

# ### Lookup Table
# 
# To create a word embedding, we need to transform the words to ids. We will create two dictionaries:
# * Dictionary to go from the words to an id, we'll call vocab_to_int
# * Dictionary to go from the id to word, we'll call int_to_vocab

# In[5]:

import numpy as np
from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of velmax scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    print('countint words')
    word_counts = Counter(text)
    print('sorting words')
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    print('generating int to vocab')
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    print('generating vocab to int')
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


# ## Tokenize punctuation
# 
# We'll be splitting the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".
# 
# The token_lookup function will return a dict that will be used to tokenize symbols like "?" into "||Question_Mark||"
# 
# This dictionary will be used to token the symbols and add the delimiter (space) around it. This separates the symbols as it's own word, making it easier for the neural network to predict on the next word

# In[6]:

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punct_dic = {
        '.' : '||Period||',
        ',' : '||Comma||',
        '"' : '||Quotation_Mark||',
        ';' : '||Semicolon||',
        '!' : '||Exclamation_Mark||',
        '?' : '||Question_Mark||',
        '(' : '||Left_Parentheses||',
        ')' : '||Right_Parentheses||',
        '--': '||Dash||',
        '\n': '||Return||'
    }
    return punct_dic


# # Pre process all data and save it

# In[7]:

import helper
data_dir = './data/script.txt'
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


# ## Checkpoint #2

# The preprocessed data has been saved to disk. We can start from here the next time

# In[8]:

import helper
import numpy as np

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


# ## Check Tensorflow version

# In[9]:

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ## Get tensors

# In[10]:

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    inputs_ = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs_, targets, learning_rate


# ## Build RNN Cell and Initialize

# In[11]:

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')
    
    return cell, initial_state


# ## Word embedding

# Apply embedding to input_data using TensorFlow. Return the embedded sequence.

# In[12]:

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


# ## Build RNN

# In[13]:

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state


# ## Build the Neural Network

# In[14]:

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    
    #Apply embedding to input
    embeddings = get_embed(input_data, vocab_size, embed_dim)
    
    #Build RNN
    output, final_state = build_rnn(cell, embeddings)
    
    
    # add fully connected layer 
    # Setting activation function as None will implement a linear activation function
    
    #weights and biases
    weights = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
    biases = tf.zeros_initializer()
    logits = tf.contrib.layers.fully_connected(output, 
                                               vocab_size, 
                                               activation_fn=None,
                                               weights_initializer=weights,
                                               biases_initializer=biases)
    
    return logits, final_state


# ## Batches

# For exmple, get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2) would return a Numpy array of the following:
# ```
# [
#   #  First Batch
#   [
#     # Batch of Input
#     [[ 1  2], [ 7  8], [13 14]]
#     # Batch of targets
#     [[ 2  3], [ 8  9], [14 15]]
#   ]
# 
#   # Second Batch
#   [
#     # Batch of Input
#     [[ 3  4], [ 9 10], [15 16]]
#     # Batch of targets
#     [[ 4  5], [10 11], [16 17]]
#   ]
# 
#   # Third Batch
#   [
#     # Batch of Input
#     [[ 5  6], [11 12], [17 18]]
#     # Batch of targets
#     [[ 6  7], [12 13], [18  1]]
#   ]
# ]
# ```
# 
# Notice that the last target value in the last batch is the first input value of the first batch. In this case, 1. This is a common technique used when creating sequence batches, although it is rather unintuitive.
# 

# In[15]:

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    words_per_batch = seq_length * batch_size
    n_batches = len(int_text)//words_per_batch
    
    #Keep enough words to make full batches
    int_text = int_text[:n_batches*words_per_batch]
    
    #transform to numpy array
    inputs, targets = np.array(int_text, dtype= np.int32), np.roll(int_text,-1)
    
    #create inputs and targets
    #reshape into batch_size row
    inputs = inputs.reshape((batch_size, -1))
    targets = targets.reshape((batch_size, -1))
    
    #Create batch array
    batches = np.empty((n_batches, 2, batch_size, seq_length), dtype= np.int32 )
    
    #for batch_idx in range(0, n_batches):
    batch_id = 0
    for n in range(0, inputs.shape[1], seq_length):
        xn = inputs[:, n:n+seq_length]
        yn = targets[:, n:n+seq_length]
        batches[batch_id][0] = xn
        batches[batch_id][1] = yn
        batch_id += 1
    return batches


# ## Neural Network Training

# ### Hyperparameters

# Tune the following parameters:
# - Set num_epochs to the number of epochs.
# - Set batch_size to the batch size.
# - Set rnn_size to the size of the RNNs.
# - Set embed_dim to the size of the embedding.
# - Set seq_length to the length of sequence.
# - Set learning_rate to the learning rate.
# - Set show_every_n_batches to the number of batches the neural network should print progress.

# In[16]:

# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 256
# Sequence Length
seq_length = 12
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 2


# ## Build the graph

# In[17]:

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# ## Training

# In[18]:

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate
            }
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# In[ ]:



