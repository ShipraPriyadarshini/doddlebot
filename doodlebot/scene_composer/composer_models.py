import random
import math
from contextlib import contextmanager
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import masked_array
import numpy.ma as ma
import tensorflow as tf

class Singleton(type):
    """Metaclass for Singleton objects

    From Python Cookbook, 3rd edition, by David Beazley and Brian K. Jones (O’Reilly).
    Copyright 2013 David Beazley and Brian Jones, 978-1-449-34037-7
    """
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class ReservoirSample1:
    """Randomly sample single item from data stream of indeterminate length
    
    Implementation after Algorithm L:
    Li, Kim-Hung. "Reservoir-sampling algorithms of time complexity O (n (1+ log (N/n)))."
    ACM Transactions on Mathematical Software (TOMS) 20, no. 4 (1994): 481-493.
    """
    
    def _update_s(self):
        """Update s based on w"""
        self.s = int(math.log(random.random()) / math.log(1.0 - self.w))
        
    def __init__(self):
        self.w = random.random()
        self._update_s()
        self.data = None
        self.stepno = 1
        self.target_step = 2 + self.s
        self.last_data_step = None # Just for debug
        
    def step(self, data):
        """Single time-step of data stream"""
        if self.stepno == 1:
            self.data = data
        elif self.stepno == self.target_step:
            self.data = data
            self.w = self.w * random.random()
            self._update_s()
            self.target_step = self.stepno + 1 + self.s
            self.last_data_step = self.stepno # Debug
        self.stepno += 1
            

class AttentionVis(metaclass=Singleton):
    """Global mechanism for saving and displaying attention-matrix visualization
    
    Store and later retrieve (and display) a single, randomly chosen attention matrix and
    its corresponding English sentence from the decoder block while the "test"
    operation is running.
    
    This class depends on the call structure (and argument order) of many stencil
    functions being preserved. It also depends on several function decorators
    and a "with" statement inside the transformer code to succesfully store and
    retrieve its data.
    """ 
    def __init__(self):
        self.enabled = False # Whether visualization even runs (decided by student)
        self.in_test = False # Shims are in the test function (as opposed to train)
        self.in_decoder = False # Shims are in the decoder (as opposed to encoder)
        self.rsample1 = ReservoirSample1() # Random sampling engine
        
        # randomly selected data within current batch
        # shared between the different shims
        self.cur_batch_data = { 
            "sent_ids": None,
            "att_mat": None,
            "index": None,
        }
        
        # Final data for showing heat map
        self.atten_matrix = None # The matrix itself (14x14)
        self.sentence = None # List of the words of the sentence
        self.rev_en_vocab = None # Reverse English vocabulary (id->word)

    def _setup_atten_heatmap(self, ax):
        """
        Create a heatmap from a numpy array and two lists of labels.
    
        This function derived from:
        https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

        ax - a "matplotlib.axes.Axes" instance to which the heatmap is plotted
        """
        
        data = self.atten_matrix
        row_labels = col_labels = self.sentence

        cbarlabel="Attention Score"
        cbar_kw={}

        # Plot the heatmap
        im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

    def setup_visualization(self, enable=False):
        """Allow the student to turn visualization on or off"""
        self.enabled = enable

    def show_atten_heatmap(self):
        """Display heatmap from saved data after test run complete"""
        if self.enabled and self.atten_matrix is not None:
            fig, ax = plt.subplots()
            self._setup_atten_heatmap(ax)
            fig.tight_layout()
            plt.show()
        
    def test_func(self, func):
        """Shim for student's top-level test function
        
        Set a flag that signals to the other shims that we're inside the test function.
        Turn off graph execution if visualization is enabled.
        """
        def wrapper(*args, **kwargs):
            if self.enabled:
                self.in_test = True
                # The attention visualization is not compatible with tf.function. Oops.
                # Turn it off (but only if student requested visualizations)
                # The good news is: this won't affect "train"
                tf.config.run_functions_eagerly(True)
            ret = func(*args, **kwargs)
            # Save the data
            if self.enabled and self.rsample1.data:
                self.atten_matrix = self.rsample1.data["att_mat"]
                self.sentence = [self.rev_en_vocab[word] for word in self.rsample1.data["sent_ids"]]
                tf.config.run_functions_eagerly(False)
                self.in_test = False
                print("Collecting att matrix from batch", self.rsample1.last_data_step)
            return ret
        return wrapper
    
    def call_func(self, func):
        """Shim for student's main transformer call function
        
        If inside the test function, pick a random decoder sentence and remember its index
        """
        def wrapper(*args, **kwargs):
            if self.enabled and self.in_test:
                decoder_input = args[2]
                ridx = random.randint(0, len(decoder_input)-1)
                self.cur_batch_data["index"] = ridx
                self.cur_batch_data["sent_ids"] = list(decoder_input[ridx])
            ret = func(*args, **kwargs)
            if self.enabled and self.in_test:
                # Attention matrix should be collected at this point
                self.rsample1.step(self.cur_batch_data)
            return ret
        return wrapper
        
    def att_mat_func(self, func):
        """Shim for student's self-attention function
        
        If in the test function, and this is a decoder, store the attention matrix
        that corresponds with the saved index from the batch.
        """
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if self.enabled and self.in_test and self.in_decoder:
                # The return value of this function contains the to-be-visualized
                # attention matrix
                self.cur_batch_data["att_mat"] = ret[self.cur_batch_data["index"]].numpy()
            return ret
        return wrapper
        
    def get_data_func(self, func):
        """Shim for student's get-data function
        
        Collect the English vocab and reverse it
        """
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if self.enabled:
                en_vocab = ret[4]
                self.rev_en_vocab = {v:k for k,v in en_vocab.items()}
            return ret
        return wrapper
        
    @contextmanager
    def trans_block(self, is_decoder):
        """Shim for recording which transformer block we're in (encoder/decoder)"""
        try:
            self.in_decoder = is_decoder
            yield
        finally:
            self.in_decoder = False

class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_shape, **kwargs):
        super().__init__(**kwargs)
        self.embed_shape = embed_shape

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel', shape=self.embed_shape,
                                 initializer=tf.keras.initializers.GlorotUniform(), trainable=True)

    def call(self, x=None):
        return self.w

av = AttentionVis()

def Attention_Matrix(K, Q, use_mask=False):
  """
  This functions runs a single attention head.
  :param K: is [batch_size x window_size_keys x embedding_size]
  :param Q: is [batch_size x window_size_queries x embedding_size]
  :return: attention matrix
  """

  window_size_queries = Q.get_shape()[1] # window size of queries
  window_size_keys = K.get_shape()[1] # window size of keys
  mask = tf.convert_to_tensor(value=np.transpose(np.tril(np.ones((window_size_queries,window_size_keys))*np.NINF,-1),(1,0)),dtype=tf.float32)
  atten_mask = tf.tile(tf.reshape(mask,[-1,window_size_queries,window_size_keys]),[tf.shape(input=K)[0],1,1])

  # TODO:
  # 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
  # 2) return the attention matrix


  # Check lecture slides for how to compute self-attention.
   # You can use tf.transpose or tf.tensordot to perform the matrix multiplication for 3D matrices
  # Remember:
  # - Q is [batch_size x window_size_queries x embedding_size]
  # - K is [batch_size x window_size_keys x embedding_size]
  # - Mask is [batch_size x window_size_queries x window_size_keys]


  # Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
  # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
  # Those weights are then used to create linear combinations of the corresponding values for each query.
  # Those queries will become the new embeddings.
  if use_mask == False:
    attn_mat = tf.nn.softmax(tf.linalg.matmul(Q,K, transpose_b = True)/np.sqrt(K.get_shape()[2]))

  else:
    attn_mat = tf.nn.softmax(tf.linalg.matmul(Q,K, transpose_b = True)/np.sqrt(K.get_shape()[2]) + atten_mask)

  return attn_mat


class Atten_Head(tf.keras.layers.Layer):
  def __init__(self, input_size, output_size, use_mask):
    super(Atten_Head, self).__init__()

    self.use_mask = use_mask

    # TODO:
    # Initialize the weight matrices for K, V, and Q.
    # They should be able to multiply an input_size vector to produce an output_size vector
    # Hint: use self.add_weight(...)
    self.K_weight = self.add_weight("k_vec",shape=(input_size, output_size),initializer='random_normal',trainable=True)
    self.V_weight = self.add_weight("v_vec",shape=(input_size, output_size),initializer='random_normal',trainable=True)
    self.Q_weight = self.add_weight("q_vec",shape=(input_size, output_size),initializer='random_normal',trainable=True)

  @tf.function
  def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):

    """
    This functions runs a single attention head.
    :param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
    :param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
    :param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
    :return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
    """

    # TODO:
    # - Apply 3 matrices to turn inputs into keys, values, and queries. You will need to use tf.tensordot for this.
    # - Call Attention_Matrix with the keys and queries, and with self.use_mask.
    # - Apply the attention matrix to the values

    K = tf.matmul(inputs_for_keys,self.K_weight)
    V = tf.matmul(inputs_for_values,self.V_weight)
    Q = tf.matmul(inputs_for_queries,self.Q_weight)

    attn_mat = Attention_Matrix(K, Q, use_mask=self.use_mask)
    output_mat = tf.matmul(attn_mat, V)

    return output_mat



class Multi_Headed(tf.keras.layers.Layer):
  def __init__(self, emb_sz, use_mask):
    super(Multi_Headed, self).__init__()

    # TODO:
    # Initialize heads
    self.emb_sz = emb_sz
    self.batch = int(emb_sz/3)
    self.head1 = Atten_Head(emb_sz, self.batch , use_mask)
    self.head2 = Atten_Head(emb_sz, self.batch, use_mask)
    self.head3 = Atten_Head(emb_sz, self.batch, use_mask)
    self.use_mask = use_mask
    self.dense = tf.keras.layers.Dense(emb_sz)

  @tf.function
  def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
    """
    FOR CS2470 STUDENTS:
    This functions runs a multiheaded attention layer.
    Requirements:
      - Splits data for 3 different heads into size embed_sz/3
      - Create three different attention heads
      - Each attention head should have input size embed_size and output embed_size/3
      - Concatenate the outputs of these heads together
      - Apply a linear layer
    :param inputs_for_keys: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
    :param inputs_for_values: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
    :param inputs_for_queries: tensor of [batch_size x [ENG/FRN]_WINDOW_SIZE x input_size ]
    :return: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x output_size ]
    """
  
    output1 = self.head1.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
    output2 = self.head2.call(inputs_for_keys, inputs_for_values, inputs_for_queries)
    output3 = self.head3.call(inputs_for_keys, inputs_for_values, inputs_for_queries)


    #output = tf.transpose(tf.stack([output1,output2,output3]), perm = [1,2,0,3])
    #output = tf.reshape(output, (inputs_for_values.get_shape()[0], -1, self.emb_sz))
    output = tf.concat([output1,output2,output3],axis=2)
    
    output = self.dense(output)

    return output


class Feed_Forwards(tf.keras.layers.Layer):
  def __init__(self, emb_sz):
    super(Feed_Forwards, self).__init__()

    self.layer_1 = tf.keras.layers.Dense(emb_sz,activation='relu')
    self.layer_2 = tf.keras.layers.Dense(emb_sz)

  @tf.function
  def call(self, inputs):
    """
    This functions creates a feed forward network as described in 3.3
    https://arxiv.org/pdf/1706.03762.pdf
    Requirements:
    - Two linear layers with relu between them
    :param inputs: input tensor [batch_size x window_size x embedding_size]
    :return: tensor [batch_size x window_size x embedding_size]
    """
    layer_1_out = self.layer_1(inputs)
    layer_2_out = self.layer_2(layer_1_out)
    return layer_2_out

class Transformer_Block(tf.keras.layers.Layer):
  def __init__(self, emb_sz, is_decoder, multi_headed=False):
    super(Transformer_Block, self).__init__()
    self.is_decoder = is_decoder
    self.ff_layer = Feed_Forwards(emb_sz)
    self.self_atten = Atten_Head(emb_sz,emb_sz,is_decoder) if not multi_headed else Multi_Headed(emb_sz,use_mask=is_decoder)
    if self.is_decoder:
      self.self_context_atten = Atten_Head(emb_sz,emb_sz,False) if not multi_headed else Multi_Headed(emb_sz,use_mask=False)

    self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

  @tf.function
  def call(self, inputs, context=None):
    """
    This functions calls a transformer block.
    There are two possibilities for when thits function is called.
        - if self.is_decoder == False, then:
            1) compute unmasked attention on the inputs
            2) residual connection and layer normalization
            3) feed forward layer
            4) residual connection and layer normalization
        - if self.is_decoder == True, then:
            1) compute MASKED attention on the inputs
            2) residual connection and layer normalization
            3) computed UNMASKED attention using context
            4) residual connection and layer normalization
            5) feed forward layer
            6) residual layer and layer normalization
    If the multi_headed==True, the model uses multiheaded attention (Only 2470 students must implement this)
    :param inputs: tensor of [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ]
    :context: tensor of [BATCH_SIZE x FRENCH_WINDOW_SIZE x EMBEDDING_SIZE ] or None
      default=None, This is context from the encoder to be used as Keys and Values in self-attention function
    """

    with av.trans_block(self.is_decoder):
      atten_out = self.self_atten(inputs,inputs,inputs)
    atten_out+=inputs
    atten_normalized = self.layer_norm(atten_out)

    if self.is_decoder:
      assert context is not None,"Decoder blocks require context"
      context_atten_out = self.self_context_atten(context,context,atten_normalized)
      context_atten_out+=atten_normalized
      atten_normalized = self.layer_norm(context_atten_out)

    ff_out=self.ff_layer(atten_normalized)
    ff_out+=atten_normalized
    ff_norm = self.layer_norm(ff_out)

    return tf.nn.relu(ff_norm)

class Position_Encoding_Layer(tf.keras.layers.Layer):
  def __init__(self, window_sz, emb_sz):
    super(Position_Encoding_Layer, self).__init__()
    self.positional_embeddings = self.add_weight("pos_embed",shape=[window_sz, emb_sz])

  @tf.function
  def call(self, x):
    """
    Adds positional embeddings to word embeddings.
    :param x: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] the input embeddings fed to the encoder
    :return: [BATCH_SIZE x (ENG/FRN)_WINDOW_SIZE x EMBEDDING_SIZE ] new word embeddings with added positional encodings
    """
    
    return x+self.positional_embeddings

import numpy as np
import tensorflow as tf

class Transformer(tf.keras.Model):
  def __init__(self):

    ######vvv DO NOT CHANGE vvv##################
    super(Transformer, self).__init__()

    

    # Define batch size and optimizer/learning rate
    self.batch_size = 100
    self.embedding_size = 200
    self.learning_rate = 0.001
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

   
    # Create positional encoder layers
    self.pos_sentence = Position_Encoding_Layer(enc_inputs.shape[1], self.embedding_size)
    self.pos_sentence1 = Position_Encoding_Layer(enc_inputs.shape[1], self.embedding_size)

    # Define decoder layers:
    self.encoder = Transformer_Block(self.embedding_size, is_decoder = False)

#TRANSFORMER DECODER
    # Define decoder layers:
    self.query_embed = FixedEmbedding((self.batch_size,enc_inputs.shape[1],self.embedding_size))
    self.decoder = Transformer_Block(self.embedding_size, is_decoder = True)

		# Define dense layer(s)
    self.box_dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.box_dense2 = tf.keras.layers.Dense(500, activation='relu')
    self.box_dense3 = tf.keras.layers.Dense(4, activation='sigmoid')

    self.class_dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.class_dense2 = tf.keras.layers.Dense(500, activation='relu')
    self.class_dense3 = tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax')

  @tf.function
  def call(self, encoder_input, decoder_input=None):
    """
    :param encoder_input: batched ids corresponding to French sentences
    :param decoder_input: batched ids corresponding to English sentences
    :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
    """
    encoder_output = self.pos_sentence(encoder_input)
    encoder_output = self.encoder(encoder_output)

    decoder_output = self.pos_sentence1(self.query_embed(None))
    decoder_output = self.decoder(decoder_output,context = encoder_output)

    box_output = self.box_dense1(decoder_output)
    box_output = self.box_dense2(box_output)
    box_output = self.box_dense3(box_output)

    class_output = self.class_dense1(decoder_output)
    class_output= self.class_dense2(class_output)
    class_output = self.class_dense3(class_output)


    return box_output , class_output

  def accuracy_function(self, prbs, labels, mask):
    """
    DO NOT CHANGE
    Computes the batch accuracy

    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: scalar tensor of accuracy of the batch between 0 and 1
    """

    decoded_symbols = tf.argmax(input=prbs, axis=2)
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
    return accuracy


  def loss_function(self, box_pred, box_labels, class_pred, class_labels):
    """
    Calculates the model cross-entropy loss after one forward pass
    Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: the loss of the model as a tensor
    """
    gl = tfa.losses.GIoULoss()
    box_loss = gl(box_pred, box_labels)
    class_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(class_labels, class_pred, from_logits = False))
    return 10*tf.cast(box_loss,tf.float32) + class_loss
    



  def __call__(self, *args, **kwargs):
    return super(Transformer, self).__call__(*args, **kwargs)

def train(model, enc_inputs, labels, class_labels):
    
    loss = 0 
    num_batches = len(enc_inputs)//model.batch_size
    epoch_loss = 0
    for batch in range(num_batches):
        encoder_input_train = enc_inputs[model.batch_size * batch: ((model.batch_size * batch) + model.batch_size)]
        #print(encoder_input_train.shape)
        batched_labels = labels[model.batch_size * batch: ((model.batch_size * batch) + model.batch_size)]
        batched_class_labels = class_labels[model.batch_size * batch: ((model.batch_size * batch) + model.batch_size)]
        #decoder_input_train = decoder_train[:,:-1]
        #print(decoder_input_train.shape)
        #decoder_labels_train = decoder_train[:,1:]

        #mask = ma.masked_where(batched_labels == [-1,-1,-1,-1], batched_labels)
        #mask = np.where(np.all(batched_labels==[-1,-1,-1,-1],axis=1), 0, 1)
        #X_masked = np.ma.getmask(mask)
        #print(X_masked)


        with tf.GradientTape() as tape:
            box_pred, class_pred = model.call(encoder_input_train, None)
            loss = model.loss_function(box_pred, batched_labels, class_pred, batched_class_labels)
            epoch_loss += loss
    
    


        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(epoch_loss/num_batches)
    return None



