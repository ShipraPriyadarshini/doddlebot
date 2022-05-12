import ijson
import json
import numpy as np
import tensorflow_addons as tfa
import tensorflow as tf
import codecs
import numpy as np

# INPUTS PREPROCESSING
'''
Input: Phrase from region_descriptions.json (individual words)
Output: Senetence Embedding 

Part 1:
- tuple of image_id and phrase from region_descriptions.json

Part 2:
- Get corresponding bounding boxes for objects in that region/phrase

Part 3:
- Preprocess the phrases into tokens

Part 4:
- Using GLOVE embeddings to convert tokens to embeddings
'''

f = open('data/image_data.json')

image_data = json.load(f)
wh_dict = {}
for i in range(len(image_data)):
  wh_dict[image_data[i]['image_id']] = [image_data[i]['width'], image_data[i]['height']]

# Part 1: Optimized (RAM issues)
label_map = []
enc_inputs = []
name_syn = {}

with open("data/region_graphs.json", "r") as f:
    count = 0
    #objt = []
    synsets = []
    for record in ijson.items(f, "item"):
        if count==5000:
          break
        image_id = record["image_id"]
        width = wh_dict[image_id][0]
        height = wh_dict[image_id][1]
        for j in range(7): #len(record["regions"])):
          phrase = record["regions"][j]["phrase"]
          region_id = record["regions"][j]["region_id"]
          obj = {}
          name2syn = {}
          for o in range(len(record["regions"][j]["objects"])):
            if record["regions"][j]["objects"][o]["synsets"]:
              name2syn[record["regions"][j]["objects"][o]["name"]] = record["regions"][j]["objects"][o]["synsets"][0]
              name_syn[record["regions"][j]["objects"][o]["name"]] = record["regions"][j]["objects"][o]["synsets"][0]
            else:
              name2syn[record["regions"][j]["objects"][o]["name"]] = record["regions"][j]["objects"][o]["name"]
              name_syn[record["regions"][j]["objects"][o]["name"]] = record["regions"][j]["objects"][o]["name"]

            '''def get_key(val):
              for key, value in name2syn.items():
                if val == value:
                  return key

            
            for i in set(synsets):
              name_synsets.append(get_key(i))'''

            x_min = record["regions"][j]["objects"][o]["x"]/width
            y_min = record["regions"][j]["objects"][o]["y"]/height
            x_max = (x_min + record["regions"][j]["objects"][o]["w"])/width
            y_max = (y_min + record["regions"][j]["objects"][o]["h"])/height
            obj[name2syn[record["regions"][j]["objects"][o]["name"]]] = [x_min,y_min,x_max,y_max]
            #objt.append(record["regions"][j]["objects"][o]['object_id'])
            synsets.append(name2syn[record["regions"][j]["objects"][o]["name"]])


          input = (image_id, region_id, phrase, obj, name2syn)
          label_map.append(input)
          enc_inputs.append(phrase)
          #dec_inputs.append(obj.values())
          
        count+=1

vocab = list(set(synsets))
vocab.append('no object')

VOCAB_SIZE = len(vocab)

# Part 3: Preprocess the phrases into tokens
#from nltk.tokenize import word_tokenize

def word_ind(sentences,vocab, name2syn):
  class_map = {}
  map = {}
  ulta_map = {}
  ulta_class = {}
  idx = 1
  class_id = 1
  for i in range(len(sentences)):
    words = sentences[i].split(" ")
    for j in words:
      if j not in map:
        map[j] = idx
        ulta_map[idx] = j
        idx+=1
      if j in name2syn and name2syn[j] in vocab:
        if name2syn[j] not in class_map:
          class_map[name2syn[j]] = class_id
          ulta_class[class_id] = name2syn[j]
          class_id+=1
      else:
          class_map[j] = 0

  return map, ulta_map, class_map, ulta_class

word_map, idx_map, class_map, ulta_class = word_ind(enc_inputs,vocab, name_syn)

enc_inputs = [sub.split(" ") for sub in enc_inputs]

for i in range(len(enc_inputs)):
  for j in range(len(enc_inputs[i])):
    enc_inputs[i][j] = word_map[enc_inputs[i][j]]

idx_map[0] = 'UNK'
ulta_class[0] = 'no object'
class_map['UNK'] = 0

#Padding the input sentences that are in integer form 

enc_inputs = tf.keras.preprocessing.sequence.pad_sequences(enc_inputs, padding="post")
class_labels = np.zeros((35000,enc_inputs.shape[1]))

for i in range(len(enc_inputs)):
  for j in range(len(enc_inputs[i])):
    if idx_map[enc_inputs[i][j]] in name_syn:
      class_labels[i][j] = class_map[name_syn[idx_map[enc_inputs[i][j]]]]
    else:
      class_labels[i][j] = class_map[idx_map[enc_inputs[i][j]]]

labels = np.zeros((len(enc_inputs),enc_inputs.shape[1], 4))

for i in range(len(enc_inputs)):
  for j in range(enc_inputs.shape[1]):
    name = idx_map[enc_inputs[i][j]]
    
    if name not in label_map[i][4]:
      continue
    
    class_label = label_map[i][4][name]
    labels[i][j] = label_map[i][3][class_label]

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def get_w2v(sentence, model, idx_map):
    """
    :param sentence: inputs a single sentences whose word embedding is to be extracted.
    :param model: inputs glove model.
    :return: returns numpy array containing word embedding of all words in input sentence.
    """
    return np.array([model.get(idx_map[val], np.zeros(200)) for val in sentence])

embedd = load_glove_model('/content/drive/MyDrive/DL Project/glove.6B/glove.6B.200d.txt')
embed_sentence = np.zeros((len(enc_inputs),len(enc_inputs[0]),200))

for i in range(len(enc_inputs)):
  embed_sentence[i] = get_w2v(enc_inputs[i], embedd, idx_map)