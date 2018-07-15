
# coding: utf-8

# In[30]:


import numpy as np
import keras
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Dropout, Activation
import string
import sys
import argparse


# In[4]:


seq_length = 60
batch_size = 512
dropout = 0.2
layer_dim = 512
nb_epochs = 30
layer_count = 4


# In[5]:


characters = list(string.printable)
characters.remove('\x0b')
characters.remove('\x0c')

vocab_size = len(characters)
char_to_int = dict((c, i) for i, c in enumerate(characters))
int_to_char = dict((i, c) for i, c in enumerate(characters))


# In[12]:


def Train():
    text_train = open("dataset_train.txt").read()
    text_train_len = len(text_train)
    text_val = open("dataset_val.txt").read()
    text_val_len = len(text_val)
    
    train_batch_count = (text_train_len - seq_length) // batch_size
    val_batch_count = (text_val_len - seq_length) // batch_size
    
    def batch_generator(text, count):
        while True:
            for batch_idx in range(count):
                X = np.zeros((batch_size, seq_length, vocab_size))
                y = np.zeros((batch_size, vocab_size))
        
                batch_offset = batch_size * batch_idx
                for sample_idx in range(batch_size):
                    sample_start = batch_offset + sample_idx
                    for s in range(seq_length):
                        X[sample_idx, s, char_to_int[text[sample_start + s]]] = 1
                    y[sample_idx, char_to_int[text[sample_start + s + 1]]] = 1
        
                yield X, y
    
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    filepath = "./model" 
    
    checkpoint = ModelCheckpoint(
        filepath,
        save_weights_only=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    callbacks_list = [checkpoint, early_stopping]
    
    print(text_train_len)
    print(text_val_len)
    print(train_batch_count)
    print(val_batch_count)
    
    def RNN():
        model = Sequential()
        model.add(LSTM(layer_dim, return_sequences = True, input_shape = (seq_length, vocab_size)))
        model.add(Dropout (0.2))
        model.add(LSTM(layer_dim, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(layer_dim, return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(layer_dim, return_sequences = False))
        model.add(Dropout(0.2))
        model.add(Dense(vocab_size, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
        return model
    
    model = RNN()
    
    history = model.fit_generator(
    batch_generator(text_train, count=train_batch_count),
    train_batch_count,
    max_queue_size=1,
    epochs=nb_epochs,
    callbacks=callbacks_list,
    validation_data=batch_generator(text_val, count=val_batch_count),
    validation_steps=val_batch_count,
    initial_epoch = 0
    )


# In[28]:


def Generate(num_gen = 5, stri = "It is", cnt = 140):
    
    def RNN():
        model = Sequential()
        model.add(LSTM(layer_dim, return_sequences = True, batch_input_shape = (1, 1, vocab_size), stateful = True))
        model.add(Dropout (0.2))
        model.add(LSTM(layer_dim, return_sequences = True, stateful = True))
        model.add(Dropout(0.2))
        model.add(LSTM(layer_dim, return_sequences = True, stateful = True))
        model.add(Dropout(0.2))
        model.add(LSTM(layer_dim, return_sequences = False, stateful = True))
        model.add(Dropout(0.2))
        model.add(Dense(vocab_size, activation = 'softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
        return model
    
    model = RNN()
    model.load_weights("./weights/weights")
    
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def predict_next_char(model, current_char, diversity=1.0):
        x = np.zeros((1, 1, vocab_size))
        x[:,:,char_to_int[current_char]] = 1
        y = model.predict(x, batch_size=1)
        next_char_idx = sample(y[0,:], temperature=diversity)
        next_char = characters[next_char_idx]
        return next_char

    def generate_text(model, seed="I am", count=cnt):
        model.reset_states()
        for s in seed[:-1]:
            next_char = predict_next_char(model, s)
        current_char = seed[-1]

        sys.stdout.write("["+seed+"]")
        
        for i in range(count - len(seed)):
            next_char = predict_next_char(model, current_char, diversity=0.5)
            current_char = next_char
            sys.stdout.write(next_char)
        print("...\n")
    
    for i in range(num_gen):
        generate_text(
            model,
            seed=stri
        )
        


# In[31]:


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str)
    parser.add_argument("--num_sentences", type = int, default = 5)
    parser.add_argument("--string", type = str, default = "It is")
    parser.add_argument("--num_chars", type = int, default = 140)
    args = parser.parse_args()
    return args


# In[32]:


if __name__ == "__main__":
    args = get_args()
    if(args.mode == "train"):
        Train()
    elif(args.mode == "generate"):
        Generate(args.num_sentences, args.string, args.num_chars)

