import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from preprocessing import *

def build_model(total_words, max_sequence_len):
    model = tf.keras.Sequential([
        Embedding(total_words, 100, input_length = max_sequence_len-1),
        Bidirectional(LSTM(150,return_sequences = True)),
        Dropout(0.2),
        LSTM(100),
        Dense(total_words/2, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)),
        Dense(total_words, activation = 'softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def train():
    input_sequence, num_words, _ = readDataTokenize('data.txt')
    max_sequence_len = max([len(x) for x in input_sequence])
    x,y = preprocessing(input_sequence, num_words)
    model = build_model(num_words,max_sequence_len)
    print(model.summary())
    model.fit(x,y, epochs = 150, verbose = 1)
    model.save("output/")


def predict(model, test_sequence):
    input_sequence, num_words, token = readDataTokenize('data.txt')
    next_words = 10
   
    for _ in range(next_words):
        token_list = token.texts_to_sequences([test_sequence])[0]
        token_list = pad_sequences([token_list], maxlen = 10, padding = 'pre')
        predicted = list(model.predict(token_list, verbose = 0)[0])
        predicted = predicted.index(max(predicted))
        print(predicted)

        output_words = ""
        for word, index in token.word_index.items():
            if index == predicted:
                output_words = word
                break
        test_sequence += " "+output_words

    print(test_sequence)    


model = tf.keras.models.load_model('output')
predict(model, "despite of wrinkles")
