# Project:
#   VQA
# Description:
#   Text feature extraction related functions and classes
# Author: 
#   Sergio Tascon-Morales

from torch import nn

def get_text_feature_extractor(config, vocab_words):
    word_embedding_size = config['word_embedding_size']
    num_layers_LSTM = config['num_layers_LSTM']
    question_feature_size = config['question_feature_size']

    # instanciate the text encoder
    embedder = LSTMEncoder(word_embedding_size, num_layers_LSTM, question_feature_size, vocab_words)

    return embedder


class LSTMEncoder(nn.Module):

    def __init__(self, word_embedding_size, num_layers_LSTM, lstm_features, vocab_words):
        super().__init__()
        self.vocab_words = vocab_words
        self.word_embedding_size = word_embedding_size
        self.num_layers_LSTM = num_layers_LSTM
        self.lstm_features = lstm_features

        # create word embedding
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab_words)+1, embedding_dim=self.word_embedding_size, padding_idx=0)

        # create sequence encoder
        self.rnn = nn.LSTM(input_size=self.word_embedding_size, hidden_size=self.lstm_features, num_layers=self.num_layers_LSTM)

    def forward(self, question_vector):
        # question vector should be [B, max_question_length]
        x = self.embedding(question_vector) # [B, max_question_length, word_embedding_size]
        x = x.transpose(0,1) # put sequence dimension first, batch dim second [max_question_length, B, word_embedding_size]
        self.rnn.flatten_parameters() # * attempt to remove warning about non contiguous weights.
        output, (hn, cn) = self.rnn(x) # output is [max_question_length, B, lstm_features], hn and cn are [1, B, lstm_features]
        return cn.squeeze(0)