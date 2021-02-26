#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        
        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        self.embed_char_size = 50
        self.embed_size = embed_size

        self.char_embeddings = nn.Embedding(num_embeddings=len(vocab.char2id),
                                            embedding_dim=self.embed_char_size,
                                            padding_idx=pad_token_idx)
                    
        self.cnn = CNN(input_channel_num=self.embed_char_size, output_channel_num=self.embed_size,kernel_size=self.embed_size)
        self.highway = Highway(embed_size=self.embed_size)
        self.dropout = nn.Dropout(0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        print('The shape check 1:',input.shape)
        output = self.char_embeddings(input)
        print('The shape check 2:',output.shape)
        output = output.reshape(output.size(0)*output.size(1),output.size(2),output.size(3)).permute(0,2,1)
        print('The shape check 3:',output.shape)
        output = self.cnn(output)
        print('The shape check 4:',output.shape)
        output = self.highway(output)
        print('The shape check 5:',output.shape)
        output = self.dropout(output)
        print('The shape check 6:',output.shape)
        output = output.reshape(input.size(0),input.size(1),output.size(1))
        print('The shape check 7:',output.shape)
        return output
        ### END YOUR CODE

