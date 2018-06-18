

from common_functions import create_HiddenLayer_para
from mlp import HiddenLayer

import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time
import math
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random

from theano.tensor.signal import downsample
from random import shuffle
from load_data import  load_fasttext_word2vec_given_file,load_word2vec_to_init, load_trainingdata_il5,load_trainingdata_il6
from common_functions import store_model_to_file,cosine_row_wise_twoMatrix, Attentive_Conv_for_Pair,create_conv_para, create_HiddenLayer_para, average_f1_two_array_by_col, create_ensemble_para, Gradient_Cost_Para


def evaluate_lenet5(learning_rate=0.05, n_epochs=700, il5_emb_size=300, train_ratio=0.8,emb_size=300, batch_size=50, filter_size=[3,5], max_il5_phrase_len=5, hidden_size=300):
    model_options = locals().copy()
    print "model options", model_options
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    # srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))

    root = '/save/wenpeng/datasets/LORELEI/multi-lingual-emb/'
    # en_word2vec=load_fasttext_word2vec_given_file(root+'wiki.en.vec') #
    en_word2vec=load_fasttext_word2vec_given_file('/save/wenpeng/datasets/word2vec_words_300d.txt',300)
    il5_word2vec=load_fasttext_word2vec_given_file(root+'il5_300d_word2vec.txt',300)

    english_vocab = set(en_word2vec.keys())
    il5_vocab = set(il5_word2vec.keys())



    source_ids, source_masks, target_ids, il5_word2id, english_word2id = load_trainingdata_il5(root, english_vocab, il5_vocab, max_il5_phrase_len)
    # print len(english_vocab)
    # print len(english_word2id)
    # print set(english_word2id.keys()) - english_vocab
    assert set(english_word2id.keys()).issubset(english_vocab)
    assert set(il5_word2id.keys()).issubset(il5_vocab)

    data_size = len(target_ids)
    train_size  = int(data_size*train_ratio)
    dev_size = data_size-train_size
    print 'trainin size: ', train_size, ' dev_size: ', dev_size


    # all_sentences, all_masks, all_labels, word2id=load_BBN_multi_labels_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents=np.asarray(source_ids[:train_size], dtype='int32')
    train_masks=np.asarray(source_masks[:train_size], dtype=theano.config.floatX)
    train_target = np.asarray(target_ids[:train_size], dtype='int32')

    dev_sents=np.asarray(source_ids[-dev_size:], dtype='int32')
    dev_masks=np.asarray(source_masks[-dev_size:], dtype=theano.config.floatX)
    dev_target = np.asarray(target_ids[-dev_size:], dtype='int32')

    en_vocab_size=  len(english_word2id)
    en_rand_values=rng.normal(0.0, 0.01, (en_vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    en_id2word = {y:x for x,y in english_word2id.iteritems()}
    en_rand_values=load_word2vec_to_init(en_rand_values, en_id2word, en_word2vec)
    en_embeddings=theano.shared(value=np.array(en_rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable

    il5_vocab_size=  len(il5_word2id)+1 # add one zero pad index
    il5_rand_values=rng.normal(0.0, 0.01, (il5_vocab_size, il5_emb_size))   #generate a matrix by Gaussian distribution
    il5_id2word = {y:x for x,y in il5_word2id.iteritems()}
    il5_rand_values=load_word2vec_to_init(il5_rand_values, il5_id2word, il5_word2vec)
    il5_embeddings=theano.shared(value=np.array(il5_rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    # source_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable
    # target_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    batch_ids=T.imatrix() #(batch, maxlen)
    batch_masks = T.fmatrix() #(batch, maxlen)
    batch_targets = T.ivector()
    batch_test_source = T.fmatrix()  #(batch, emb_size)

    input_batch = il5_embeddings[batch_ids.flatten()].reshape((batch_size, max_il5_phrase_len, il5_emb_size)) #(batch, emb_size)
    masked_input_batch = T.sum(input_batch*batch_masks.dimshuffle(0,1,'x'), axis=1) #(batch, emb_size)
    # masked_input_batch = masked_input_batch / T.sum(batch_masks,axis=1).dimshuffle(0,'x')

    target_embs = en_embeddings[batch_targets]


    HL_layer_W1, HL_layer_b1 = create_HiddenLayer_para(rng, hidden_size, hidden_size)
    HL_layer_W2, HL_layer_b2 = create_HiddenLayer_para(rng, hidden_size, hidden_size)
    HL_layer_W3, HL_layer_b3 = create_HiddenLayer_para(rng, hidden_size, hidden_size)
    HL_layer_params = [HL_layer_W3, HL_layer_b3]#][HL_layer_W1, HL_layer_b1, HL_layer_W2, HL_layer_b2, HL_layer_W3, HL_layer_b3]
    #doc, by pos
    # HL_layer_1=HiddenLayer(rng, input=masked_input_batch, n_in=il5_emb_size, n_out=emb_size, W=HL_layer_W1, b=HL_layer_b1, activation=T.nnet.relu)
    # HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=emb_size, n_out=emb_size, W=HL_layer_W2, b=HL_layer_b2, activation=T.nnet.relu)
    HL_layer_3=HiddenLayer(rng, input=masked_input_batch, n_in=emb_size, n_out=emb_size, W=HL_layer_W3, b=HL_layer_b3, activation=T.tanh)

    batch_raw_pred = HL_layer_3.output
    # batch_pred = batch_raw_pred/T.sqrt(1e-8+T.sum(batch_raw_pred**2, axis=1)).dimshuffle(0,'x')
    batch_cosine = T.mean(cosine_row_wise_twoMatrix(batch_raw_pred, target_embs))#T.mean(T.sum(batch_pred *target_embs, axis=1))
    batch_distance = T.mean(T.sqrt(1e-8+T.sum((batch_raw_pred-target_embs)**2,axis=1)))
    cos_loss = -T.log(1.0+batch_cosine)#1.0-batch_cosine
    loss = -T.log(1.0/(1.0+batch_distance))


    params = HL_layer_params   # put all model parameters together
    cost=cos_loss+loss#+1e-4*((HL_layer_W3**2).sum())
    updates =   Gradient_Cost_Para(cost,params, learning_rate)

    '''
    testing
    '''


    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([batch_ids, batch_masks, batch_targets], loss, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    # dev_model = theano.function([sents_id_matrix, sents_mask, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([batch_ids, batch_masks, batch_targets], batch_cosine, allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    # n_dev_batches=dev_size/batch_size
    # dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=dev_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[dev_size-batch_size]


    train_indices = range(train_size)
    cost_i=0.0
    max_cosine = 0.0
    while epoch < n_epochs:
        epoch = epoch + 1
        random.Random(100).shuffle(train_indices)
        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]

            cost_i+= train_model(
                                train_sents[train_id_batch],
                                train_masks[train_id_batch],
                                train_target[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if  iter%20==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                test_loss=0.0
                for test_batch_id in test_batch_start: # for each test batch
                    test_loss_i=test_model(
                                dev_sents[test_batch_id:test_batch_id+batch_size],
                                dev_masks[test_batch_id:test_batch_id+batch_size],
                                dev_target[test_batch_id:test_batch_id+batch_size])

                    test_loss+=test_loss_i
                test_loss/=len(test_batch_start)

                if test_loss > max_cosine:
                    max_cosine = test_loss
                print '\t\t\t\t\t\t\t\tcurrent mean_cosin:', test_loss, ' max cosine: ', max_cosine


        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))





if __name__ == '__main__':
    evaluate_lenet5()
