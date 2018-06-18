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

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from theano.tensor.nnet.bn import batch_normalization

from load_data import  load_fasttext_multiple_word2vec_given_file,load_word2vec_to_init, load_BBN_multi_labels_dataset
from common_functions import create_LR_para,store_model_to_file,Conv_with_Mask, create_conv_para, average_f1_two_array_by_col, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para



def evaluate_lenet5(learning_rate=0.01, n_epochs=100, emb_size=300, batch_size=50, filter_size=[3,5], maxSentLen=100, hidden_size=[300,300]):

    model_options = locals().copy()
    print "model options", model_options
    emb_root = '/save/wenpeng/datasets/LORELEI/multi-lingual-emb/'
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))

    all_sentences, all_masks, all_labels, word2id=load_BBN_multi_labels_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents=np.asarray(all_sentences[0], dtype='int32')
    train_masks=np.asarray(all_masks[0], dtype=theano.config.floatX)
    train_labels=np.asarray(all_labels[0], dtype='int32')
    train_size=len(train_labels)

    dev_sents=np.asarray(all_sentences[1], dtype='int32')
    dev_masks=np.asarray(all_masks[1], dtype=theano.config.floatX)
    dev_labels=np.asarray(all_labels[1], dtype='int32')
    dev_size=len(dev_labels)

    test_sents=np.asarray(all_sentences[2], dtype='int32')
    test_masks=np.asarray(all_masks[2], dtype=theano.config.floatX)
    test_labels=np.asarray(all_labels[2], dtype='int32')
    test_size=len(test_labels)

    comb_sents = np.concatenate([train_sents,test_sents], axis=0)
    comb_masks = np.concatenate([train_masks,test_masks], axis=0)
    comb_labels = np.asarray([0]*train_size+[1]*test_size, dtype='int32')
    comb_size = len(comb_labels)

    vocab_size=  len(word2id)+1 # add one zero pad index

    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_fasttext_multiple_word2vec_given_file([emb_root+'wiki.en.vec',emb_root+'mono-lingual-il5-xinli.vec'], 300)
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.imatrix('labels')  #batch*12
    domain_labels = T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    bow_emb = T.sum(common_input*sents_mask.dimshuffle(0,'x',1),axis=2)
    # bow_mean_emb = bow_emb/T.sum(sents_mask,axis=1).dimshuffle(0,'x')



    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W2, conv_b2=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]))
    NN_para = [conv_W, conv_b, conv_W2, conv_b2]

    conv_model = Conv_with_Mask(rng, input_tensor3=common_input,
             mask_matrix = sents_mask,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings=conv_model.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    conv_model2 = Conv_with_Mask(rng, input_tensor3=common_input,
             mask_matrix = sents_mask,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]), W=conv_W2, b=conv_b2)    #mutiple mask with the conv_out to set the features by UNK to zero
    sent_embeddings2=conv_model2.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    '''
    adversarial
    '''
    domain_input = T.concatenate([sent_embeddings,sent_embeddings2, bow_emb], axis=1)
    domain_input_size = hidden_size[0]*2+emb_size
    HL_layer_1_W, HL_layer_1_b = create_HiddenLayer_para(rng, domain_input_size, hidden_size[0])
    HL_layer_2_W, HL_layer_2_b = create_HiddenLayer_para(rng, hidden_size[0], hidden_size[0])
    adver_HL_para = [HL_layer_1_W, HL_layer_1_b, HL_layer_2_W, HL_layer_2_b]
    HL_layer_1=HiddenLayer(rng, input=domain_input, n_in=domain_input_size, n_out=hidden_size[0], W=HL_layer_1_W, b=HL_layer_1_b, activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[0], W=HL_layer_2_W, b=HL_layer_2_b, activation=T.tanh)
    disc_para_W, disc_para_b = create_LR_para(rng,hidden_size[0],2)
    conf_para_W, conf_para_b = create_LR_para(rng,hidden_size[0],2)
    conf_para_W2, conf_para_b2 = create_LR_para(rng,hidden_size[0],2)
    adver_LR_para = [disc_para_W, disc_para_b, conf_para_W, conf_para_b,conf_para_W2, conf_para_b2]
    disc_layer_LR=LogisticRegression(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=2, W=disc_para_W, b=disc_para_b)
    conf_layer_LR=LogisticRegression(rng, input=HL_layer_2.output, n_in=hidden_size[0], n_out=2, W=conf_para_W, b=conf_para_b)
    conf2_layer_LR=LogisticRegression(rng, input=HL_layer_2.output, n_in=hidden_size[0], n_out=2, W=conf_para_W2, b=conf_para_b2)
    disc_loss = disc_layer_LR.negative_log_likelihood(domain_labels)
    conf_loss = conf_layer_LR.negative_log_likelihood_specific_label(0)
    conf_loss2 = conf2_layer_LR.negative_log_likelihood_specific_label(1)
    adver_loss = disc_loss+conf_loss+conf_loss2
    adver_para = adver_HL_para+adver_LR_para   +NN_para
    adver_updates =   Gradient_Cost_Para(adver_loss,adver_para, learning_rate)

    '''
    SF classification
    '''
    LR_input = HL_layer_2.output#T.concatenate([sent_embeddings,sent_embeddings2, bow_emb], axis=1)
    LR_input_size = hidden_size[0]
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    U_a = create_ensemble_para(rng, 12, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((12,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=12, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    score_matrix = T.nnet.sigmoid(layer_LR.before_softmax)  #batch * 12
    prob_pos = T.where( labels < 1, 1.0-score_matrix, score_matrix)

    loss = -T.mean(T.log(prob_pos))


    # loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    params = NN_para+adver_HL_para+LR_para   # put all model parameters together
    cost=loss+1e-4*((conv_W**2).sum()+(conv_W2**2).sum())
    updates =   Gradient_Cost_Para(cost,params, learning_rate)

    '''
    testing
    '''
    binarize_prob = T.where(score_matrix > 0.3, 1, 0)

    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    comb_model = theano.function([sents_id_matrix, sents_mask, domain_labels], adver_loss, updates=adver_updates, allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_id_matrix, sents_mask], binarize_prob, allow_input_downcast=True, on_unused_input='ignore')

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
    n_comb_batches=comb_size/batch_size
    comb_batch_start=list(np.arange(n_comb_batches)*batch_size)+[comb_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    # max_acc_dev=0.0
    max_meanf1_test=0.0
    max_weightf1_test=0.0
    train_indices = range(train_size)
    comb_indices = range(comb_size)
    cost_i=0.0
    adver_cost_i=0.0
    while epoch < n_epochs:
        epoch = epoch + 1
        random.Random(100).shuffle(train_indices)
        random.Random(100).shuffle(comb_indices)
        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]

            cost_i+= train_model(
                                train_sents[train_id_batch],
                                train_masks[train_id_batch],
                                train_labels[train_id_batch])

            comb_id_batch = comb_indices[batch_id:batch_id+batch_size]
            adver_cost_i+= comb_model(
                                comb_sents[comb_id_batch],
                                comb_masks[comb_id_batch],
                                comb_labels[comb_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if  iter%20==0:
                print 'Epoch ', epoch, 'iter ', iter, ' average cost: ', cost_i/iter , adver_cost_i/iter , 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                error_sum=0.0
                all_pred_labels = []
                all_gold_labels = []
                for test_batch_id in test_batch_start: # for each test batch
                    pred_labels=test_model(
                                test_sents[test_batch_id:test_batch_id+batch_size],
                                test_masks[test_batch_id:test_batch_id+batch_size])
                    gold_labels = test_labels[test_batch_id:test_batch_id+batch_size]
                    # print 'pred_labels:', pred_labels
                    # print 'gold_labels;', gold_labels
                    all_pred_labels.append(pred_labels)
                    all_gold_labels.append(gold_labels)
                all_pred_labels = np.concatenate(all_pred_labels)
                all_gold_labels = np.concatenate(all_gold_labels)


                test_mean_f1, test_weight_f1 =average_f1_two_array_by_col(all_pred_labels, all_gold_labels)
                if test_weight_f1 > max_weightf1_test:
                    max_weightf1_test=test_weight_f1
                if test_mean_f1 > max_meanf1_test:
                    max_meanf1_test=test_mean_f1
                print '\t\t\t\t\t\t\t\tcurrent f1s:', test_mean_f1, test_weight_f1, '\t\tmax_f1:', max_meanf1_test, max_weightf1_test


        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))





if __name__ == '__main__':
    evaluate_lenet5()
