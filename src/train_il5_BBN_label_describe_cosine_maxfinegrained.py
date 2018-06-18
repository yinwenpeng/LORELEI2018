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

from load_data import  load_fasttext_multiple_word2vec_given_file,load_word2vec_to_init,load_BBN_multi_labels_dataset, load_il6_with_BBN,load_SF_type_descriptions
from common_functions import normalize_matrix_rowwise,normalize_tensor3_colwise,store_model_to_file,Conv_with_Mask, create_conv_para, average_f1_two_array_by_col, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para



def evaluate_lenet5(learning_rate=0.01, n_epochs=100, emb_size=40, batch_size=50, describ_max_len=20, type_size=12,filter_size=[3,5], maxSentLen=100, hidden_size=[300,300]):

    model_options = locals().copy()
    print "model options", model_options
    emb_root = '/save/wenpeng/datasets/LORELEI/multi-lingual-emb/'
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))

    all_sentences, all_masks, all_labels, word2id=load_BBN_multi_labels_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    label_sent, label_mask = load_SF_type_descriptions(word2id, type_size, describ_max_len)
    label_sent=np.asarray(label_sent, dtype='int32')
    label_mask=np.asarray(label_mask, dtype=theano.config.floatX)


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

    vocab_size=  len(word2id)+1 # add one zero pad index

    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_fasttext_multiple_word2vec_given_file([emb_root+'IL5-cca-wiki-lorelei-d40.eng.vec',emb_root+'IL5-cca-wiki-lorelei-d40.IL5.vec'], 40)
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.imatrix('labels')  #batch*12

    des_id_matrix = T.imatrix()
    des_mask = T.fmatrix()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    repeat_common_input = T.repeat(normalize_tensor3_colwise(common_input), type_size, axis=0) #(batch_size*type_size, emb_size, maxsentlen)

    des_input=embeddings[des_id_matrix.flatten()].reshape((type_size,describ_max_len, emb_size)).dimshuffle(0,2,1)
    repeat_des_input = T.tile(normalize_tensor3_colwise(des_input), (batch_size,1,1))#(batch_size*type_size, emb_size, maxsentlen)

    fine_grained_cosine = T.batched_dot(repeat_common_input.dimshuffle(0,2,1),repeat_des_input) #(batch_size*type_size,maxsentlen,describ_max_len)

    fine_grained_cosine_to_matrix = fine_grained_cosine.reshape((batch_size*type_size,maxSentLen*describ_max_len))
    sort_fine_grained_cosine_to_matrix = T.sort(fine_grained_cosine_to_matrix, axis=1)
    top_5_simi = sort_fine_grained_cosine_to_matrix[:,-30:] # (batch_size*type_size, 5)


    max_fine_grained_cosine = T.mean(top_5_simi, axis=1)




    cosine_scores = max_fine_grained_cosine.reshape((batch_size, type_size))
    # score_matrix = T.nnet.sigmoid(cosine_scores) #(batch_size, type_size)
    score_matrix = T.nnet.sigmoid(cosine_scores)

    prob_pos = T.where( labels < 1, 1.0-score_matrix, score_matrix)

    loss = -T.mean(T.log(prob_pos))


    # loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    '''
    do not update emb, so that can generalize from english to ils
    '''
    params = []#+NN_para+LR_para   # put all model parameters together
    cost=loss#+1e-4*((conv_W**2).sum()+(conv_W2**2).sum())
    updates =   Gradient_Cost_Para(cost,params, learning_rate)

    '''
    testing
    '''
    binarize_prob = T.where(cosine_scores > 0.3, 1, 0)

    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([sents_id_matrix, sents_mask, labels, des_id_matrix, des_mask], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    # dev_model = theano.function([sents_id_matrix, sents_mask, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_id_matrix, sents_mask,  des_id_matrix, des_mask], binarize_prob, allow_input_downcast=True, on_unused_input='ignore')

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
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    # max_acc_dev=0.0
    max_meanf1_test=0.0
    max_weightf1_test=0.0
    train_indices = range(train_size)
    cost_i=0.0
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
                                train_labels[train_id_batch],
                                label_sent,
                                label_mask)

            #after each 1000 batches, we test the performance of the model on all test data
            if  iter%20==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                error_sum=0.0
                all_pred_labels = []
                all_gold_labels = []
                for test_batch_id in test_batch_start: # for each test batch
                    pred_labels=test_model(
                                test_sents[test_batch_id:test_batch_id+batch_size],
                                test_masks[test_batch_id:test_batch_id+batch_size],
                                label_sent,
                                label_mask)
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
