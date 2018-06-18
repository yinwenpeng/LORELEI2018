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

from load_data import  load_fasttext_word2vec_given_file,load_word2vec_to_init, load_il6_with_BBN,load_SF_type_descriptions
from common_functions import normalize_matrix_rowwise,store_model_to_file,Conv_with_Mask, create_conv_para, average_f1_two_array_by_col, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para



def evaluate_lenet5(learning_rate=0.01, n_epochs=100, emb_size=40, batch_size=50, describ_max_len=20, type_size=12,filter_size=[3,5], maxSentLen=100, hidden_size=[300,300]):

    model_options = locals().copy()
    print "model options", model_options
    emb_root = '/save/wenpeng/datasets/LORELEI/multi-lingual-emb/'
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))

    all_sentences, all_masks, all_labels, word2id=load_il6_with_BBN(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
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
    word2vec=load_fasttext_word2vec_given_file([emb_root+'IL6-cca-wiki-lorelei-d40.eng.vec',emb_root+'IL6-cca-wiki-lorelei-d40.IL6.vec'], 40)
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.imatrix('labels')  #batch*12

    des_id_matrix = T.imatrix()
    des_mask = T.imatrix()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    bow_emb = T.sum(common_input*sents_mask.dimshuffle(0,'x',1),axis=2)

    des_input=embeddings[des_id_matrix.flatten()].reshape((type_size,describ_max_len, emb_size)).dimshuffle(0,2,1)
    bow_des = T.sum(des_input*des_mask.dimshuffle(0,'x',1),axis=2) #(tyope_size, emb_size)


    # conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    # conv_W2, conv_b2=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]))
    # NN_para = [conv_W, conv_b, conv_W2, conv_b2]
    #
    # conv_model = Conv_with_Mask(rng, input_tensor3=common_input,
    #          mask_matrix = sents_mask,
    #          image_shape=(batch_size, 1, emb_size, maxSentLen),
    #          filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]), W=conv_W, b=conv_b)    #mutiple mask with the conv_out to set the features by UNK to zero
    # sent_embeddings=conv_model.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    #
    # conv_model2 = Conv_with_Mask(rng, input_tensor3=common_input,
    #          mask_matrix = sents_mask,
    #          image_shape=(batch_size, 1, emb_size, maxSentLen),
    #          filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]), W=conv_W2, b=conv_b2)    #mutiple mask with the conv_out to set the features by UNK to zero
    # sent_embeddings2=conv_model2.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
    #
    # LR_input = T.concatenate([sent_embeddings,sent_embeddings2, bow_emb], axis=1)
    # LR_input_size = hidden_size[0]*2+emb_size
    # #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    # U_a = create_ensemble_para(rng, 12, LR_input_size) # the weight matrix hidden_size*2
    # LR_b = theano.shared(value=np.zeros((12,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    # LR_para=[U_a, LR_b]
    # layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=12, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector


    score_matrix = T.nnet.sigmoid(normalize_matrix_rowwise(bow_emb).dot(normalize_matrix_rowwise(bow_des).T)) #(batch_size, type_size)
    # score_matrix = T.nnet.sigmoid(layer_LR.before_softmax)  #batch * 12
    prob_pos = T.where( labels < 1, 1.0-score_matrix, score_matrix)

    loss = -T.mean(T.log(prob_pos))


    # loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    params = [embeddings]#+NN_para+LR_para   # put all model parameters together
    cost=loss#+1e-4*((conv_W**2).sum()+(conv_W2**2).sum())
    updates =   Gradient_Cost_Para(cost,params, learning_rate)

    '''
    testing
    '''
    binarize_prob = T.where(score_matrix > 0.3, 1, 0)

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

    return max_acc_test



if __name__ == '__main__':
    evaluate_lenet5()
