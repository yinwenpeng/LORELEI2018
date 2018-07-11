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

from load_data import  load_trainingData_types,load_trainingData_types_plus_others,load_official_testData_only_il,load_fasttext_multiple_word2vec_given_file,load_word2vec_to_init, load_BBN_il5Trans_il5_dataset,load_SF_type_descriptions
from common_functions import create_LR_para,normalize_matrix_rowwise,normalize_tensor3_colwise,store_model_to_file,Conv_with_Mask, create_conv_para, average_f1_two_array_by_col, create_HiddenLayer_para, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, GRU_Batch_Tensor_Input_with_Mask,Gradient_Cost_Para, Attentive_Conv_for_Pair, create_GRU_para,create_LSTM_para
from preprocess_common import generate_2018_official_output


def evaluate_lenet5(learning_rate=0.01, n_epochs=4, emb_size=300, batch_size=50, describ_max_len=20, type_size=12,filter_size=[3,5], maxSentLen=100, hidden_size=[300,300]):

    model_options = locals().copy()
    print "model options", model_options
    emb_root = '/save/wenpeng/datasets/LORELEI/multi-lingual-emb/2018-il9-il10/multi-emb/'
    test_file_path = '/save/wenpeng/datasets/LORELEI/il9/il9-setE-as-test-input_ner_filtered_w2.txt'
    output_file_path = '/save/wenpeng/datasets/LORELEI/il9/il9_system_output_noMT_BBN_NI_epoch4.json'
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results
    srng = T.shared_randomstreams.RandomStreams(rng.randint(seed))
    word2id={}
    # all_sentences, all_masks, all_labels, all_other_labels, word2id=load_BBN_il5Trans_il5_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_p1_sents, train_p1_masks, train_p1_labels,word2id = load_trainingData_types(word2id, maxSentLen)
    train_p2_sents, train_p2_masks, train_p2_labels, train_p2_other_labels,word2id = load_trainingData_types_plus_others(word2id, maxSentLen)
    test_sents, test_masks, test_lines,word2id = load_official_testData_only_il(word2id, maxSentLen, test_file_path)

    label_sent, label_mask = load_SF_type_descriptions(word2id, type_size, describ_max_len)
    label_sent=np.asarray(label_sent, dtype='int32')
    label_mask=np.asarray(label_mask, dtype=theano.config.floatX)


    train_p1_sents=np.asarray(train_p1_sents, dtype='int32')
    train_p1_masks=np.asarray(train_p1_masks, dtype=theano.config.floatX)
    train_p1_labels=np.asarray(train_p1_labels, dtype='int32')
    train_p1_size=len(train_p1_labels)

    train_p2_sents=np.asarray(train_p2_sents, dtype='int32')
    train_p2_masks=np.asarray(train_p2_masks, dtype=theano.config.floatX)
    train_p2_labels=np.asarray(train_p2_labels, dtype='int32')
    train_p2_other_labels = np.asarray(train_p2_other_labels, dtype='int32')
    train_p2_size=len(train_p2_labels)
    '''
    combine train_p1 and train_p2
    '''
    train_sents=np.concatenate([train_p1_sents,train_p2_sents],axis=0)
    train_masks=np.concatenate([train_p1_masks,train_p2_masks],axis=0)
    train_labels=np.concatenate([train_p1_labels,train_p2_labels],axis=0)
    train_size=train_p1_size+train_p2_size

    test_sents=np.asarray(test_sents, dtype='int32')
    test_masks=np.asarray(test_masks, dtype=theano.config.floatX)
    # test_labels=np.asarray(all_labels[2], dtype='int32')
    test_size=len(test_sents)

    vocab_size=  len(word2id)+1 # add one zero pad index

    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_fasttext_multiple_word2vec_given_file([emb_root+'100k-ENG-multicca.300.ENG.txt',emb_root+'100k-IL9-multicca.d300.IL9.txt'], 300)
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.imatrix('labels')  #batch*12
    other_labels = T.imatrix()  #batch*4

    des_id_matrix = T.imatrix()
    des_mask = T.fmatrix()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    bow_emb = T.sum(common_input*sents_mask.dimshuffle(0,'x',1),axis=2)
    repeat_common_input = T.repeat(normalize_tensor3_colwise(common_input), type_size, axis=0) #(batch_size*type_size, emb_size, maxsentlen)


    des_input=embeddings[des_id_matrix.flatten()].reshape((type_size,describ_max_len, emb_size)).dimshuffle(0,2,1)
    bow_des = T.sum(des_input*des_mask.dimshuffle(0,'x',1),axis=2) #(tyope_size, emb_size)
    repeat_des_input = T.tile(normalize_tensor3_colwise(des_input), (batch_size,1,1))#(batch_size*type_size, emb_size, maxsentlen)


    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W2, conv_b2=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]))
    multiCNN_para = [conv_W, conv_b, conv_W2, conv_b2]

    conv_att_W, conv_att_b=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))
    conv_att_W2, conv_att_b2=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]))
    conv_W_context2, conv_b_context2=create_conv_para(rng, filter_shape=(hidden_size[0], 1, emb_size, 1))
    ACNN_para = [conv_att_W, conv_att_b,conv_W_context,conv_att_W2, conv_att_b2,conv_W_context2]

    '''
    multi-CNN
    '''
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
    GRU
    '''
    U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size[0])
    GRU_NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
    # gru_input = common_input.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
    gru_layer=GRU_Batch_Tensor_Input_with_Mask(common_input, sents_mask,  hidden_size[0], U1, W1, b1)
    gru_sent_embeddings=gru_layer.output_sent_rep  # (batch_size, hidden_size)



    '''
    ACNN
    '''
    attentive_conv_layer = Attentive_Conv_for_Pair(rng,
            origin_input_tensor3=common_input,
            origin_input_tensor3_r = common_input,
            input_tensor3=common_input,
            input_tensor3_r = common_input,
             mask_matrix = sents_mask,
             mask_matrix_r = sents_mask,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             image_shape_r = (batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[0], 1,emb_size, 1),
             W=conv_att_W, b=conv_att_b,
             W_context=conv_W_context, b_context=conv_b_context)
    sent_att_embeddings = attentive_conv_layer.attentive_maxpool_vec_l

    attentive_conv_layer2 = Attentive_Conv_for_Pair(rng,
            origin_input_tensor3=common_input,
            origin_input_tensor3_r = common_input,
            input_tensor3=common_input,
            input_tensor3_r = common_input,
             mask_matrix = sents_mask,
             mask_matrix_r = sents_mask,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             image_shape_r = (batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[0], 1, emb_size, filter_size[1]),
             filter_shape_context=(hidden_size[0], 1,emb_size, 1),
             W=conv_att_W2, b=conv_att_b2,
             W_context=conv_W_context2, b_context=conv_b_context2)
    sent_att_embeddings2 = attentive_conv_layer2.attentive_maxpool_vec_l

    '''
    cross-DNN-dataless
    '''
    #first map label emb into hidden space
    HL_layer_1_W, HL_layer_1_b = create_HiddenLayer_para(rng, emb_size, hidden_size[0])
    HL_layer_1_params = [HL_layer_1_W, HL_layer_1_b]
    HL_layer_1=HiddenLayer(rng, input=bow_des, n_in=emb_size, n_out=hidden_size[0], W=HL_layer_1_W, b=HL_layer_1_b, activation=T.tanh)
    des_rep_hidden = HL_layer_1.output #(type_size, hidden_size)
    dot_dnn_dataless_1 = T.tanh(sent_embeddings.dot(des_rep_hidden.T)) #(batch_size, type_size)
    dot_dnn_dataless_2 = T.tanh(sent_embeddings2.dot(des_rep_hidden.T))


    '''
    dataless cosine
    '''
    cosine_scores = normalize_matrix_rowwise(bow_emb).dot(normalize_matrix_rowwise(bow_des).T)
    cosine_score_matrix = T.nnet.sigmoid(cosine_scores) #(batch_size, type_size)

    '''
    dataless top-30 fine grained cosine
    '''
    fine_grained_cosine = T.batched_dot(repeat_common_input.dimshuffle(0,2,1),repeat_des_input) #(batch_size*type_size,maxsentlen,describ_max_len)
    fine_grained_cosine_to_matrix = fine_grained_cosine.reshape((batch_size*type_size,maxSentLen*describ_max_len))
    sort_fine_grained_cosine_to_matrix = T.sort(fine_grained_cosine_to_matrix, axis=1)
    top_k_simi = sort_fine_grained_cosine_to_matrix[:,-30:] # (batch_size*type_size, 5)
    max_fine_grained_cosine = T.mean(top_k_simi, axis=1)
    top_k_cosine_scores = max_fine_grained_cosine.reshape((batch_size, type_size))
    top_k_score_matrix = T.nnet.sigmoid(top_k_cosine_scores)





    acnn_LR_input = T.concatenate([dot_dnn_dataless_1, dot_dnn_dataless_2,cosine_score_matrix,top_k_score_matrix,sent_embeddings,sent_embeddings2, gru_sent_embeddings,sent_att_embeddings,sent_att_embeddings2, bow_emb], axis=1)
    acnn_LR_input_size = hidden_size[0]*5+emb_size+4*type_size
    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    acnn_U_a, acnn_LR_b = create_LR_para(rng,acnn_LR_input_size, 12)
    acnn_LR_para=[acnn_U_a, acnn_LR_b]
    acnn_layer_LR=LogisticRegression(rng, input=acnn_LR_input, n_in=acnn_LR_input_size, n_out=12, W=acnn_U_a, b=acnn_LR_b) #basically it is a multiplication between weight matrix and input feature vector
    acnn_score_matrix = T.nnet.sigmoid(acnn_layer_LR.before_softmax)  #batch * 12
    acnn_prob_pos = T.where( labels < 1, 1.0-acnn_score_matrix, acnn_score_matrix)
    acnn_loss = -T.mean(T.log(acnn_prob_pos))

    acnn_other_U_a, acnn_other_LR_b = create_LR_para(rng,acnn_LR_input_size, 16)
    acnn_other_LR_para=[acnn_other_U_a, acnn_other_LR_b]
    acnn_other_layer_LR=LogisticRegression(rng, input=acnn_LR_input, n_in=acnn_LR_input_size, n_out=16, W=acnn_other_U_a, b=acnn_other_LR_b)
    acnn_other_prob_matrix = T.nnet.softmax(acnn_other_layer_LR.before_softmax.reshape((batch_size*4,4)) )
    acnn_other_prob_tensor3 = acnn_other_prob_matrix.reshape((batch_size, 4, 4))
    acnn_other_prob = acnn_other_prob_tensor3[T.repeat(T.arange(batch_size), 4), T.tile(T.arange(4), (batch_size)), other_labels.flatten()]
    acnn_other_field_loss = -T.mean(T.log(acnn_other_prob))




    params = multiCNN_para  + GRU_NN_para   +ACNN_para +acnn_LR_para + HL_layer_1_params# put all model parameters together
    cost=acnn_loss+   1e-4*((conv_W**2).sum()+(conv_W2**2).sum()+(conv_att_W**2).sum()+(conv_att_W2**2).sum())
    updates =   Gradient_Cost_Para(cost,params, learning_rate)


    other_paras = params+acnn_other_LR_para
    cost_other = cost  + acnn_other_field_loss
    other_updates =   Gradient_Cost_Para(cost_other,other_paras, learning_rate)

    '''
    testing
    '''

    ensemble_NN_scores = acnn_score_matrix#T.max(T.concatenate([att_score_matrix.dimshuffle('x',0,1), score_matrix.dimshuffle('x',0,1), acnn_score_matrix.dimshuffle('x',0,1)],axis=0),axis=0)
    # '''
    # majority voting, does not work
    # '''
    # binarize_NN = T.where(ensemble_NN_scores > 0.5, 1, 0)
    # binarize_dataless = T.where(cosine_score_matrix > 0.5, 1, 0)
    # binarize_dataless_finegrained = T.where(top_k_score_matrix > 0.5, 1, 0)
    # binarize_conc =  T.concatenate([binarize_NN.dimshuffle('x',0,1), binarize_dataless.dimshuffle('x',0,1),binarize_dataless_finegrained.dimshuffle('x',0,1)],axis=0)
    # sum_binarize_conc = T.sum(binarize_conc,axis=0)
    # binarize_prob = T.where(sum_binarize_conc > 0.0, 1, 0)
    # '''
    # sum up prob, works
    # '''
    # ensemble_scores_1 = 0.6*ensemble_NN_scores+0.4*top_k_score_matrix
    # binarize_prob = T.where(ensemble_scores_1 > 0.3, 1, 0)
    '''
    sum up prob, works
    '''
    ensemble_scores = ensemble_NN_scores#0.6*ensemble_NN_scores+0.4*0.5*(cosine_score_matrix+top_k_score_matrix)
    binarize_prob = T.where(ensemble_scores > 0.3, 1, 0)
    '''
    test for other fields
    '''
    sum_tensor3 = acnn_other_prob_tensor3 #(batch, 4, 3)

    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_p1_model = theano.function([sents_id_matrix, sents_mask, labels, des_id_matrix, des_mask], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    train_p2_model = theano.function([sents_id_matrix, sents_mask, labels,des_id_matrix, des_mask,other_labels], cost_other, updates=other_updates,allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_id_matrix, sents_mask, des_id_matrix, des_mask], [binarize_prob,ensemble_scores,sum_tensor3], allow_input_downcast=True, on_unused_input='ignore')

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
    n_train_p2_batches=train_p2_size/batch_size
    train_p2_batch_start=list(np.arange(n_train_p2_batches)*batch_size)+[train_p2_size-batch_size]
    n_test_batches=test_size/batch_size
    n_test_remain=test_size%batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]

    train_p2_batch_start_set = set(train_p2_batch_start)
    # max_acc_dev=0.0
    # max_meanf1_test=0.0
    # max_weightf1_test=0.0
    train_indices = range(train_size)
    train_p2_indices = range(train_p2_size)
    cost_i=0.0
    other_cost_i = 0.0
    min_mean_frame = 100.0
    while epoch < n_epochs:
        epoch = epoch + 1
        random.Random(100).shuffle(train_indices)
        random.Random(100).shuffle(train_p2_indices)
        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]

            cost_i+= train_p1_model(
                                train_sents[train_id_batch],
                                train_masks[train_id_batch],
                                train_labels[train_id_batch],
                                label_sent,
                                label_mask)

            if batch_id in train_p2_batch_start_set:
                train_p2_id_batch = train_p2_indices[batch_id:batch_id+batch_size]
                other_cost_i+=train_p2_model(
                                    train_p2_sents[train_p2_id_batch],
                                    train_p2_masks[train_p2_id_batch],
                                    train_p2_labels[train_p2_id_batch],
                                    label_sent,
                                    label_mask,
                                    train_p2_other_labels[train_p2_id_batch]
                                    )
            # else:
            #     random_batch_id = random.choice(train_p2_batch_start)
            #     train_p2_id_batch = train_p2_indices[random_batch_id:random_batch_id+batch_size]
            #     other_cost_i+=train_p2_model(
            #                         train_p2_sents[train_p2_id_batch],
            #                         train_p2_masks[train_p2_id_batch],
            #                         train_p2_labels[train_p2_id_batch],
            #                         label_sent,
            #                         label_mask,
            #                         train_p2_other_labels[train_p2_id_batch]
            #                         )
            #after each 1000 batches, we test the performance of the model on all test data
            if  iter%20==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter),str(other_cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                pred_types = []
                pred_confs = []
                pred_others = []
                for i, test_batch_id in enumerate(test_batch_start): # for each test batch
                    pred_types_i, pred_conf_i, pred_fields_i=test_model(
                                test_sents[test_batch_id:test_batch_id+batch_size],
                                test_masks[test_batch_id:test_batch_id+batch_size],
                                label_sent,
                                label_mask
                                )
                    if i < len(test_batch_start)-1:
                        pred_types.append(pred_types_i)
                        pred_confs.append(pred_conf_i)
                        pred_others.append(pred_fields_i)
                    else:
                        pred_types.append(pred_types_i[-n_test_remain:])
                        pred_confs.append(pred_conf_i[-n_test_remain:])
                        pred_others.append(pred_fields_i[-n_test_remain:])
                pred_types = np.concatenate(pred_types, axis=0)
                pred_confs = np.concatenate(pred_confs, axis=0)
                pred_others = np.concatenate(pred_others, axis=0)
                mean_frame = generate_2018_official_output(test_lines, output_file_path, pred_types, pred_confs, pred_others, min_mean_frame)
                if mean_frame < min_mean_frame:
                    min_mean_frame = mean_frame
                print '\t\t\t test  over, min_mean_frame:', min_mean_frame




        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




if __name__ == '__main__':
    evaluate_lenet5()
