import numpy as np
import theano
import theano.tensor as T


sents_id_matrix=T.imatrix() #(2, 4)
# repeat_common_input = T.repeat(sents_id_matrix, 3, axis=0)
# output = repeat_common_input.reshape((2*3,3))
output = T.tile(sents_id_matrix, (3,1))

train_model = theano.function([sents_id_matrix], output)

if __name__ == '__main__':
    cost_i= train_model(np.asarray([[1,2,3,4],[5,6,7,8]], dtype='int32'))
    print cost_i
