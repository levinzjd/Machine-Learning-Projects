#import os    
#os.environ['THEANO_FLAGS'] = "device=gpu0"    
import theano
import theano.tensor as T
import numpy as np
import load2
import time


training_data, validation_data, test_data=load2.load_data()

def share(data):
    shared_x = theano.shared(np.asarray(data[0],dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(np.eye(10)[data[1]],dtype=theano.config.floatX))
    return shared_x, shared_y
x,y=share(training_data)
te_x,te_y=share(test_data)

num_train=len(training_data)
#define hyper parameter
reg_lambda=np.float32(0.01) #regularization term
eta=np.float32(0.01) #learning rate

w1_val=np.asarray(np.random.randn(784,100)).astype('float32')
w1=theano.shared(w1_val,'w1')
w2=theano.shared(np.asarray(np.random.randn(100,10).astype('float32')),'w2')
b1=theano.shared(np.zeros(100).astype('float32'),'b1')
b2=theano.shared(np.zeros(10).astype('float32'),'b2')

params=[w1,b1,w2,b2]
z1=T.dot(x,w1)+b1
a1=T.nnet.sigmoid(z1)
z2=T.dot(a1,w2)+b2
#a2=T.nnet.sigmoid(z2)
a2=T.nnet.softmax(z2)

pred_tr = np.argmax(a2,axis=1)
y_arg = np.argmax(y,axis=1)
acc_tr = T.mean(T.eq(pred_tr,y_arg))

cost_reg=(T.sqr(w1).sum()+T.sqr(w2).sum())*reg_lambda/(2*num_train)
cost_tr = T.nnet.categorical_crossentropy(a2,y).mean()+cost_reg

zt1=T.dot(te_x,w1)+b1
at1=T.nnet.sigmoid(zt1)
zt2=T.dot(at1,w2)+b2
#at2=T.nnet.sigmoid(zt2)
at2=T.nnet.softmax(zt2)

cost_te = T.nnet.categorical_crossentropy(at2,te_y).mean()
pred_te = np.argmax(at2,axis=1)
te_y_arg=np.argmax(te_y,axis=1)
acc_te = T.mean(T.eq(pred_te,te_y_arg))

gp = T.grad(cost_tr,params)
updates = [(p,p-eta*g) for p,g in zip(params,gp)]

#i = T.lscalar()

fn=theano.function([],[cost_tr,acc_tr],updates=updates,allow_input_downcast=True)
fn_test=theano.function([],[cost_te,acc_te],allow_input_downcast=True)

def test_run(epochs):
    w1.set_value((np.random.randn(784,100) / np.sqrt(784)).astype('float32'))
    b1.set_value(np.zeros(100).astype('float32'))
    w2.set_value((np.random.randn(100,10) / np.sqrt(100)).astype('float32'))
    b2.set_value(np.zeros(10).astype('float32'))
    for e in range(epochs):
        #for i in range(mbs):
        c,p=fn()
        if e % 1000 == 0:
            print('Epoch:{0} Train Cost:{1} Accuracy:{2}'.format(e,c,p))

    ct,pt=fn_test()
    print('Test Cost:{0} Test Accuracy:{1}'.format(ct,pt))
t0=time.time()
test_run(5000)
t1=time.time()
print
print ('Time used:{0}'.format(t1-t0))
