#import os    
#os.environ['THEANO_FLAGS'] = "device=gpu0"    
import theano
import theano.tensor as T
import numpy as np
import load

training_data, validation_data, test_data=load.load_data_wrapper()


class mlp(object):
	def __init__(self):
	    #MLP
		w1_val=np.asarray(np.random.randn(100, 784))
		w1=theano.shared(w1_val,'w1')
		w2=theano.shared(np.asarray(np.random.randn(10, 100)),'w2')
		b1=theano.shared(np.asarray(np.random.randn(100, 1)),'b1')
		b2=theano.shared(np.asarray(np.random.randn(10, 1)),'b2')


		x=T.matrix('x')
		y=T.matrix('y')
		params=[w1,b1,w2,b2]
		a1=T.nnet.sigmoid(T.dot(w1,x)+b1)
		a2=T.nnet.sigmoid(T.dot(w2,a1)+b2)
		pred = np.argmax(a2)
		xent = -y*T.log(a2) - (1-y)*T.log(1-a2)
		cost= xent.mean()
		gp = T.grad(cost,params)
		updates = [(p,p-0.05*g) for p,g in zip(params,gp)]

		self.fn=theano.function([x,y],[cost,pred],updates=updates,allow_input_downcast=True)
		self.fn_test=theano.function([x,y],[cost,pred],allow_input_downcast=True)
	
	def test_run(self,epochs,training_data,test_data):
		l=50000#len(training_data)
		lt=10000#len(test_data)
		cost_t=0
		correct_t=0
		train_cost=[]
		train_acc=[]
		for e in range(epochs):
			cost=0
			correct=0
			for x,y in training_data:
				c,p=self.fn(x,y)
				cost+=c
				if p==np.argmax(y):
					correct+=1
			train_cost.append(cost)
			tac=float(correct)/l
			train_acc.append(tac)
			print('Epoch:{0} Train Cost:{1} Accuracy:{2}'.format(e+1,cost,tac))
		for x,y in test_data:
			ct,pt=self.fn_test(x,y)
			cost_t+=ct
			if pt==np.argmax(y):
				correct_t+=1
		print('Test Cost:{0} Test Accuracy:{1}'.format(cost_t,(float(correct_t)/lt)))

	import matplotlib.pyplot as plt
	plt.plot(range(epochs),train_cost)
	plt.title("Training cost by epochs")
	plt.show()
	plt.plot(range(epochs),train_acc)
	plt.title("Training accuracy by epochs")
	plt.show()

