from gensim.models import Doc2Vec
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
from bow import bow
fileroot = os.path.dirname(os.path.abspath(__file__))

class feature:
	def __init__(self,model_name,model_dbow_name):
	#model = Doc2Vec.load(os.path.join(fileroot,'imdb.txt'))
		self.model = Doc2Vec.load(os.path.join(fileroot,model_name))
		self.model_dbow= Doc2Vec.load(os.path.join(fileroot,model_dbow_name))

	#semantic orientation
	def SO(self,tag,dm=False):
		pos_set=['excellent','outstanding','amazing']
		neg_set=['disappointing','poor']
		print self.model_dbow.docvecs[pos_set]
		if dm is True:
			result=[self.model.n_similarity(tag,pos_set)
			-self.model.n_similarity(tag,neg_set)]
			return np.array(result)
		else:
			result=[self.model_dbow.n_similarity(tag,pos_set)
			-self.model_dbow.n_similarity(tag,neg_set)]
			return np.array(result)
	

	def feature_select(self,so=False,bow_m=False):
		trainX=[] #training examples
		trainY=[] #labels
		model=self.model
		model_dbow=self.model_dbow
		if bow_m is True:
			b=bow()
			train_bow=b.bow_create("train")
			test_bow=b.bow_create("test")
		for i in range(3000):
			train_pos='train_pos'+str(i)
			# trainX.append(model.docvecs[train_pos]+model_dbow.docvecs[train_pos]+SO(model_dbow,train_pos))
			
			if bow_m is True:
				pos_idx=i
				pos_bow=train_bow[i][:]
				trainX.append(np.concatenate((model.docvecs[train_pos],model_dbow.docvecs[train_pos],pos_bow)))
			elif so is True:
				print train_pos
				trainX.append(np.concatenate((model.docvecs[train_pos],model_dbow.docvecs[train_pos],self.SO(train_pos))))
			else:
				trainX.append(np.concatenate((model.docvecs[train_pos],model_dbow.docvecs[train_pos])))
				
				
			# print 'SO postive is:',SO(model,train_pos)
			# print len(trainX)
			# print trainX[0].shape
			
			trainY.append(1)
			
			train_neg='train_neg'+str(i)
			if bow_m is True:
				neg_idx=i+3000
				neg_bow=train_bow[neg_idx][:]
				trainX.append(np.concatenate((model.docvecs[train_neg],model_dbow.docvecs[train_neg],neg_bow)))
			elif so is True:
				trainX.append(np.concatenate((model.docvecs[train_neg],model_dbow.docvecs[train_neg],self.SO(train_neg))))
			else:
				trainX.append(np.concatenate((model.docvecs[train_neg],model_dbow.docvecs[train_neg])))
				
			# trainX.append(model.docvecs[train_neg]+model_dbow.docvecs[train_neg]+SO(model_dbow,train_neg))
			trainY.append(0)
			# print 'SO negative is:',SO(model,train_neg)
			#raw_input()

		
		testX=[]
		testY=[]
		for i in range(2331):
			test_pos='test_pos'+str(i)
			if bow_m is True:
				# testX.append(model.docvecs[test_pos]+model_dbow.docvecs[test_pos]+SO(model_dbow,test_pos))
				pos_bow=test_bow[i][:]
				testX.append(np.concatenate((model.docvecs[test_pos],model_dbow.docvecs[test_pos],pos_bow)))
			elif so is True:
				testX.append(np.concatenate((model.docvecs[test_pos],model_dbow.docvecs[test_pos],self.SO(test_pos))))
			else:
				testX.append(np.concatenate((model.docvecs[test_pos],model_dbow.docvecs[test_pos])))
				
			testY.append(1)
			test_neg='test_neg'+str(i)
			if bow_m is True:
			# testX.append(model.docvecs[test_neg]+model_dbow.docvecs[test_neg]+SO(model_dbow,test_neg))
				neg_idx=i+2331
				neg_bow=test_bow[neg_idx][:]
				testX.append(np.concatenate
					((model.docvecs[test_neg],model_dbow.docvecs[test_neg],neg_bow)))
			elif so is True:
				testX.append(np.concatenate((model.docvecs[test_neg],model_dbow.docvecs[test_neg],self.SO(test_neg))))
			else:
				testX.append(np.concatenate((model.docvecs[test_neg],model_dbow.docvecs[test_neg])))
				
			testY.append(0)
		#release memory
		train_bow=[]
		test_bow=[]
		model=[]
		model_dbow=[]
		return trainX,trainY,testX,testY

if __name__ == "__main__":
	f=feature("rtploarity_c_alpha.txt","rtploarity_c_alpha_dbow.txt")
	trainX,trainY,testX,testY=f.feature_select(so=True)
	clf = svm.SVC()
	clf.fit(trainX, trainY)
	print clf.score(testX,testY)
	clf2 = MLPClassifier()
	clf2.fit(trainX, trainY)
	clf3 = LogisticRegression()
	clf3.fit(trainX, trainY)
	print clf2.score(testX,testY)
	print clf3.score(testX,testY)