import nltk
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


class MySentences_file(object):
	def __init__(self, filenames):
		self.filenames = filenames
		
	def __iter__(self):
		for filename in self.filenames:
			f= open(filename,'r')
			for line in f:
				doc_arr=[]
				line_list=line.strip().decode('utf-8').split(' ')
				doc_arr.append(line_list[0])
				words=line_list[1:]
				
				print 'doc_id is :',line_list[0]
				
				yield LabeledSentence(words,doc_arr)
			f.close()


def get_doc2vec_file(fnames):
	#get doc2vec model
	
	sentences_file=MySentences_file(fnames)
	model = Doc2Vec(size=300, window=10, min_count=2, workers=7,alpha=0.025, min_alpha=0.025) # use fixed learning rate
	model_dbow=Doc2Vec(size=300, window=10, min_count=2, workers=7,dm=0,alpha=0.025, min_alpha=0.025)
	model.build_vocab(sentences_file)
	model_dbow.build_vocab(sentences_file)
	print 'building vocab is complete'
	
	#model.intersect_word2vec_format('/Users/dcard/Documents/wikidump/ptt.dcard.300.text.bin', binary=True)
	# model.train(sentences_file)
   
	for epoch in range(10):
		model.train(sentences_file)
		model_dbow.train(sentences_file)
		model.alpha -= 0.002  # decrease the learning rate`
		model.min_alpha = model.alpha  # fix the learning rate, no decay
		model_dbow.alpha -= 0.002
		model_dbow.min_alpha= model_dbow.alpha
	print 'training is complete'
	return model,model_dbow


if __name__ == "__main__":
	inp_list=["train_neg.txt","train_pos.txt","test_neg.txt","test_pos.txt"]
	
	model,model_dbow=get_doc2vec_file(inp_list)
	model.save("rtploarity_c_alpha.txt")
	model_dbow.save("rtploarity_c_alpha_dbow.txt")