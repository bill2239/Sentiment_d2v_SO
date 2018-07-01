import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords 
import numpy as np
import os
import re
fileroot = os.path.dirname(os.path.abspath(__file__))
class bow:
	def review_to_words(self, reviews_filename ):
		clean_reviews=[]
		file=open(reviews_filename,'r')
		for line in file.readlines():
			line_list=line.strip().decode('utf8').split(' ')
			review_text=" ".join(line_list[1:])
			letters_only = re.sub("[^a-zA-Z]", " ", review_text)
			words = letters_only.lower().split()
			stops = set(stopwords.words("english"))
			meaningful_words = [w for w in words if not w in stops]
			clean_review=" ".join( meaningful_words )
			clean_reviews.append(clean_review)
		return clean_reviews
	def bow_create(self,string):
		file_pos=string+"_pos.txt"
		file_neg=string+"_neg.txt"
		clean_train_reviews=self.review_to_words(file_pos)+self.review_to_words(file_neg)
		print len(clean_train_reviews)
		vectorizer = CountVectorizer(analyzer = "word",   
		                             tokenizer = None,    
		                             preprocessor = None, 
		                             stop_words = None,   
		                             max_features = 5000) 
		bow=vectorizer.fit_transform(clean_train_reviews)
		bow = bow.toarray()
		print bow.shape
		return bow
# b=bow()
# data=b.bow_create("train")

# print data.shape
# print data[1][:]
# print data[1][:].shape