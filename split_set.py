# coding=utf-8
import random
import os
fileroot = "./data"
indexes= [i for i in range(1,5331)]
train_idx=random.sample(indexes,3000)
#filenames=["rt-polarity.neg","rt-polarity.pos"]
filename="rt-polarity.neg"
file_path=os.path.join(fileroot,filename)
fin_neg=open(file_path,'r')
count=1
fout_train=open("train_neg.txt",'w')
fout_test=open("test_neg.txt",'w')
i_train=0
i_test=0
for line in fin_neg.readlines():

	if count in train_idx:
		prefix="train_neg"+str(i_train)+" "
		fout_train.write(prefix)
		fout_train.write(line.lower().rstrip('\n'))
		fout_train.write("\n")
		i_train+=1
	else:
		prefix='test_neg'+str(i_test)+" "
		fout_test.write(prefix)
		fout_test.write(line.lower().rstrip('\n'))
		fout_test.write("\n")
		i_test+=1
	count+=1

filename="rt-polarity.pos"
file_path=os.path.join(fileroot,filename)
fin_pos=open(file_path,'r')
count=1
fout_train=open("train_pos.txt",'w')
fout_test=open("test_pos.txt",'w')
i_train=0
i_test=0
for line in fin_pos.readlines():
	if count in train_idx:
		prefix='train_pos'+str(i_train)+" "
		fout_train.write(prefix)
		fout_train.write(line.lower().rstrip('\n'))
		fout_train.write("\n")
		i_train+=1
	else:
		prefix='test_pos'+str(i_test)+" "
		fout_test.write(prefix)
		fout_test.write(line.lower().rstrip('\n'))
		fout_test.write("\n")
		i_test+=1
	count+=1