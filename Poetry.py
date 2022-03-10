import os
import collections
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

class Poetry(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.poetrys = []
        if os.path.exists(self.root+'words_vector'):
            with open(self.root+'words_vector',"rb") as f:
                self.words_vector = pickle.load(f)
        if os.path.exists(self.root+'poetrys_vector'):
            with open(self.root+'poetrys_vector',"rb") as f:
                self.poetrys_vector = pickle.load(f)
        else:
            with open(os.path.join(self.root,"poetry.txt"),'r',encoding='utf-8',) as f:
                for line in f:
                    try:
                        self.poetrys.append(line.strip())
                    except Exception as e:
                        pass
            #按诗的字数排序
            
            self.poetrys = sorted(self.poetrys,key=lambda line:len(line))
            #print(self.poetrys[100])
            if os.path.exists(self.root+'word_id'):
                try:
                    with open(self.root+'word_id',"rb") as f:
                        word_id = pickle.load(f)
                except Exception as e:
                    print(e)
            
            else:
                self.all_word = []
                for poetry in self.poetrys:
                    self.all_word+=[word for word in poetry]
                counter = collections.Counter(self.all_word)
                count_pairs = sorted(counter.items(), key=lambda x: -x[1])
                words, _ = zip(*count_pairs)
                if os.path.exists(self.root+'words_vector'):
                    pass
                else:
                    with open(self.root+'words_vector',"wb") as f:
                        pickle.dump(words,f)
                    self.words_vector = words
                word_id = dict(zip(words, range(len(words))))

                with open(self.root+'word_id',"wb") as f:
                    pickle.dump(word_id,f)
            to_num = lambda word: word_id.get(word, len(words))
            self.poetrys_vector = [list(map(to_num, poetry)) for poetry in self.poetrys]
            with open(self.root+"poetrys_vector","wb") as f:
                pickle.dump(self.poetrys_vector,f)

        with open(self.root+'words_vector',"rb") as f:
            self.words_vector = pickle.load(f)
        #print(self.poetrys_vector[1000])



    def __len__(self):
        return len(self.poetrys_vector)

    def __getitem__(self,index):
        x = np.array(self.poetrys_vector[index])
        #y = np.copy(x)
        #y = np.roll(y,-1)
        #y[len(y)-1]=y[len(y)-2]
        #x = x.reshape(-1,1)
        #y = y.reshape(-1,1)
        if self.transform is not None:
            x = self.transform(x)
            #y = self.transform(y)
        return x#y

    def get_words_vector(self):
    	return self.words_vector

def main():
    poetry = Poetry(root="./dataset/",transform=None)
    print(len(poetry))
    print(poetry[100])
    print(poetry[101])
    print(poetry[101][0].shape,poetry[101][1].shape)
    words_vector = poetry.get_words_vector()
    #for i in poetry[101][0]:
    	#print(words_vector[i[0]])
    print(words_vector[0])
    print(len(words_vector))

if __name__ == '__main__':
    main()
