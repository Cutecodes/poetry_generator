import torch
import pickle
from Poetry import Poetry
from Poetrymodel import *


def generate(model,start_words,idx2word,word2idx):
    results = list(start_words)
    start_words_len =len(start_words)

    input_ = torch.Tensor([word2idx["["]]).view(1,1).long()
    hidden = None
    for i in range(500):
        output,hidden =model(input_,hidden)
        if i<start_words_len:
            w = results[i]
            input_ = input_.data.new([word2idx[w]]).view(1,1)
        else:
            max = output.data[0].topk(1)[1][0].item()
            w = idx2word[max]
            results.append(w)
            input_ = input_.data.new([max]).view(1, 1)
        if w=="]":
            del results[-1]
            break
    return results

def main():
    with open("./dataset/word_id","rb") as f:
        word2idx = pickle.load(f)
    with open("./dataset/words_vector","rb") as f:
        idx2word = pickle.load(f)
    model = PoetryModel(len(idx2word))
    model.load_state_dict(torch.load('./model.pth'))
    poem = generate(model,"花",idx2word,word2idx)
    for i in poem:
        print(i)

if __name__ == '__main__':
    main()
