#coding: utf-8

import collections
#_P_FILES = "../data/poetry_utf8.txt"
_P_FILES = "../data/poetry.txt"
import codecs

top_words_len = 1000

def poem_reader():
    poetrys = []
    with codecs.open(_P_FILES,'r','utf-8') as f:
        for line in f:
            try:
                title,content = line.strip().split(":")
                content = content.replace(" ",'')
                content = content.replace("_","")
                content = content.replace("，","")
                #print(content)
                poetrys.append(content)
                #if '_' in content or '(' in content or "《" in content or '[' in content:
                #    continue
                        #if len(content) < 5 or len(content) > 79:
                        #    continue
                #print(content)
                #content = '['+content+']'
                #poetrys.append(content)
            except Exception as e:
                pass
    poetrys = sorted(poetrys,key=lambda l:len(l))
    return poetrys


def poem_preprocess():
    poems = poem_reader()
    all_words = []
    for p in poems:
        all_words += [w for w in p ]
    counter = collections.Counter(all_words)
    counter_pairs = sorted(counter.items(),key=lambda x:-x[-1])
    words,_ = zip(*counter_pairs)

    words = words[:top_words_len] + (' ',)
    word_num_map = dict(zip(words,range(len(words))))
    to_num = lambda w: word_num_map.get(w,len(words))
    poetrys_vec = [list(map(to_num,p)) for p in poems]

    batch_size = 64
    n_chunk = len(poetrys_vec)



def main():
    poem_preprocess()


if __name__ == "__main__":
    main()