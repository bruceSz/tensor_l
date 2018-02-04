#coding: utf-8

import collections
_P_FILES = "../data/poetry_utf8.txt"
import codecs

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
    return poetrys


def poem_preprocess():
    poems = poem_reader()
    all_words = []
    for p in poems:
        all_words += [w for w in p ]
    counter = collections.Counter(all_words)
    counter_pairs = sorted(counter.items(),key=lambda x:-x[-1])
    words,_ = zip(*counter_pairs)

    for w in words[:100]:
        print(w)



def main():
    poem_preprocess()


if __name__ == "__main__":
    main()