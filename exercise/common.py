#coding: utf-8 -*-

import numpy as np
import random
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

from collections import OrderedDict

import sys
print(sys.path)

def create_lexicon(pos_file,neg_file):
    lex = []
    def process_file(f):
        with open(pos_file,encoding='latin-1',mode ='r') as f:
            lex = []
            lines = f.readlines()
            for line in lines:
                words = word_tokenize(line.lower())
                lex += words
            return lex
    lex += process_file(pos_file)
    lex += process_file(neg_file)

    lemmatizer = WordNetLemmatizer()
    # cats -> cat
    lex = [lemmatizer.lemmatize(word) for word in lex]

    word_count = Counter(lex)

    lex = []
    for word in word_count:
        # Should be using percent
        if word_count[word] < 2000 and word_count[word] > 20:
            lex.append(word)
    return lex



def useful_filed(org_file,output_file):
    output = open(output_file,'w',encoding='utf-8')
    with open(org_file,buffering=10000,encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                clf = line.split(',')[0]
                if clf == '0':
                    clf = [0,0,1]
                elif clf == '2':
                    clf = [0,1,0]
                elif clf == '4':
                    clf = [1,0,0]
                tweet = line.split(',')[-1]
                outputline = str(clf) + ':%:%:%:' + tweet
                output.write(outputline)
        except Exception as e:
            print(e)
    output.close()

def get_test_dataset(test_file,lex):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(":%:%:%:")[0]
            tweet = line.split(":%:%:%:")[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(w) for w in words]
            fts = np.zeros(len(lex))
            for w in words:
                if w in lex:
                    fts[lex.index(w)] = 1
            test_x.append(list(fts))
            test_y.append(eval(label))
    return test_x,test_y


def get_random_line(f,point):
    """ f: file object"""
    f.seek(point)
    f.readline()
    return f.readline()

def get_n_random_line(f_path, n=150):
    lines = []
    file = open(f_path, encoding='latin-1')
    total_bytes = os.stat(file).st_size
    for i in range(n):
        random_p = random.randint(0,total_bytes)
        lines.append(get_random_line(file,random_p))
    file.close()
    return  lines



def create_tweet_lexicon_buffer(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with open(train_file,buffering=10000, encoding='latin-1') as f:
        try:
            word_count = {}
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words  = word_tokenize(line.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in word_count:
                        word_count[word] = 1
                    else:
                        word_count[word] += 1
            word_count = OrderedDict(sorted(word_count.items(), key=lambda x:x[1]))
            for w in word_count:
                if word_count[w]< 100000 and word_count[w] > 100:
                    lex.append(w)
        except Exception as e:
            print(e)
    return lex



def normalize_dataset(lex,pos_txt, neg_txt):
    dataset = []
    # review to lex.
    def string_to_vector(lex, review, clf):
        words = word_tokenize(review)
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        #words = word_tokenize(line.lower())
        fts = np.zeros(len(lex))
        for word in words:
            if word in lex:
                fts[lex.index(word)] = 1
        return [fts,clf]

    with open(pos_txt, encoding='latin-1' ,mode= 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex,line.lower(),[1,0])
            dataset.append(one_sample)

    with open(neg_txt,encoding='latin-1' ,mode= 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex,line,[0,1])
            dataset.append(one_sample)
    return dataset