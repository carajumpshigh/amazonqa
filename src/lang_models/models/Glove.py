import numpy as np
import os
import codecs
import re


class Glove:
    
    def loadGloveModel(glove_file):
        f = open(glove_file, 'r', encoding='UTF-8')
        model = {}
        for line in f:
            splitline = line.split()
            word = splitline[0].replace("'", "")
            embedding = np.array([float(val) for val in splitline[1: ]])
            model[word] = embedding
        print("Done.", len(model), "words loaded!")
        return model


    def load_index_dic(glove_file):
        f = open(glove_file, 'r', encoding='UTF-8')
        dic = []
        for line in f:
            splitline = line.split()
            dic.append(splitline[0])
        f.close()
        return dic


    def glove_embedding_one_string(string, pretrained_glove):
        words = string.split()
        new_words = [re.sub('[{}!#?,.:";@$%^&*()_+-=|[]:;">/?<,.~]', '', word) for word in words]
        temp = [pretrained_glove[i] for i in new_words if i in pretrained_glove.keys()]
        temp = np.array(temp)
        return np.sum(temp, axis=0)


    def embedding(list, pretrained_glove):
        n, t = len(list), 0
        l = pretrained_glove['a'].shape[0]
        temp = np.zeros((n, l))
        for i in list:
            temp[t] = Glove.glove_embedding_one_string(i, pretrained_glove)
            t += 1
        return np.array(temp)
