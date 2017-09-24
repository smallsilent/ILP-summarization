# coding = utf-8

import os
import codecs
import re
import nltk
from nltk.corpus import stopwords

name = 'DUC06'
newname = 'newDUC06'
top_n = 20

file_list = os.listdir(name)
print file_list
for file in file_list:
    print file
    file_open = codecs.open(name + '/' +file,'r','utf-8')
    lines = file_open.readlines()
    document = {}
    score = {}
    sentences = []
    file_write = codecs.open(newname + '/' + file, 'w', 'utf-8')
    for line in lines:
        line = line.strip()
        line = line.split('::')
        line1 = re.sub('[^A-Za-z ]', '', line[1]).lower()
        line1 = re.sub(' +', ' ', line1)
        # print line
        s = nltk.stem.SnowballStemmer('english')
        words = line1.split(' ')
        stopset = set(stopwords.words('english'))
        wordlist = [word for word in words if word not in stopset]
        # print wordlist
        line1 = [s.stem(word) for word in wordlist]
        line1s = [word for word in line1 if word.isalpha()]
        count = 0.0
        if line[0].find('q') >-1:
            file_write.write(line[0] + '::' + line[1] + '\n')
            query = line1s
            print query
        elif line[0].find('s') > -1:
            file_write.write(line[0] + '::' + line[1] + '\n')
        elif line[0].find('d') == 0:
            print line[0]
            for word in line1s:
                if word in query:
                    count += 1
            count = count/len(line1s)
            print count
            document[line[0]] = line[1]
            score[line[0]] = count
        else:
            sentences.append(line[0] + '::' + line[1])
    score = sorted(score.iteritems(), key=lambda e : e[1],reverse=True)
    print score
    for i in range(top_n):
        file_write.write(score[i][0] + '::' + document[score[i][0]] + '\n')
        sen_index = score[i][0].split('_')
        for j in range(len(sentences)):
            index = sentences[j].split('::')
            if sen_index[1] == index[0]:
                file_write.write(sentences[j] + '\n')
    file_write.close()
