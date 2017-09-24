#coding=utf-8
from gurobipy import *
import cPickle
import math
import numpy as np
import codecs
import nltk
from nltk.corpus import stopwords
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vecdim = 64
Iter = '49'
name = 'newDUC06'
data_name =  '06data_w2v_64_0.05_0.0001'
sen_vec_path  = '06data_w2v_64_0.05_0.0001/sen_vector'+Iter
sen_index_path = '06data_w2v_64_0.05_0.0001/senindex'
sen_topic_path = '06data_w2v_64_0.05_0.0001/sen_topic'+Iter
sum_name = '06summ_data_w2v_64_0.05_0.0001_ILP'
L = 250
C = 0.5

def test():
    # Create a new model
    m = Model("mip1")
    # Create variables
    x = m.addVar(obj = 1,vtype=GRB.BINARY, name="x")
    y = m.addVar(obj = 1,vtype=GRB.BINARY, name="y")
    z = m.addVar(obj = 2,vtype=GRB.BINARY, name="z")
    # Integrate new variables
    m.update()
    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")
    m.optimize()
    for v in m.getVars():
        print (v.varName, v.x)
    print ('Obj:', m.objVal)

def test2():
    # Model data
    commodities = ['Pencils', 'Pens']
    nodes = ['Detroit', 'Denver', 'Boston', 'New York', 'Seattle']
    arcs, capacity = multidict({
        ('Detroit', 'Boston'): 100,
        ('Detroit', 'New York'): 80,
        ('Detroit', 'Seattle'): 120,
        ('Denver', 'Boston'): 120,
        ('Denver', 'New York'): 120,
        ('Denver', 'Seattle'): 120})
    arcs = tuplelist(arcs)
    cost = {
        ('Pencils', 'Detroit', 'Boston'): 10,
        ('Pencils', 'Detroit', 'New York'): 20,
        ('Pencils', 'Detroit', 'Seattle'): 60,
        ('Pencils', 'Denver', 'Boston'): 40,
        ('Pencils', 'Denver', 'New York'): 40,
        ('Pencils', 'Denver', 'Seattle'): 30,
        ('Pens', 'Detroit', 'Boston'): 20,
        ('Pens', 'Detroit', 'New York'): 20,
        ('Pens', 'Detroit', 'Seattle'): 80,
        ('Pens', 'Denver', 'Boston'): 60,
        ('Pens', 'Denver', 'New York'): 70,
        ('Pens', 'Denver', 'Seattle'): 30}

    inflow = {('Pencils', 'Detroit'): 50,
              ('Pencils', 'Denver'): 60,
              ('Pencils', 'Boston'): -50,
              ('Pencils', 'New York'): -50,
              ('Pencils', 'Seattle'): -10,
              ('Pens', 'Detroit'): 60,
              ('Pens', 'Denver'): 40,
              ('Pens', 'Boston'): -40,
              ('Pens', 'New York'): -30,
              ('Pens', 'Seattle'): -30}
    # Create optimization model
    m = Model('netflow')
    # Create variables
    flow = {}
    for h in commodities:
        for i, j in arcs:
            flow[h, i, j] = m.addVar(ub=capacity[i, j], obj=cost[h, i, j],
                                     name='flow_%s_%s_%s' % (h, i, j))
            m.update()
    # Arc capacity constraints
    for i, j in arcs:
        m.addConstr(quicksum(flow[h, i, j] for h in commodities) <= capacity[i, j],
                    'cap_%s_%s' % (i, j))
    # Flow conservation constraints
    for h in commodities:
        for j in nodes:
            m.addConstr(
                quicksum(flow[h, i, j] for i, j in arcs.select('*', j)) +
                inflow[h, j] ==
                quicksum(flow[h, j, k] for j, k in arcs.select(j, '*')),
                'node_%s_%s' % (h, j))
    # Compute optimal solution
    m.optimize()
    # Print solution
    if m.status == GRB.Status.OPTIMAL:
        for h in commodities:
            print ('\nOptimal flows for', h, ':')
            for i, j in arcs:
                if flow[h, i, j].x > 0:
                    print (i, '->', j, ':', flow[h, i, j].x)

def computecos(x,y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return
    x = np.matrix(x)
    y = np.matrix(y)
    result1 = x * y.T
    result2 = x * x.T
    result3 = y * y.T
    result = result1 / (math.pow(result2 * result3 , 0.5))
    return result

def position():
    file_r = open(data_name + '/position_list', 'rb')
    position_list = cPickle.load(file_r)
    file_r.close
    return position_list

def get_sigdoc_position_score(sen_index, position_list):
    sigdoc_positions = [position_list[i] for i in sen_index]

    sigdoc_position_score = []
    for sigdoc_position in sigdoc_positions:
        if sigdoc_position < 10:
            score = C ** sigdoc_position
        else:
            score = C ** 10
        sigdoc_position_score.append(score)
    return sigdoc_position_score


def get_topics():
    file_r = open(sen_topic_path,'rb')
    topics = cPickle.load(file_r)
    return topics

def get_sen_vecs():
    file_r = open(sen_vec_path,'rb')
    sen_vec = cPickle.load(file_r)

    return sen_vec


    # similarity_maxtri = []
    # for i in range(sen_len):
    #     similarity_vec = []
    #     for j in range(i):
    #         similarity_vec.append(computecos(sen_vec[i],sen_vec[j]))
    #     similarity_maxtri.append(similarity_vec)

def get_sen_index():
    file_r = open(sen_index_path, 'rb')
    sen_index = cPickle.load(file_r)

    return sen_index


def get_sigdoc_topic_weight(query_index,sen_index,topics):
    query_topic = topics[query_index]
    sigdoc_topic = [topics[i] for i in sen_index]

    counts = 0
    topics_weight = []
    for i in range(vecdim):
        count = sigdoc_topic.count(i)
        counts = counts + count
        topics_weight.append(count)

    topics_weight = [float(x) / counts for x in topics_weight]
    topics_weight[query_topic] = 1

    # sigdoc_topic = [topics[i] for i in sen_index]
    # topics_weight = [0 for i in sen_index]
    # topics_weight[query_topic] = 1


    return sigdoc_topic, topics_weight

def get_sigdoc_sen_index_and_sentf(file,senindex,sen_vec):
    file_open = codecs.open(name + '/' + file, 'r', 'utf-8')
    lines = file_open.readlines()
    file_open.close()

    sigdoc_sen_string = []
    sigdoc_sen_vec = []
    sigdoc_sen_index = []
    sen_tfidf_list = []
    print file

    text = []
    sen_doc_index = []
    doc_index = -1
    for line in lines:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') == 0:#统计tf-idf
            doc_index = doc_index + 1
            line[1] = re.sub('[^A-Za-z ]', '', line[1]).lower()
            line[1] = re.sub('\s+', ' ', line[1])
            # print line
            s = nltk.stem.SnowballStemmer('english')
            words = line[1].split(' ')
            stopset = set(stopwords.words('english'))
            wordlist = [word for word in words if word not in stopset]
            # print wordlist
            line1 = [s.stem(word) for word in wordlist]
            line[1] = [word.encode("utf-8") for word in line1 if word.isalpha()]
            line1 = ''
            for i in range(len(line[1])):
                line1 = line1 + line[1][i] + ' '
            line1 = line1.strip()
            text.append(line1)
        elif '--------' in line[1]:
            continue
        elif line[0].find('s') > -1:
            query_index = senindex[line[1]]
            query = sen_vec[query_index]

        elif line[0].find('q') > -1:
            line[0]
        else:
            if line[1] not in senindex:
                continue
            elif len(line[1].split(' ')) < 8:
                continue
            sigdoc_sen_string.append(line[1])
            sen_index = senindex[line[1]]
            sigdoc_sen_index.append(sen_index)
            sigdoc_sen_vec.append(sen_vec[sen_index])
            sen_doc_index.append(doc_index)

    #利用tf-idf计算句子重要性
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
    tdidf_word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for sen_index, sen in enumerate(sigdoc_sen_string):
        sen = re.sub('[^A-Za-z ]', '', sen).lower()
        sen = re.sub('\s+', ' ', sen)
        # print line
        s = nltk.stem.SnowballStemmer('english')
        words = sen.split(' ')
        stopset = set(stopwords.words('english'))
        wordlist = [word for word in words if word not in stopset]
        # print wordlist
        line1 = [s.stem(word) for word in wordlist]
        words = [word.encode("utf-8") for word in line1 if word.isalpha()]

        sen_tfidf = 0
        for word in words:
            if word not in tdidf_word:
                continue
            word_index = tdidf_word.index(word)
            sen_tfidf = sen_tfidf + weight[sen_doc_index[sen_index]][word_index]
        sen_tfidf_list.append(sen_tfidf)



    return query, query_index, sigdoc_sen_string, sigdoc_sen_vec, sigdoc_sen_index, sen_tfidf_list

# def get_sigdoc_sen_index(file,senindex,sen_vec):
#
#     file_open = codecs.open(name + '/' + file, 'r', 'utf-8')
#     lines = file_open.readlines()
#     file_open.close()
#
#     sigdoc_sen_string = []
#     sigdoc_sen_vec = []
#     sigdoc_sen_index = []
#     print file
#
#     for line in lines:
#         line = line.strip()
#         line = line.split('::')
#         if line[0].find('d') == 0:
#             line[0]
#         elif '--------' in line[1]:
#             continue
#         elif line[0].find('s') > -1:
#             query_index = senindex[line[1]]
#             query = sen_vec[query_index]
#
#         elif line[0].find('q') > -1:
#             line[0]
#         else:
#             if line[1] not in senindex:
#                 continue
#             # elif len(line[1].split(' ')) < 10:
#             #     continue
#             sigdoc_sen_string.append(line[1])
#             sen_index = senindex[line[1]]
#             sigdoc_sen_index.append(sen_index)
#             sigdoc_sen_vec.append(sen_vec[sen_index])
#
#     return query,query_index,sigdoc_sen_string, sigdoc_sen_vec, sigdoc_sen_index

def compute_sen_salience(query,sigdoc_sen_vec):
    sigdoc_sen_salience = []
    for sen_vec in sigdoc_sen_vec:
        salience = computecos(query,sen_vec)[0,0]
        sigdoc_sen_salience.append(salience)

    return sigdoc_sen_salience

def compute_similarity_maxtri(sigdoc_sen_vec):
    similarity_maxtri = []
    sen_len = len(sigdoc_sen_vec)
    for i in range(sen_len):
        similarity_vec = []
        for j in range(i):
            similarity_vec.append(computecos(sigdoc_sen_vec[i],sigdoc_sen_vec[j])[0,0])
        similarity_maxtri.append(similarity_vec)
    return similarity_maxtri

def compute_sigdoc_sen_len(sigdoc_sen_string):
    sigdoc_sen_len = []
    for sen in sigdoc_sen_string:
        lenth = len(sen.strip().split(' '))
        sigdoc_sen_len.append(lenth)

    return sigdoc_sen_len

def compute_sen_index_with_ILP(sigdoc_topic,sigdoc_topic_weight,sigdoc_sen_salience,sigdoc_similarity_maxtri, sigdoc_sen_len, sen_tfidf_list,sigdoc_position_score):
    m = Model('netflow')
    m.setParam('MIPFocus', 3)
    # m.setParam('Heuristics', 0.5)
    m.setParam('TimeLimit', 200)


    topic_vars = {}
    for index,topic in enumerate(sigdoc_topic_weight):
         topic_vars[index] = m.addVar(lb=0, ub=1, obj=1, vtype= GRB.INTEGER)
    sen_vars = {}
    for index, salience in enumerate(sen_tfidf_list):
        sen_vars[index] = m.addVar(lb=0, ub=1, obj=1, vtype=GRB.INTEGER)
    similarity_vars = {}
    for i in range(len(sigdoc_similarity_maxtri)):
        for j in range(i):
            similarity_vars[i,j] = m.addVar(lb=0, ub=1, obj=1, vtype=GRB.INTEGER)
    # Integrate new variables
    m.update()
    # Set objective
    topic_salience = quicksum(topic_vars[i]*sigdoc_topic_weight[i] for i in topic_vars)
    sen_salience = quicksum(sen_vars[i]*(sen_tfidf_list[i]+sigdoc_sen_salience[i])*sigdoc_position_score[i] for i in sen_vars)
    # similarity_punish = quicksum(similarity_vars[i,j]*(sen_tfidf_list[i]*sigdoc_position_score[i] + sen_tfidf_list[j]*sigdoc_position_score[j])*sigdoc_similarity_maxtri[i][j] for i,j in similarity_vars)
    similarity_punish = quicksum(similarity_vars[i, j] * sigdoc_similarity_maxtri[i][j] for i, j in similarity_vars)
    m.setObjective(topic_salience + sen_salience - similarity_punish, GRB.MAXIMIZE)
    # m.setObjective(sen_salience - similarity_punish, GRB.MAXIMIZE)

    # Add constraint
    m.addConstr(quicksum(sen_vars[i] * sigdoc_sen_len[i] for i in sen_vars) <= L)
    # for i in sigdoc_topic_weight:
    #     if i == 0:
    #         m.addConstr(sen_vars[i] <= 0)
    for index,i in enumerate(sigdoc_sen_salience):
        if sigdoc_sen_salience < 0.4:
            m.addConstr(sen_vars[index] <= 0)
    for j in sen_vars:
        for i in topic_vars:
            if sigdoc_topic[j] == i:
                occij = 1
            else:
                occij = 0
            m.addConstr(sen_vars[j]*occij <= topic_vars[i])

    for i in topic_vars:
        sum = 0
        for j in sen_vars:
            if sigdoc_topic[j] == i:
                occij = 1
            else:
                occij = 0
            sum = sum + sen_vars[j]*occij
        m.addConstr(sum >= topic_vars[i])

    for i,j in similarity_vars:
        m.addConstr(similarity_vars[i,j] - sen_vars [i] <= 0)
        m.addConstr(similarity_vars[i,j] - sen_vars[j] <= 0)
        m.addConstr(sen_vars[i] + sen_vars[j] - similarity_vars[i,j] <= 1)


    m.optimize()
    status = m.status

    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % m.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)

    judge_sen = []
    for i in sen_vars:
        print sen_vars[i].x
        judge_sen.append(sen_vars[i].x)
    return judge_sen




def main():
    position_list = position()
    topics = get_topics()
    sen_vec = get_sen_vecs()
    sen_index = get_sen_index()

    file_list = os.listdir(name)
    for file in file_list:
        # query,query_index, sigdoc_sen_string, sigdoc_sen_vec, sigdoc_sen_index = get_sigdoc_sen_index(file,sen_index,sen_vec)
        query,query_index, sigdoc_sen_string, sigdoc_sen_vec, sigdoc_sen_index, sen_tfidf_list = get_sigdoc_sen_index_and_sentf(file,sen_index,sen_vec)
        sigdoc_position_score = get_sigdoc_position_score(sigdoc_sen_index,position_list)
        sigdoc_sen_len = compute_sigdoc_sen_len(sigdoc_sen_string)
        # sen_tfidf_list = [x/sigdoc_sen_len[sen_tfidf_list.index(x)] for x in sen_tfidf_list]
        sigdoc_sen_salience = compute_sen_salience(query,sigdoc_sen_vec)
        sigdoc_similarity_maxtri = compute_similarity_maxtri(sigdoc_sen_vec)
        sigdoc_topic, sigdoc_topic_weight = get_sigdoc_topic_weight(query_index,sigdoc_sen_index,topics)
        judge_sen = compute_sen_index_with_ILP(sigdoc_topic, sigdoc_topic_weight, sigdoc_sen_salience, sigdoc_similarity_maxtri, sigdoc_sen_len,sen_tfidf_list,sigdoc_position_score)

        if 1 not in judge_sen:
            lengh = 0
            for index, i in enumerate(sigdoc_position_score):
                if i >= 0.25:
                    judge_sen[index] = 1
                    lengh = lengh + sigdoc_sen_len[index]
                if lengh > 250:
                    break
            for index, judge in enumerate(judge_sen):
                if judge == 1:
                    if not os.path.exists(sum_name):
                        os.makedirs(sum_name)
                    with open(sum_name + '/position/task' + file.split('.')[0] + '_1.txt', 'a') as w:
                        w.write(sigdoc_sen_string[index])
                        w.write('\n')
        else:
            for index,judge in enumerate(judge_sen):
                if judge == 1:
                    if not os.path.exists(sum_name):
                        os.makedirs(sum_name)
                    with open(sum_name + '/task' + file.split('.')[0] + '_1.txt','a') as w:
                       w.write(sigdoc_sen_string[index])
                       w.write('\n')
        # file_w = open(sum_name + '/' + file + 'score', 'wb')
        # cPickle.dump(score, file_w)



if __name__ == '__main__':
    # test()
    # test2()
    main()