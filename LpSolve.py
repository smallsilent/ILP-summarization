import cPickle
import math
import numpy as np
import codecs
from lpsolve55 import *



vecdim = 64
Iter = '49'
name = 'newDUC05'
sen_vec_path  = 'data_w2v_64_0.05_0.0001/sen_vector'+Iter
sen_index_path = 'data_w2v_64_0.05_0.0001/senindex'
sen_topic_path = 'data_w2v_64_0.05_0.0001/sen_topic'+Iter
L = 250

def test():
    lp = lpsolve('make_lp', 0, 4)
    lpsolve('set_verbose', lp, IMPORTANT)
    ret = lpsolve('set_obj_fn', lp, [1, 3, 6.24, 0.1])
    ret = lpsolve('add_constraint', lp, [0, 78.26, 0, 2.9], GE, 92.3)
    ret = lpsolve('add_constraint', lp, [0.24, 0, 11.31, 0], LE, 14.8)
    ret = lpsolve('add_constraint', lp, [12.68, 0, 0.08, 0.9], GE, 4)
    ret = lpsolve('set_lowbo', lp, 1, 28.6)
    ret = lpsolve('set_lowbo', lp, 4, 18)
    ret = lpsolve('set_upbo', lp, 4, 48.98)
    ret = lpsolve('set_col_name', lp, 1, 'COLONE')
    ret = lpsolve('set_col_name', lp, 2, 'COLTWO')
    ret = lpsolve('set_col_name', lp, 3, 'COLTHREE')
    ret = lpsolve('set_col_name', lp, 4, 'COLFOUR')
    ret = lpsolve('set_row_name', lp, 1, 'THISROW')
    ret = lpsolve('set_row_name', lp, 2, 'THATROW')
    ret = lpsolve('set_row_name', lp, 3, 'LASTROW')
    ret = lpsolve('write_lp', lp, 'a.lp')
    print lpsolve('get_mat', lp, 1, 2)

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


def get_sigdoc_topic_weight(sen_index,topics):
    sigdoc_topic = [topics[i] for i in sen_index]

    counts = 0
    topics_weight = []
    for i in range(vecdim):
        count = sigdoc_topic.count(i)
        counts = counts + count
        topics_weight.append(count)

    topics_weight = [float(x) / counts for x in topics_weight]


    return sigdoc_topic, topics_weight

def get_sigdoc_sen_index(file,senindex,sen_vec):

    file_open = codecs.open(name + '/' + file, 'r', 'utf-8')
    lines = file_open.readlines()
    file_open.close()

    sigdoc_sen_string = []
    sigdoc_sen_vec = []
    sigdoc_sen_index = []
    print file

    for line in lines:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') == 0:
            line[0]
        elif '--------' in line[1]:
            continue
        elif line[0].find('s') > -1:
            query = sen_vec[senindex[line[1]]]
        elif line[0].find('q') > -1:
            line[0]
        else:
            if line[1] not in senindex:
                continue
            sigdoc_sen_string.append(line[1])
            sen_index = senindex[line[1]]
            sigdoc_sen_index.append(sen_index)
            sigdoc_sen_vec.append(sen_vec[sen_index])

    return query,sigdoc_sen_string, sigdoc_sen_vec, sigdoc_sen_index

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

def compute_sen_index_with_ILP(sigdoc_topic,sigdoc_topic_weight,sigdoc_sen_salience,sigdoc_similarity_maxtri, sigdoc_sen_len):
    print 1


def main():
    topics = get_topics()
    sen_vec = get_sen_vecs()
    sen_index = get_sen_index()

    file_list = os.listdir(name)
    for file in file_list:
        query, sigdoc_sen_string, sigdoc_sen_vec, sigdoc_sen_index = get_sigdoc_sen_index(file,sen_index,sen_vec)
        sigdoc_sen_len = compute_sigdoc_sen_len(sigdoc_sen_string)
        sigdoc_sen_salience = compute_sen_salience(query,sigdoc_sen_vec)
        sigdoc_similarity_maxtri = compute_similarity_maxtri(sigdoc_sen_vec)
        sigdoc_topic, sigdoc_topic_weight = get_sigdoc_topic_weight(sigdoc_sen_index,topics)
        compute_sen_index_with_ILP(sigdoc_topic, sigdoc_topic_weight, sigdoc_sen_salience, sigdoc_similarity_maxtri, sigdoc_sen_len)

        # file_w = open(sum_name + '/' + file + 'score', 'wb')
        # cPickle.dump(score, file_w)


if __name__ == '__main__':
    # test()
    # test2()
    main()