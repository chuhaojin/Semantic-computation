import socket
import utils
import threading
from numpy import zeros
import logging
import os
import time
import warnings
warnings.filterwarnings('ignore')

tim = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
log_file = 'log/'+tim[:8]+'.log'
if not os.path.exists("log"):
    os.makedirs("log")

if not os.path.exists(log_file):
    f = open(log_file, 'w')
    f.close()

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=log_file,
                filemode='a')

logging.debug('This is start-up message')


word2vec_path = "nlp/word2vec/sgns.weibo.300d"
dimension = 100
word2vec_model = utils.Word2Vec(word2vec_path, dimension=dimension)

HOST = '127.0.0.1'
PORT = 10271

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(10)


def similarity(data, conn, id):
    try:
        result = ("%.4f" % word2vec_model.similarity(data[1], data[2])).encode("utf-8")
    except KeyError:
        result = "-1".encode("utf-8")
    conn.sendall(result)
    conn.close()
    logging.info("id:" + str(id) + ", result:" + str(result.decode("utf-8")))
    return


def get_weight_word(temp):
    weight = 1
    if "*" in temp:
        temp = temp.split("*")
        try:
            weight = float(temp[0])
            word = temp[1]
        except ValueError:
            weight = float(temp[1])
            word = temp[0]
    else:
        word = temp
    return weight, word


def word_operation(data, conn, id):
    pos_words = []
    neg_words = []
    sentence = data[1]
    flag = 1
    index = 0
    pos_weights = []
    neg_weights = []
    for i in range(len(sentence)):
        if sentence[i] == " " or sentence[i] == "-":
            if i != index:
                if flag == 1:
                    temp = sentence[index:i]
                    weight, word = get_weight_word(temp)
                    pos_weights.append(weight)
                    pos_words.append(word)
                elif flag == -1:
                    temp = sentence[index:i]
                    weight, word = get_weight_word(temp)
                    neg_weights.append(weight)
                    neg_words.append(word)
                index = i + 1
            if sentence[i] == " ":
                flag = 1
            elif sentence[i] == "-":
                flag = -1
    if flag == 1:
        temp = sentence[index:]
        weight, word = get_weight_word(temp)
        pos_weights.append(weight)
        pos_words.append(word)
    elif flag == -1:
        temp = sentence[index:]
        weight, word = get_weight_word(temp)
        neg_weights.append(weight)
        neg_words.append(word)
    if len(pos_words) + len(neg_words) > 3:
        conn.sendall("-1".encode("utf-8"))
        conn.close()
        return
    return_words = ''
    # try:
    #     result = model.most_similar(positive=pos_words, negative=neg_words, topn=50)
    #     print("result:", result)
    #     for i in range(len(result)):
    #         for p in range(len(pos_words)):
    #             simi = model.similarity(result[i][0], pos_words[p])
    #             if simi > max_simi:
    #                 max_simi = simi
    #                 return_words = result[i][0]
    #     result = return_words.encode("utf-8")
    try:
        result = word2vec_model.similarity_for_many_words_with_weights(pos_words, neg_words,
                                                                       pos_weights, neg_weights, topn=50)
        sort_results = zeros(len(result))
        for i in range(len(result)):
            simi = 0
            for p in range(len(pos_words)):
                simi = simi + pos_weights[p] * word2vec_model.similarity(result[i][0], pos_words[p])
            sort_results[i] = simi
        sorted = (-sort_results).argsort(-1)
        return_words = ""
        topn = 5
        for i in range(topn):
            return_words = return_words + result[sorted[i]][0] + "  ,  "
        return_words = return_words + result[sorted[topn]][0]
        result = return_words.encode("utf-8")
    except KeyError:
        result = "Error".encode("utf-8")
    conn.sendall(result)
    conn.close()
    logging.info("id:" + str(id) + ", result:" + str(result.decode("utf-8")))
    return


id = 0
while True:
    print("waiting...")
    conn, addr = s.accept()
    try:
        bit_data = conn.recv(1024)
        utf_data = bit_data.decode('utf-8')
        data = utf_data.split("##")
        if data[0] == "1":
            if len(threading.enumerate()) > 3:
                conn.sendall("熊孩子别点了！!(叉腰凶！)".encode("utf-8"))
                conn.close()
                continue
            id += 1
            logging.info("id:" + str(id) + ", type:"+"similarity, " + "word1:" + data[1] + "   word2:" + data[2])
            thread_fit = threading.Thread(target=similarity, args=(data, conn, id))
            thread_fit.start()
        elif data[0] == "2":
            logging.info("id:" + str(id) + ", type:" + "word_operation, " + "sentence:" + data[1])
            if len(threading.enumerate()) > 3:
                conn.sendall("熊孩子别点了！!(叉腰凶！)".encode("utf-8"))
                conn.close()
                continue
            thread_fit = threading.Thread(target=word_operation, args=(data, conn, id))
            thread_fit.start()
    except:
        conn.close()
        continue
    id += 1

