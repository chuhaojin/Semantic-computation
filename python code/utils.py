#import pandas as pd
import numpy as np
import pickle
#import jieba
#from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('error')


def compute(in_x, dataset, norm=1):
    dataset_size = dataset.shape[0]
    # 计算距离
    result = np.zeros([dataset_size])
    for i in range(dataset_size):
        # if i % 10000 == 0:
        #     print(i)
        try:
            result[i] = cos(in_x, dataset[i]) ** norm
        except RuntimeWarning:
            result[i] = 0
            continue
    sorted_dist_indicies = (-result).argsort(-1)
    return sorted_dist_indicies, result


def cos(v1, v2):
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))


def cut_words(sentence, cut_type = True):
    return jieba.lcut(sentence, cut_all = cut_type)


def batch_cut_words(sentence_list, cut_type = True):
    words_list = []
    for sentence in sentence_list:
        words_list.append(cut_words(sentence, cut_type=cut_type))
    return words_list


def get_csv_file(file_path):
    '''
    :param file_path: a csv file path.
    :return: csv_data:list,shape:[[row1],[row2],[row3]...],first row not included.
    '''
    csv_data = pd.read_csv(file_path)
    csv_data = csv_data.values.tolist()
    return csv_data


def get_column(list_data, column=0):
    column_data = []
    for row in list_data:
        column_data.append(row[column])
    return column_data


def get_batch_column(list_data, begin_column=0, end_column=-1):
    column_data = []
    for row in list_data:
        column_data.append(row[begin_column:end_column])
    return column_data


def make_softmax_labels(labels, num=16):
    softmax_labels = np.zeros([len(labels), num])
    try:
        softmax_labels[[i for i in range(len(labels))], labels] = 1
    except:
        pass
    return softmax_labels


def words_statistics(words_list):
    words_stat = []
    for i in range(len(words_list)):
        words_stat.extend(words_list[i])
    words_stat = list(set(words_stat))
    words_dic = {}
    for i, word in enumerate(words_stat):
        words_dic[word]=i
    words_stat = words_dic
    return words_stat


def get_words_index(words_stat, word_list, sequence_length=25, get_step=False):
    index = np.zeros(sequence_length, dtype=np.int) + len(words_stat)
    step = 0
    for i in range(min(len(word_list), sequence_length)):
        try:
            index[step] = words_stat[word_list[i]]
            step += 1
        except KeyError:
            continue
    if get_step:
        return index, step
    else:
        return index


def get_batch_words_index(words_stat, batch_words_list, sequence_length=25, get_steps=False):
    batch_index = np.zeros([len(batch_words_list), sequence_length], dtype=np.int) + len(words_stat)
    steps = np.zeros(len(batch_words_list), dtype=np.int)
    for i, word_list in enumerate(batch_words_list):
        batch_index[i], steps[i] = get_words_index(words_stat, word_list, sequence_length, get_step=True)
    if get_steps:
        return batch_index, steps
    else:
        return batch_index


def sentence_split(sentence, symbol=' '):
    for i in range(len(sentence)):
        try:
            sentence[i] = sentence[i].split(symbol)
        except:
            sentence[i] = [""]
    words = sentence
    return words


def remove_start_and_end(input_str, start=1, end=-1):
    return input_str[1:-1]


def batch_reomve_start_and_end(str_list, start=1, end=-1):
    retun_str_list = []
    for i, input_str in enumerate(str_list):
        retun_str_list.append(remove_start_and_end(input_str=input_str))
    return retun_str_list


def get_mean_vector(vector, step):
    vector = vector[0:step]
    mean_vector = np.mean(vector, axis=0)
    return mean_vector


def get_batch_mean_vector(batch_vector, steps):
    batch_size, _, dimension = batch_vector.shape
    batch_mean_vector = np.zeros([batch_size, dimension])
    for i in range(batch_size):
        batch_mean_vector[i] = get_mean_vector(batch_vector[i], step=steps[i])
    return batch_mean_vector


def score_main(y_true, y_pred):
    f1score = f1_score(y_true, y_pred, average='macro')
    return f1score


def balanced_data_distribution(labels):
    '''
    :param labels: input labels
    :return: balanced_data's index
    '''
    distr = np.bincount(labels)
    balanced = np.sum(distr) / distr
    balanced_int = np.array(balanced / np.min(balanced) + 0.5, dtype=np.int)
    balanced_index = np.zeros(np.sum(balanced_int * distr), dtype=np.int)
    t = 0
    for i, label in enumerate(labels):
        balanced_index[t:t+balanced_int[labels[i]]] += i
        t += balanced_int[labels[i]]
    np.random.shuffle(balanced_index)
    return balanced_index


class StopWords:
    def __init__(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        words = {}
        for line in lines:
            words[line[0:-1]] = ''
        f.close()
        self.words = words

    def remove(self, words_list):
        '''

        :param words_list: [word1,word2,word3...]
        :return:
        '''
        return_words = []
        for word in words_list:
            try:
                _ = self.words[word]
            except KeyError:
                return_words.append(word)
        if return_words == []:
            return_words = [""]
        return return_words

    def batch_remove(self, batch_words_list):
        '''
        :param batch_words_list: [[word1,word2..],[word1,word2..],[word1,word2..]..]
        :return:
        '''
        return_list = []
        for i in range(len(batch_words_list)):
            try:
                return_list.append(self.remove(batch_words_list[i]))
            except TypeError:
                return_list.append([""])

        return return_list


class Word2Vec:
    def __init__(self, load_path, dimension=False, norm=1):
        self.norm = norm
        f = open(load_path, 'rb')
        self.model = pickle.load(f)
        self.vector = pickle.load(f)
        mean = np.mean(self.vector)
        std = np.std(self.vector)
        self.vector = (self.vector - mean) / std
        f.close()
        self.words = []
        if not dimension:
            self.dimension = self.vector.shape[1]
        else:
            self.dimension = dimension
        for word in self.model:
            self.words.append(word)
        self.middle = self.vector.shape[1] / 2
        self.down = int(self.middle - self.dimension / 2)
        self.up = int(self.middle + self.dimension / 2)
        return

    def get_vector(self, word):
        return self.vector[self.model[word]][self.down:self.up]

    def get_batch_vector(self, words, extend=False):
        if extend:
            batch_vector = np.zeros([len(words) + 1, self.dimension])
        else:
            batch_vector = np.zeros([len(words), self.dimension])
        if type(words) == list:
            for i in range(len(words)):
                try:
                    batch_vector[i] = self.get_vector(words[i])
                except KeyError:
                    continue
        elif type(words) == dict:
            for i, word in enumerate(words.keys()):
                try:
                    batch_vector[i] = self.get_vector(word)
                except KeyError:
                    continue
        return batch_vector

    def most_similar(self, word, topn=100):
        similar_words = []
        try:
            sort, result = compute(self.get_vector(word), self.vector[:, self.down:self.up])
            for i in range(1, topn + 1):
                similar_words.append((self.words[sort[i]], result[sort[i]]))
            return similar_words
        except KeyError:
            return False

    def words_to_vector(self, words_list, vector_length=30):
        '''

        :param words_list: [word1,word2,word3,...]
        :param vector_length: the shape of return vector:[dimension, vector_length]
        :return:temp:vector,shape:[dimension, vector_length]
        :return:step: the valid length of words_list
        '''
        words_length = len(words_list)
        temp = np.zeros([vector_length, self.dimension])
        length = min(words_length, vector_length)
        # for i in range():
        # temp[[i for i in range(length)]] = [self.get_vector(words_list[i]) for i in range(length)]
        step = 0
        error_words = []
        for i in range(length):
            try:
                temp[step] = self.get_vector(words_list[i])
                step = step+1
            except KeyError:
                error_words.append(words_list[i])
        return temp, step

    def similarity(self, word1, word2):
        return cos(self.get_vector(word1), self.get_vector(word2))

    def filter_words(self, words):
        if type(words) == dict:
            return_words = {}
            k = 0
            for word in words.keys():
                try:
                    _ = self.model[word]
                    return_words[word] = k
                    k += 1
                except KeyError:
                    continue
            return return_words
        elif type(words) == list:
            return_words = []
            for word in words:
                try:
                    _ = self.model[word]
                    return_words.append(word)
                except KeyError:
                    continue
            return return_words

    def get_mean_vector(self, words):
        words_vector,step = self.words_to_vector(words)
        mean_vector = np.sum(words_vector, axis=0) / step
        return mean_vector

    def get_batch_mean_vector(self, words_list):
        batch_mean_vector = np.zeros([len(words_list), self.dimension])
        for i, words in enumerate(words_list):
            batch_mean_vector[i] = self.get_mean_vector(words)
        return batch_mean_vector

    def similarity_for_many_words(self, pos_words, neg_words, topn=10):
        pos_result = np.zeros(len(self.vector))
        neg_result = np.zeros(len(self.vector))
        for pos_word in pos_words:
            sort, result = compute(self.get_vector(pos_word), self.vector[:, self.down:self.up])
            pos_result = pos_result + result
        for neg_word in neg_words:
            sort, result = compute(self.get_vector(neg_word), self.vector[:, self.down:self.up])
            neg_result = neg_result + result
        pos_max = np.max(pos_result)
        pos_min = np.min(pos_result)
        neg_max = np.max(neg_result)
        neg_min = np.min(neg_result)
        if len(pos_words) != 0 and len(neg_words) != 0:
            pos_result = (pos_result - pos_min) / (pos_max - pos_min)
            neg_result = (neg_result - pos_min) / (neg_max - neg_min)
            result = pos_result / (neg_result + 0.000001)
            sort = (-result).argsort(-1)
        elif len(pos_words) == 0:
            result = 1 / (neg_result + 0.000001)
            sort = (-result).argsort(-1)
        elif len(neg_words) == 0:
            result = pos_result
            sort = (-result).argsort(-1)
        return_result = []
        for i in range(topn):
            if self.words[sort[i]] not in pos_words + neg_words:
                return_result.append((self.words[sort[i]], result[sort[i]]))
        return return_result

    def similarity_for_many_words_with_weights(self, pos_words, neg_words, pos_weights, neg_weights, topn=10):
        pos_result = np.zeros(len(self.vector))
        neg_result = np.zeros(len(self.vector))
        for i, pos_word in enumerate(pos_words):
            sort, result = compute(self.get_vector(pos_word), self.vector[:, self.down:self.up])
            pos_result = pos_result + pos_weights[i] * result
        for i, neg_word in enumerate(neg_words):
            sort, result = compute(self.get_vector(neg_word), self.vector[:, self.down:self.up])
            neg_result = neg_result + neg_weights[i] * result
        pos_max = np.max(pos_result)
        pos_min = np.min(pos_result)
        neg_max = np.max(neg_result)
        neg_min = np.min(neg_result)
        if len(pos_words) != 0 and len(neg_words) != 0:
            pos_result = (pos_result - pos_min) / (pos_max - pos_min)
            neg_result = (neg_result - pos_min) / (neg_max - neg_min)
            result = pos_result / (neg_result + 0.000001)
            sort = (-result).argsort(-1)
        elif len(pos_words) == 0:
            result = 1 / (neg_result + 0.000001)
            sort = (-result).argsort(-1)
        elif len(neg_words) == 0:
            result = pos_result
            sort = (-result).argsort(-1)
        return_result = []
        for i in range(topn):
            if self.words[sort[i]] not in pos_words + neg_words:
                return_result.append((self.words[sort[i]], result[sort[i]]))
        return return_result


# def reprocess(csv_data):
#     '''
#     :param csv_data: list,shape:[[row1],[row2],[row3]...],first line not included.
#     :return:
#     '''
#     for i in range(len(csv_data)):
#         csv_data[i][1] = csv_data[i][1][1:-1]
#     return csv_data


# if __name__ == "__main__":
#
#     train_path = "../train/sentiment_analysis_trainingset.csv"
#     data = get_csv_file(train_path)
#     # data = reprocess(data)
#     print(data[0])
#     model = Word2Vec('../nlp/word2vec/sgns.weibo.300d')
#     vector, error_words = model.words_to_vector(['嗯', '天气', '海边'], vector_length=5)
#     print(vector.shape)
