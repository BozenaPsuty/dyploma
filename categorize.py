import os
import pandas as pd
from collections import Counter, Iterable
from itertools import dropwhile
import ast
import glob
import numpy as np
from collections import defaultdict
import re

class Categorize:

    def food(self, file, categories_dir):
        df = pd.read_csv(file)
        categories = self.list(categories_dir)
        df['aspects'] = df['0']
        i=0
        for x in df['0']:
            if x in categories:
                df['0'][i] = 'FOOD'
            i += 1
        df.to_csv('C:/Users/Bosia/Desktop/aspects.csv', index=None)

    def list(self, directory):
        df = pd.read_csv(directory)
        col_one_list = df['FOOD NAME'].tolist()
        col_one_list = [x.upper() for x in col_one_list]
        return col_one_list

    def flatten(self, lis):
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in self.flatten(item):
                    yield x
            else:
                yield item

    def frequency(self, line):
        word_bag = []
        line = line.split('.')
        for sent in line:
            sent = sent.split(' ')
            for word in sent:
                word_bag.append(word)
        return word_bag

    def word_frequency(self, file_path):
        word_list = []
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                for x in df['removed_aspects']:
                    word_list.append(self.frequency(x))
        word_list = list(self.flatten(word_list))
        counter = Counter(word_list)
        for key, count in dropwhile(lambda key_count: key_count[1] >= 1000, counter.most_common()):
            del counter[key]
        print(counter)
        counter = pd.DataFrame(counter, index=[0])
        counter = counter.T
        counter.to_csv("list.csv", index=None)
        return counter

    def remove(self, file, bag_of_words):
        inputTupples = ast.literal_eval(file)
        removed = []
        inputTupples = list(self.flatten(inputTupples))
        """for key, value in inputTupples.items():
            for word, tag in value:
                if word in bag_of_words:
                    removed.append({word, tag})"""
        for word in inputTupples:
            if word not in list(bag_of_words) and word is not '':
                removed.append(word)
        return removed


    def remove_words(self, file_path, bag_of_words):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                text = {}
                i = 0
                for x in df['removed_aspects_2']:
                    text[i] = self.remove(x, bag_of_words)
                    i += 1
                df['removed_aspects_3'] = text.values()
                df.to_csv(file_path + 'review' + str(a) + "_clean.csv", index=None)
            print(a)
            a += 1

    def list(self, directory):
        df = pd.read_csv(directory)
        col_one_list = []
        for row in df['removed_aspects_3']:
            row = ast.literal_eval(row)
            for word in row:
                # list_a = list(row.split(","))[0::2]
                # list_a = [self.remove_punctuation(x) for x in list_a]
                col_one_list.append(word)
        col_one_list = list(col_one_list)
        return col_one_list

    def list_aspects(self, file_path):
        a = 0
        text = {}
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                # df = pd.read_csv(entry)
                text[a] = self.list(entry)
                # text = pd.DataFrame(text, index=[0])
                # text = text.T
                a += 1
        text = pd.DataFrame.from_dict(text, orient='index')
        text = text.transpose()
        text = text.stack().reset_index()
        text.to_csv('C:/Users/Bosia/Desktop/aspects2.csv', index=None)

    def delete_duplicates(self, file):
        df = pd.read_csv(file)
        df = df.iloc[:, 2]
        df = df.dropna()
        df = df.drop_duplicates()
        print(len(df))
        df.to_csv(file)

    def new_files(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                columns = ['review_id', 'user_id', 'business_id', 'date', 'removed_aspects_3', 'new']
                df = df[columns]
                df.to_csv(f'C:/Users/Bosia/Desktop/done/review_{a}.csv', index=None)
                print(a)
                a += 1

    def add_df_friend(self, directory_users, directory_reviews):
        a = 0
        df_friend = self.create_friends_df(directory_users)
        for entry in os.scandir(directory_reviews):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                result = pd.merge(df, df_friend, how="inner", on=["user_id"])
                result.to_csv(f'C:/Users/Bosia/Desktop/done2/review_{a}.csv', index=None)
                print(a)
                a += 1

    def add_groups(self, directory_users, directory_reviews):
        a = 0
        df_friend = pd.read_csv(directory_users)
        for entry in os.scandir(directory_reviews):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                result = pd.merge(df, df_friend, how="inner", on=["user_id"])
                result.to_csv(entry.path, index=None)
                print(a)
                a += 1

    def create_friends_df(self, directory):
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
        df_merged = pd.concat(df_from_each_file, ignore_index=True)
        columns = ['user_id', 'name', 'friends']
        df_merged = df_merged[columns]
        return df_merged

    def clean_up(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                i = 0
                for x in df['new']:
                    df['new'][i] = self.sent_clean(x)
                    i += 1
                df.to_csv(f'C:/Users/Bosia/Desktop/test/review_{a}.csv', index=None)
                print(a)
                a += 1

    def sent_clean(self, sentence):
        inputTupples = ast.literal_eval(sentence)
        text = []
        for key, value in inputTupples.items():
            text.append(value)
        return text

    def neg_pos(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                i = 0
                df['pos'] = None
                df['obj'] = None
                df['neg'] = None
                for x in df['new']:
                    inputTupples = ast.literal_eval(x)
                    pos = []
                    neg = []
                    obj = []
                    for value in inputTupples:
                        neg.append(value[0])
                        obj.append(value[1])
                        pos.append(value[2])
                    df['neg'][i] = neg
                    df['obj'][i] = obj
                    df['pos'][i] = pos
                    i += 1
                df.to_csv(f'C:/Users/Bosia/Desktop/test/review_{a}.csv', index=None)
                print(a)
                a += 1

    def cal_average(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                aspects = df.groupby(['user_id'])['removed_aspects_3'].apply(','.join).reset_index()
                pos = df.groupby(['user_id'])['pos'].apply(','.join).reset_index()
                neg = df.groupby(['user_id'])['neg'].apply(','.join).reset_index()
                obj = df.groupby(['user_id'])['obj'].apply(','.join).reset_index()
                result = pd.merge(aspects, pos,  how="inner", on=["user_id"])
                result = pd.merge(result, neg,  how="inner", on=["user_id"])
                result = pd.merge(result, obj,  how="inner", on=["user_id"])
                i = 0
                for z in range(len(result)):
                    names = ast.literal_eval(result['removed_aspects_3'][i])
                    values_pos = ast.literal_eval(result['pos'][i])
                    values_neg = ast.literal_eval(result['neg'][i])
                    values_obj = ast.literal_eval(result['obj'][i])
                    if isinstance(values_pos, tuple):
                        temp = result['user_id'][i]
                        result.iloc[i] = self.averages(names, values_pos, values_obj, values_neg)
                        result['user_id'][i] = temp
                    i += 1
                pd.DataFrame.from_dict(result)
                result.to_csv(entry, index=None)
                print(a)
                a += 1

    def averages(self, names, values_pos, values_obj, values_neg):
        # Group the items by name.
        aspects = [i for sub in names for i in sub]
        pos = [i for sub in values_pos for i in sub]
        neg = [i for sub in values_neg for i in sub]
        obj = [i for sub in values_obj for i in sub]
        value_list = defaultdict(list)
        for aspect, pos, obj, neg in zip(aspects, pos, obj, neg):
            value_list[f"{aspect}_pos"].append(pos)
            value_list[f"{aspect}_obj"].append(obj)
            value_list[f"{aspect}_neg"].append(neg)
        result = {}
        for name, value in value_list.items():
            result[name] = [float(sum(value)) / float(len(value))]
        return self.reformat(result)

    def reformat(self, result):
        data = {}
        names = [key for key, val in result.items()]
        names = [x.split("_")[0] for x in names]
        names = list(dict.fromkeys(names))
        pos_key = "pos"
        pos = [val for key, val in result.items() if pos_key in key]
        neg_key = "neg"
        neg = [val for key, val in result.items() if neg_key in key]
        obj_key = "obj"
        obj = [val for key, val in result.items() if obj_key in key]
        data['removed_aspects_3'] = names
        data['pos'] = pos
        data['obj'] = obj
        data['neg'] = neg
        return data

    def score_sen(self, line):
        neg = ast.literal_eval(line.pos)
        pos = ast.literal_eval(line.neg)
        scores = []
        i = 0
        for x in pos:
            scores.append(x-neg[i])
            i += 1
        return scores

    def score(self, file_path):
        a = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                i = 0
                sentiment = []
                for x in df['pos']:
                    sentiment.append(self.score_sen(df.iloc[i]))
                    i += 1
                df['scores'] = sentiment
                df.to_csv(entry.path, index=None)
                print(a)
                a += 1




