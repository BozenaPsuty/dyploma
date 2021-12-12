import os
import json
import pandas as pd
import glob
import ast


class Cutting:

    def cut(self, jsondata, path, encoding, size):
        with open(jsondata, 'r', encoding=encoding) as f1:
            ll = [json.loads(line.strip()) for line in f1.readlines()]
            total = len(ll) // size
            for i in range(total):
                json.dump(ll[i * size:(i + 1) * size],
                          open(path + str(i) + ".json", 'w', encoding=encoding),
                          ensure_ascii=False, indent=True)

    def change_to_csv(self, directory, path):
        i = -1
        for entry in os.scandir(directory):
            if entry.path.endswith(".json"):
                i = i + 1
                df = pd.read_json(entry)
                df.to_csv(directory + path + str(i) + ".csv", index=None)

    def delete_instances_business(self, directory, X, name):
        i = -1
        for entry in os.scandir(directory):
            if entry.path.endswith(".csv"):
                i = i + 1
                df = pd.read_csv(entry)
                df[df["review_count"] > X].to_csv(directory + name + str(i) + "_new.csv", index=False)

    def combine(self, directory):
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
        df_merged = pd.concat(df_from_each_file, ignore_index=True)
        df_merged.to_csv("C:/Users/Bosia/Desktop/merged_reviews.csv")

    def list(self, directory, name):
        df = pd.read_csv(directory)
        col_one_list = df[name].tolist()
        return col_one_list

    def delete_instances_review(self, directory_business, directory_review):
        business_id_list = self.list(directory_business)
        i = -1
        for entry in os.scandir(directory_review):
            if entry.path.endswith(".csv"):
                i = i + 1
                df = pd.read_csv(entry)
                df = df[df['business_id'].isin(business_id_list)]
                df.to_csv(directory_review + 'review' + str(i) + "_new.csv", index=None)

    def split_by_group(self, file_path):
        df = pd.read_csv(file_path)
        grouped = df.groupby(df['business_id'])
        for name, group in grouped:
            group.to_csv(f"C:/Users/Bosia/Desktop/sen/{name}.csv")

    def get_only(self, file_path):
        i = -1
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                i = i + 1
                df = pd.read_csv(entry)
                df = df.iloc[:, 0:9]
                df.to_csv(file_path + 'review' + str(i) + "_new.csv", index=False)

    def delete_instances_user(self, directory, X):
        i = -1
        for entry in os.scandir(directory):
            if entry.path.endswith(".csv"):
                i = i + 1
                df = pd.read_csv(entry)
                df['friend_count'] = [len(X.split(",")) for X in df.friends]
                df[df['friend_count'] > X].to_csv(directory + 'user' + str(i) + ".csv", index=False)

    def table_user(self, file_path):
        df = pd.read_csv(file_path)
        columns = ['user_id', "name", "friends"]
        new = df[columns]
        new.to_csv("C:/Users/Bosia/Desktop/merged.csv")

    def smaller(self, file, file_path):
        i = -1
        users = pd.read_csv(file)
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                print(str(entry).split("'")[1])
                list_users = self.list(entry, 'user_id')
                df = users[users['user_id'].isin(list_users)]
                df.to_csv(file_path.replace('sen', 'user_csv') + str(entry).split("'")[1], index=None)
                print(i)
                i = i + 1

    def explode(self, file_path):
        i = 0
        for entry in os.scandir(file_path):
            if entry.path.endswith(".csv"):
                df = pd.read_csv(entry)
                df['removed_aspects_3'] = [ast.literal_eval(x) for x in df['removed_aspects_3']]
                df['scores'] = [ast.literal_eval(x) for x in df['scores']]
                df['pos'] = [ast.literal_eval(x)for x in df['pos']]
                df['neg'] = [ast.literal_eval(x) for x in df['neg']]
                df['obj'] = [ast.literal_eval(x) for x in df['obj']]
                df = df.apply(lambda x: x.explode() if x.name in ['removed_aspects_3', 'scores', 'pos', 'neg', 'obj'] else x)
                df.to_csv(entry.path.replace('sen', 'sen2'), index=None)
                print(i)
                i = i + 1
