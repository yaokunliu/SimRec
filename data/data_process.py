import collections
import json
import pandas as pd
from tqdm import tqdm


def fix_data_type_error(dataset_name):
    data = []
    with open(f"meta_{dataset_name}.json") as f:
        for line in tqdm(f.readlines()):
            data.append(eval(line))

    # 保存为json lines文件
    with open(f"meta_{dataset_name}_new.json", "w") as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")


def convert_Retailrocket():
    df1 = pd.read_csv("item_properties_part1.csv")
    df2 = pd.read_csv("item_properties_part2.csv")
    df = pd.concat([df1, df2])

    # 去除property不为categoryid的行
    df = df[df.property == "categoryid"]

    # 将itemid相同的行的value合并
    data = collections.defaultdict(list)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        data[row["itemid"]].append(row["value"])

    # 保存为json lines文件
    with open("meta_Retailrocket_new.json", "w") as f:
        for key, value in tqdm(data.items()):
            f.write(json.dumps({"asin": key, "categories": [list(set(value))]}) + "\n")


def convert_Yelp():
    df = pd.read_json("yelp_academic_dataset_business.json", lines=True)

    df = df[["business_id", "categories"]]

    def process_fn(x):
        if x is None:
            return [[]]
        else:
            return [x.split(", ")]

    df["categories"] = df["categories"].apply(lambda x: process_fn(x))

    # 保存为json lines文件
    with open("meta_Yelp_new.json", "w") as f:
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            f.write(json.dumps({"asin": row["business_id"], "categories": row["categories"]}) + "\n")


if __name__ == '__main__':
    # fix_data_type_error("Beauty")
    # fix_data_type_error("Books")
    # convert_Retailrocket()
    convert_Yelp()
