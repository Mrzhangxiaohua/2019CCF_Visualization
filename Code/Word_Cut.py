# str = 'm.openapi.q7a.cn'
# result = str.replace(".", ",").replace(":", ",")
# split_result = result.split(",")
# string = ""
# for i in split_result:
#     if len(i) != 0:
#         string += i + " "
# print("----")
# print(string)

import pandas as pd
import numpy as np

def split_uri_result(str):
    if str != np.nan:
        result = str.replace("/", ",").replace("..","").replace("?",",").replace(".",",").replace("%",",").replace("=",",").replace("&", ",").replace("(",",").replace(")",",")
        result = result.replace("|",",").replace(":",",").replace("'",",").replace("---","").replace("@",",").replace("--",",").replace("*",",").replace("!",",").replace("\\",",")
        # print(result)
        split_result = result.split(",")
        # print(split_result)
        string = ""
        for i in split_result:
            if len(i) != 0:
                string += i + " "
    else:
        string = ""
    return string

def split_host_result(str):
    result = str.replace(".", ",").replace(":", ",")
    split_result = result.split(",")
    string = ""
    for i in split_result:
        if len(i) != 0:
            string += i + " "
    return string

def process_uri():
    # 对uri进行分词处理
    with open("../data/flow_used_data/crops.txt", "a") as file:
        df = pd.read_csv('../data/flow_used_data/2019_06_05_process_flow.csv')['uri'].replace([np.nan], '/').to_numpy().tolist()
        for i in range(len(df)):
            string = df[i]
            result = split_uri_result(string)
            # print(result)
            file.write(result + "\n")
    print("---------->host信息处理完毕")

def process_host():
    with open("../data/flow_used_data/crops_host.txt", "a") as file:
        df = pd.read_csv('../data/flow_used_data/2019_06_05_process_flow.csv')['host'].replace([np.nan], '/').to_numpy().tolist()
        for i in range(len(df)):
            string = df[i]
            result = split_host_result(string)
            # print(result)
            file.write(result + "\n")
    print("---------->host信息处理完毕")

process_host()