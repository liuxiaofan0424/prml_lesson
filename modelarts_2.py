import requests
import json
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report


def get_token():
    url = "https://iam.cn-north-4.myhuaweicloud.com/v3/auth/tokens"  # 注意自己的节点
    headers = {"Content-Type": "application/json"}
    data = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        "name": "",  # 账户
                        "password": "",  # 密码
                        "domain": {
                            "name": ""  # 域账户，普通账户这里就还是填账户
                        }
                    }
                }
            },
            "scope": {
                "project": {
                    "name": "cn-north-4"  # 注意自己的节点
                }
            }
        }
    }

    data = json.dumps(data)

    r = requests.post(url, data=data, headers=headers)
    print(r.headers['X-Subject-Token'])


def data2dict():
    df = pd.read_csv('train.csv', names=["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6", "attr_7"])
    target_list = df.copy()['attr_7'].to_list()
    count = df.shape[0]
    # df['attr_7'] = ""
    bank_dict = df.to_dict(orient='records')
    # print(count)
    # print(bank_dict)
    # print(target_list)
    return count, bank_dict, target_list, df


def predict(count, bank_dict, target_list, df):
    X_Auth_Token = ''  # 放入自己的X_Auth_Token
    url = ''  # 华为自动学习给出的url

    # json example
    # data = {
    #     "data":
    #         {
    #             "count": 2,
    #             "req_data":
    #                 [
    #                     {
    #                         "attr_1": "39",
    #                         "attr_2": "technician",
    #                         "attr_3": "single",
    #                         "attr_4": "secondary",
    #                         "attr_5": "yes",
    #                         "attr_6": "no",
    #                         "attr_7": ""
    #                     },
    #                     {
    #                         "attr_1": "34",
    #                         "attr_2": "services",
    #                         "attr_3": "single",
    #                         "attr_4": "secondary",
    #                         "attr_5": "yes",
    #                         "attr_6": "no",
    #                         "attr_7": ""
    #                     }
    #                 ]
    #         }
    # }

    # all of data
    data = {
        "data":
            {
                "count": count,
                "req_data":
                    bank_dict
            }
    }

    data = json.dumps(data)
    # print(data)

    headers = {
        "content-type": "application/json",
        'X-Auth-Token': X_Auth_Token
    }
    response = requests.request("POST", url, data=data, headers=headers)
    result_dict = json.loads(response.text)

    predictions = result_dict['result']['resp_data']

    # for i in predictions:
    #     prediction = i['predictioncol']
    #     pred_list.append(prediction)
    #     # probability = i['probabilitycol']
    #     # print(prediction, probability)

    # acc_score = accuracy_score(target_list, pred_list)
    # print('共预测', len(pred_list), '条：')
    # print('acc:', acc_score)
    # print('pred_list:', pred_list)
    # print('target_list:', target_list)

    pred_df = pd.DataFrame.from_dict(predictions).drop_duplicates()
    concat_df = pd.merge(df, pred_df)
    print(concat_df)

    report = classification_report(concat_df['attr_7'], concat_df['predictioncol'])
    print('共预测', df.shape[0], '条')
    print(report)


if __name__ == '__main__':
    # get_token()
    count, bank_dict, target_list, df = data2dict()
    predict(count, bank_dict, target_list, df)
