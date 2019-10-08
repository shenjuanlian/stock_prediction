import datetime

import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def claculate_RDP(dataset,days,amount,sign='index'):
    if sign == 'index':
        newdata = pd.DataFrame(columns=['Date','Open_RDP{}'.format(days),'High_RDP{}'.format(days),'Low_RDP{}'.format(days),'Close_RDP{}'.format(days),'Adj Close_RDP{}'.format(days),'Volume_RDP{}'.format(days)])
    else:
        newdata = pd.DataFrame(columns=['Date','Close_RDP{}'.format(days),'Open_RDP{}'.format(days),'High_RDP{}'.format(days),'Low_RDP{}'.format(days)])
    op_data = dataset.drop(columns=['Date'])
    for i in range(amount):
        newline = (op_data.iloc[i]-op_data.iloc[i+days])/op_data.iloc[i+days]
        date_ser = pd.Series(dataset['Date'][i])
        newline = date_ser.append(newline,ignore_index=True)
        if sign == 'index':
            newline.index =['Date','Open_RDP{}'.format(days),'High_RDP{}'.format(days),'Low_RDP{}'.format(days),'Close_RDP{}'.format(days),'Adj Close_RDP{}'.format(days),'Volume_RDP{}'.format(days)]
        else:
            newline.index =['Date','Close_RDP{}'.format(days),'Open_RDP{}'.format(days),'High_RDP{}'.format(days),'Low_RDP{}'.format(days)]
        newdata = newdata.append(newline, ignore_index=True)
    # print(newdata)
    return newdata

def complete_blank(dataset):
    date_time = pd.to_datetime(dataset['Date'])
    j = 0
    newdata = dataset
    for i in range(len(date_time)):
        while date_time[i].date() != date_time[0].date()-datetime.timedelta(j):
           above = newdata[:j]
           below = newdata[j:]
           insertRow = dataset.iloc[i]
           newdata = above.append(insertRow, ignore_index=True).append(below, ignore_index=True)
           j+=1
        j+=1
    # print(newdata)
    return newdata
def cal_trend(data_RDP_1):
    labels = [-1]
    for i in range(len(data_RDP_1)):
        if data_RDP_1['Close_RDP1'][i] >= 0:
            labels.append(1)
        else:
            labels.append(0)
    labels.pop()
    data_RDP_1['label'] = labels
    return data_RDP_1

def export_hsi():
    _hsi = pd.read_csv("./data_new/HSI2010-2019.csv")
    hsi_RDP_1 = claculate_RDP(_hsi, 1, 2192)
    hsi_RDP_1 = cal_trend(hsi_RDP_1)
    print("hsi_RDP_1 finished ---------------------------------")
    hsi_RDP_7 = claculate_RDP(_hsi, 7, 2192)
    print("hsi_RDP_7 finished ---------------------------------")
    hsi_RDP_30 = claculate_RDP(_hsi, 30, 2192)
    print("hsi_RDP_30 finished ---------------------------------")
    hsi = pd.merge(pd.merge(hsi_RDP_1, hsi_RDP_7), hsi_RDP_30)
    final_hsi = complete_blank(hsi)
    final_hsi.to_csv("./data_new/hsi.csv")
def export_sp500():
    _sp500 = pd.read_csv("./data_new/GSPC2010-2019.csv")
    sp500_RDP_1 = claculate_RDP(_sp500, 1, 2235)
    sp500_RDP_7 = claculate_RDP(_sp500, 7, 2235)
    sp500_RDP_30 = claculate_RDP(_sp500, 30, 2235)
    sp500 = pd.merge(pd.merge(sp500_RDP_1, sp500_RDP_7), sp500_RDP_30)
    final_sp500 = complete_blank(sp500)
    final_sp500.to_csv("./data_new/sp500.csv")
# def export_CNYHKD_test():
#     cnyhkd_tet = pd.read_csv("./data/CNY_HKD2017-2019.csv")
#     for i in range(len(cnyhkd_tet)):
#         date_list = cnyhkd_tet['Date'][i][:-1].replace('年','-').replace('月','-').split('-')
#         _date = date_list[1]+'/'+date_list[2]+'/'+date_list[0]
#         cnyhkd_tet['Date'][i] = _date
#     cnyhkd_tet = cnyhkd_tet.drop(columns=['percentage'])
#     cnyhkd_RDP_1 = claculate_RDP(cnyhkd_tet,1,440,'currency')
#     cnyhkd_RDP_7 = claculate_RDP(cnyhkd_tet, 7, 440,'currency')
#     cnyhkd_RDP_30 = claculate_RDP(cnyhkd_tet, 30, 440,'currency')
#     cnyhkd = pd.merge(pd.merge(cnyhkd_RDP_1,cnyhkd_RDP_7),cnyhkd_RDP_30)
#     final_snyhkd = complete_blank(cnyhkd)
#     print(final_snyhkd)
#     final_snyhkd.to_csv("./data/final_cnyhkd_test.csv",index=False)
# def export_final_hsi_train():
#     hsi_train = pd.read_csv("./data/HSI2013-2018.csv")
#     hsi_RDP_1 = claculate_RDP(hsi_train,1,1200)
#     hsi_RDP_1 = cal_trend(hsi_RDP_1)
#     hsi_RDP_7 = claculate_RDP(hsi_train, 7, 1200)
#     hsi_RDP_30 = claculate_RDP(hsi_train, 30, 1200)
#     hsi = pd.merge(pd.merge(hsi_RDP_1, hsi_RDP_7), hsi_RDP_30)
#     final_hsi = complete_blank(hsi)
#     final_hsi.to_csv("./data/final_hsi_train.csv", index=False)
# def export_final_sp500_train():
#     sp500_train = pd.read_csv("./data/SP500_2013-2018.csv")
#     sp500_RDP_1 = claculate_RDP(sp500_train, 1, 1230)
#     sp500_RDP_7 = claculate_RDP(sp500_train, 7, 1230)
#     sp500_RDP_30 = claculate_RDP(sp500_train, 30, 1230)
#     sp500 = pd.merge(pd.merge(sp500_RDP_1, sp500_RDP_7), sp500_RDP_30)
#     final_sp500 = complete_blank(sp500)
#     final_sp500.to_csv("./data/final_sp500_train.csv", index=False)
def merge_data():
    hsi = pd.read_csv("./data_new/hsi.csv")
    hsi.drop(columns = ['Date'],inplace=True)
    hsi_columns = hsi.columns.values.tolist()
    for i in range(len(hsi_columns)):
        hsi_columns[i] = "hsi_"+hsi_columns[i]
    hsi.columns = hsi_columns

    sp500 = pd.read_csv("./data_new/sp500.csv")
    sp500.drop(columns=['Date'], inplace=True)
    sp500_columns = sp500.columns.values.tolist()
    print(sp500_columns)
    for i in range(len(sp500_columns)):
        sp500_columns[i] = "sp500_"+sp500_columns[i]
    sp500.columns = sp500_columns
    final_data = pd.concat([hsi,sp500],axis=1)
    final_data.to_csv("./data_new/final_data.csv",index=False)
# def merge_testset():
#     hsi_test = pd.read_csv("./data/final_hsi_test.csv")
#     sp500_test = pd.read_csv("./data/final_sp500_test.csv")
#     final_test = pd.merge(hsi_test, sp500_test)
#     final_test = final_test.drop(columns=['Date'])
#     final_test.to_csv("./data/final_test.csv", index=False)
def normalization():
    dataset = pd.read_csv("./data_new/final_data.csv")
    dataset.dropna(axis=0, how='any', inplace=True)
    data_columns = dataset.columns.values.tolist()
    for i in data_columns:
        data = dataset[i]
        dataset[i] = (data - np.min(data)) / (np.max(data) - np.min(data))
    dataset = dataset.sort_index(axis=0,ascending=False)
    print(dataset.head())
    dataset.to_csv("./data_new/norm_final_data_dis.csv",index=False)
# def appendAndNorm():
#     trainset = pd.read_csv("./data/final_train.csv")
#     testset = pd.read_csv("./data/final_test.csv")
#     out = testset.append(trainset)
#     out.dropna(axis=0,how="any",inplace=True)
#     out_columns = out.columns.values.tolist()
#     for i in out_columns:
#         data = out[i]
#         out[i] = (data - np.min(data)) / (np.max(data) - np.min(data))
#     out.to_csv("./fine_data/total_data.csv",index=False)
if __name__ == '__main__':
    normalization()