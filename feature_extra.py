from sklearn.ensemble import RandomForestClassifier
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

def feature_extract(index,max_depth,min_samples_split,min_samples_leaf):
    final_data = pd.read_csv('./data_new/norm_final_data_dis.csv')
    score = final_data['hsi_label']
    final_data = final_data.drop(columns=['hsi_label'])
    alg = RandomForestClassifier(bootstrap=True, n_estimators=100,min_samples_split = min_samples_split,min_samples_leaf=min_samples_leaf,
                                random_state=50,oob_score=True,max_depth=max_depth)  # 随机数生成种子
    alg.fit(final_data.values, score)

    combine_list = lambda item:[item[0],item[1]]
    feature_importance = list(map(combine_list,zip(final_data.columns,alg.feature_importances_)))

    feature_importance = pd.DataFrame(feature_importance,columns=["feature","importance"]).sort_values(by = "importance", ascending=False)
    filter_feature = feature_importance[feature_importance["importance"]>0]

    print(filter_feature)

    filter_feature.to_csv("./data_new/feature_extracted_{}.csv".format(index))
def adjustment():
    index = 1
    for max_depth in [5,20]:
        for min_samples_split in [2, 4, 8]:
            for min_samples_leaf in [1, 2]:
                feature_extract(index,max_depth,min_samples_split,min_samples_leaf)
                index = index +1
def comprehensive_eva():
    score = {}
    data = pd.read_csv("./data_new/norm_final_data_dis.csv")
    data = data.drop(columns=['hsi_label'])
    columns = data.columns.values.tolist()
    for i in range(len(columns)):
        score[columns[i]] = 0
    for i in range(10):
        features = pd.read_csv("./data_new/feature_extracted_{}".format(i))
        for j in range(len(columns)):
            score[columns[j]] += features[features['feature']==columns[j]]
if __name__ == "__main__":
    adjustment()
