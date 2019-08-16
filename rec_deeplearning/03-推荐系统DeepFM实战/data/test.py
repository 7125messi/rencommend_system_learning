import pandas as pd

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/test.csv'

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]

dfTrain = pd.read_csv(TRAIN_FILE)
dfTest = pd.read_csv(TEST_FILE)
print(dfTrain.shape) # (10000,59)

df = pd.concat([dfTrain,dfTest])
print(df.shape) # (12000, 59)

feature_dict = {}
total_feature = 0

for col in df.columns:
    if col in IGNORE_COLS:
        continue
    elif col in NUMERIC_COLS:
        feature_dict[col] = total_feature
        total_feature += 1
    else:
        unique_val = df[col].unique()
        feature_dict[col] = dict(zip(unique_val,range(total_feature,total_feature+len(unique_val))))
        total_feature += len(unique_val)
print(total_feature)
print(feature_dict)

# 训练集进行转化
print(dfTrain.columns)
train_y = dfTrain['target'].values.tolist()
dfTrain.drop(['target','id'],axis=1,inplace=True)
train_feature_index = dfTrain.copy()
train_feature_value = dfTrain.copy()

for col in train_feature_index.columns:
    if col in IGNORE_COLS:
        train_feature_index.drop(col,axis=1,inplace=True)
        train_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        train_feature_index[col] = feature_dict[col]
    else:
        train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
        train_feature_value[col] = 1

# 测试集进行转化
print(dfTest.columns)
test_ids = dfTest['id'].values.tolist()
dfTest.drop('id',axis=1,inplace=True)
test_feature_index = dfTest.copy()
test_feature_value = dfTest.copy()

for col in test_feature_value.columns:
    if col in IGNORE_COLS:
        test_feature_index.drop(col,axis=1,inplace=True)
        test_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        test_feature_value[col] = feature_dict[col]
    else:
        test_feature_index[col] = test_feature_index[col].map(feature_dict[col])
        test_feature_value[col] = 1

print(train_feature_index.shape)
print(train_feature_value.shape)
print(test_feature_index.shape)
print(test_feature_value.shape)