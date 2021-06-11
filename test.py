from random import sample
import pandas as pd
import numpy as np

column_names = ["image", "label"]
ans = pd.DataFrame(columns=column_names)
ans1 = pd.DataFrame(columns=column_names)
sample_pred_dict1 = {
    'image': 'a',
    'label': 0
}
sample_pred_dict2 = {
    'image': 'a',
    'label': 1
}
sample_pred_dict3 = {
    'image': 'b',
    'label': 2
}

ans = ans.append(sample_pred_dict1, ignore_index=True)
ans = ans.append(sample_pred_dict3, ignore_index=True)
ans1 = ans1.append(sample_pred_dict3, ignore_index=True)
ans1 = ans1.append(sample_pred_dict2, ignore_index=True)

# print(ans)
# print(ans1)

assert(len(ans) == len(ans1))
ans_order = np.asarray(ans.iloc[0:, 0]).argsort()
ans1_order = np.asarray(ans1.iloc[0:, 0]).argsort()
print(ans_order)
tot = 0
acc = 0
for i in range(len(ans)):
    tot += 1
    print(ans.iloc[ans_order[i]][0], ans1.iloc[ans1_order[i]][0])
    print(ans.iloc[ans_order[i]][1], ans1.iloc[ans1_order[i]][1])
    acc += ans.iloc[ans_order[i]][1] == ans1.iloc[ans1_order[i]][1]
print(tot, acc)
