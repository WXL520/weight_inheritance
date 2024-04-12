# from transformers import BertTokenizer, BertModel, BertForMaskedLM
# import numpy as np
#
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('3L256H-to-6L256H-1.000newparams-stacking-output/checkpoints/checkpoint-00400000')
# # text = "Replace me by any text you'd like."
# # encoded_input = tokenizer(text, return_tensors='pt')
# # output = model(**encoded_input)
# print(model.state_dict())
# np.savez('bert_stacking_weights.npz', **model.state_dict())


#####################################
# QQT, VProj, W1W2
#####################################
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

cols = 3
rows = 2
fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(25, 7))
axes = axes.flat
bert_base_weights = np.load('bert_stacking_weights.npz')
for k, v in bert_base_weights.items():
    print(k, v.shape)

# creating a colormap
colormap = sns.color_palette("Greys_r")

for i in range(6):
    query_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.query.weight']
    key_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.key.weight']
    value_weights = bert_base_weights[f'encoder.layer.{i}.attention.self.value.weight']
    proj_weights = bert_base_weights[f'encoder.layer.{i}.attention.output.dense.weight']
    fc1 = bert_base_weights[f'encoder.layer.{i}.intermediate.dense.weight'].T  # [768, 3072]
    fc2 = bert_base_weights[f'encoder.layer.{i}.output.dense.weight']  # [768, 3072]
    QKT = np.absolute(np.sqrt(256) * np.matmul(query_weights, key_weights.T))
    VProj = np.absolute(np.sqrt(256) * np.matmul(value_weights, proj_weights))
    W1W2 = np.absolute(np.sqrt(256) * np.matmul(fc1, fc2.T))

    # query_head0 = query_weights.reshape((-1, 12, 64))[:, 11, :]
    # key_head0 = key_weights.reshape((-1, 12, 64))[:, 11, :]
    # QKT_head = np.absolute(np.sqrt(768) * np.matmul(query_head0, key_head0.T))
    # QQT_head = np.absolute(np.sqrt(768) * np.matmul(query_head0, query_head0.T))[:32, :32]

    sns.heatmap(QKT, ax=axes[i], cmap=colormap)
    axes[i].set_title(f'layer - {i}')
plt.show()
