import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


# x = np.linspace(10, 400, num=40)
# y_scratch = [41.15, 20.12, 16.31, 14.58, 13.54, 12.81, 12.25, 11.84, 11.51, 11.20, 10.98, 10.74, 10.54,
#              10.39, 10.22, 10.09, 9.97, 9.83, 9.71, 9.58, 9.48, 9.38, 9.25, 9.17, 9.07, 8.99, 8.88, 8.79,
#              8.70, 8.62, 8.54, 8.44, 8.34, 8.27, 8.19, 8.11, 8.03, 7.96, 7.90, 7.86]
# y_scratch_pad = y_scratch + [0] * (len(x) - len(y_scratch))
# y_growth = [16.25, 14.76, 13.86, 13.15, 12.62, 12.23, 11.89, 11.61, 11.43, 11.16, 10.99, 10.81, 10.63, 10.53,
#             10.36, 10.27, 10.13, 10.00, 9.88, 9.77, 9.67, 9.58, 9.47, 9.36, 9.28, 9.21, 9.09, 9.01, 8.91,
#             8.81, 8.74, 8.63, 8.55, 8.47, 8.37, 8.30, 8.21, 8.14, 8.07, 8.03]
# y_growth2 = [16.16, 14.77, 13.84, 13.14, 12.63, 12.25, 11.92, 11.64, 11.40, 11.18, 11.02, 10.81, 10.64, 10.51, 10.37,
#              10.26, 10.13, 10.00, 9.88, 9.78, 9.68, 9.59, 9.47, 9.36, 9.27, 9.20, 9.09, 9.00, 8.91, 8.82, 8.74, 8.64,
#              8.56, 8.48, 8.39, 8.32, 8.23, 8.16, 8.10, 8.06]
# y_small = [101.68, 36.06, 27.80, 24.63, 22.77, 21.61, 20.78, 20.18, 19.69, 19.32, 18.99, 18.71, 18.45, 18.27,
#            18.04, 17.92, 17.73, 17.60, 17.47, 17.34, 17.19, 17.09, 16.95, 16.83, 16.75, 16.63, 16.54, 16.40, 16.30,
#            16.19, 16.09, 15.97, 15.87, 15.76, 15.67, 15.56, 15.47, 15.38, 15.30, 15.26]
#
# plt.xlabel('k step')
# plt.ylabel('log ppl')
# plt.plot(x, np.log(y_scratch), marker='o', markersize=3)
# plt.plot(x, np.log(y_growth), marker='o', markersize=3)
# plt.plot(x, np.log(y_small), marker='o', markersize=3)
# plt.plot(x, np.log(y_growth2), marker='o', markersize=3)
# plt.legend(['scratch_6L_512H', 'grow_from_400k_steps', 'small_3L_256H', 'grow2_from_200k_steps'])
# plt.show()

x = np.linspace(10, 400, num=40)

# eval_new02 = []
# with Path('0.2new.log').open('r') as fr:
#     for i, line in enumerate(fr):
#         eval_ppl = re.search('eval_ppl=*', line).span()
#         eval_new02.append(float(line[eval_ppl[1]:-3]))
# eval_new02_pad = eval_new02 + [0] * (len(x) - len(eval_new02))
#
# eval_new05 = []
# with Path('0.5new.log').open('r') as fr:
#     for i, line in enumerate(fr):
#         eval_ppl = re.search('eval_ppl=*', line).span()
#         eval_new05.append(float(line[eval_ppl[1]:-3]))
# eval_new05_pad = eval_new05 + [0] * (len(x) - len(eval_new05))
#
# eval_new15 = []
# with Path('1.5new.log').open('r') as fr:
#     for i, line in enumerate(fr):
#         eval_ppl = re.search('eval_ppl=*', line).span()
#         eval_new15.append(float(line[eval_ppl[1]:-3]))
# eval_new15_pad = eval_new15 + [0] * (len(x) - len(eval_new15))
#
# eval_new20 = []
# with Path('2.0new.log').open('r') as fr:
#     for i, line in enumerate(fr):
#         eval_ppl = re.search('eval_ppl=*', line).span()
#         eval_new20.append(float(line[eval_ppl[1]:-3]))
# eval_new20_pad = eval_new20 + [0] * (len(x) - len(eval_new20))
#
# eval_new50 = []
# with Path('5.0new.log').open('r') as fr:
#     for i, line in enumerate(fr):
#         eval_ppl = re.search('eval_ppl=*', line).span()
#         eval_new50.append(float(line[eval_ppl[1]:-3]))
# eval_new50_pad = eval_new50 + [0] * (len(x) - len(eval_new50))

eval_new1_copy = []
train_new1_copy = []
with Path('1.0new_copy.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        eval_new1_copy.append(float(line[eval_ppl[1]:-3]))
        train_new1_copy.append(float(line[train_loss[1]:-67]))

eval_new1_rand = []
train_new1_rand = []
with Path('1.0new_rand.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        eval_new1_rand.append(float(line[eval_ppl[1]:-3]))
        train_new1_rand.append(float(line[train_loss[1]:-67]))

eval_new2_rand = []
train_new2_rand = []
with Path('2.0new_rand.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        eval_new2_rand.append(float(line[eval_ppl[1]:-3]))
        train_new2_rand.append(float(line[train_loss[1]:-67]))
eval_new2_rand_pad = eval_new2_rand + [0] * (len(x) - len(eval_new2_rand))
train_new2_rand_pad = train_new2_rand + [0] * (len(x) - len(train_new2_rand))

eval_new13_rand = []
train_new13_rand = []
with Path('1.3new_rand.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        eval_new13_rand.append(float(line[eval_ppl[1]:-3]))
        train_new13_rand.append(float(line[train_loss[1]:-67]))

eval_new1_rand_freeze_old = []
train_new1_rand_freeze_old = []
with Path('1.0new_rand_freeze_old.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        eval_new1_rand_freeze_old.append(float(line[eval_ppl[1]:-3]))
        train_new1_rand_freeze_old.append(float(line[train_loss[1]:-67]))
eval_new1_rand_freeze_old_pad = eval_new1_rand_freeze_old + [0] * (len(x) - len(eval_new1_rand_freeze_old))
train_new1_rand_freeze_old_pad = train_new1_rand_freeze_old + [np.log(0)] * (len(x) - len(train_new1_rand_freeze_old))

eval_new1_rand_lora_old = []
train_new1_rand_lora_old = []
with Path('1.0new_rand_lora_old.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        eval_new1_rand_lora_old.append(float(line[eval_ppl[1]:-3]))
        train_new1_rand_lora_old.append(float(line[train_loss[1]:-67]))
eval_new1_rand_lora_old_pad = eval_new1_rand_lora_old + [0] * (len(x) - len(eval_new1_rand_lora_old))
train_new1_rand_lora_old_pad = train_new1_rand_lora_old + [np.log(0)] * (len(x) - len(train_new1_rand_lora_old))

y_small = [101.68, 36.06, 27.80, 24.63, 22.77, 21.61, 20.78, 20.18, 19.69, 19.32, 18.99, 18.71, 18.45, 18.27,
           18.04, 17.92, 17.73, 17.60, 17.47, 17.34, 17.19, 17.09, 16.95, 16.83, 16.75, 16.63, 16.54, 16.40, 16.30,
           16.19, 16.09, 15.97, 15.87, 15.76, 15.67, 15.56, 15.47, 15.38, 15.30, 15.26]

standard = []
train_standard = []
with Path('standard.log').open('r') as fr:
    for i, line in enumerate(fr):
        eval_ppl = re.search('eval_ppl=*', line).span()
        train_loss = re.search('train_loss=*', line).span()
        standard.append(float(line[eval_ppl[1]:-3]))
        train_standard.append(float(line[train_loss[1]:-67]))
standard_pad = standard + [0] * (len(x) - len(standard))

plt.xlabel('k step')
plt.ylabel('eval loss')
plt.plot(x, np.log(eval_new1_copy), marker='o', markersize=3)
plt.plot(x, np.log(y_small), marker='o', markersize=3)
plt.plot(x, np.log(eval_new1_rand), marker='o', markersize=3)
plt.plot(x, np.log(eval_new13_rand), marker='o', markersize=3)
plt.plot(x, np.log(eval_new2_rand_pad), marker='o', markersize=3)
plt.plot(x, np.log(eval_new1_rand_freeze_old_pad), marker='o', markersize=3)
plt.plot(x, np.log(eval_new1_rand_lora_old_pad), marker='o', markersize=3)
plt.plot(x, np.log(standard_pad), marker='o', markersize=3)
# plt.ylabel('train loss')
# plt.plot(x, train_new1_copy, marker='o', markersize=3)
# plt.plot(x, train_new1_rand, marker='o', markersize=3)
# plt.plot(x, train_new13_rand, marker='o', markersize=3)
# plt.plot(x, train_new2_rand_pad, marker='o', markersize=3)
# plt.plot(x, train_new1_rand_freeze_old_pad, marker='o', markersize=3)
# plt.plot(x, train_new1_rand_lora_old_pad, marker='o', markersize=3)
# plt.plot(x, train_standard, marker='o', markersize=3)
plt.legend(['1.0new_copy', 'y_small', '1.0new_rand', '1.3new_rand', '2.0new_rand', '1.0new_rand_freeze_old', '1.0new_rand_lora_old', 'standard'])
plt.show()



# subnet0 = [0.1242, 0.1092, 0.0930, 0.0904, 0.0975, 0.0896, 0.0905, 0.0866, 0.0788, 0.0819, 0.0755, 0.0869, 0.0938, 0.0893, 0.1089, 0.0870, 0.0858, 0.0803, 0.0885, 0.0831, 0.0839, 0.0845]
# subnet1 = [0.1884, 0.1985, 0.1589, 0.1593, 0.1786, 0.1713, 0.1662, 0.1690, 0.1585, 0.1561, 0.1575, 0.1669, 0.1657, 0.1647, 0.1665, 0.1622, 0.1922, 0.1539, 0.1873, 0.1610, 0.1669, 0.1689]
# plt.xlabel('k step')
# plt.ylabel('log ppl')
# x = np.linspace(10, 400, num=40)
# subnet0_pad = subnet0 + [0] * (len(x) - len(subnet0))
# subnet1_pad = subnet1 + [0] * (len(x) - len(subnet1))
# plt.plot(x, np.log(subnet0_pad), marker='o', markersize=3)
# plt.plot(x, np.log(subnet1_pad), marker='o', markersize=3)
# plt.show()



