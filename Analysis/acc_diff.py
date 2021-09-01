
# model diff
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['Logistic', 'SVM', 'RF', 'NN']

rect_acc = [0.559, 0.537, 0.397, 0.543]
rect_std = [0.007, 0.017, 0.014, 0.013]

x_1 = 2*np.arange(len(rect_acc))
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 6))
plt.bar(x_1, rect_acc, color='#0A81AB')

plt.errorbar(x_1, rect_acc, rect_std, color='black', ls='none', capsize=3)

for x, y in zip(x_1, rect_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy in different model', font_title)

plt.xticks(x_1, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,7.5)
plt.plot([-1.5,7.5], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_model.jpg'))
plt.close()

#%% scale acc comparasion bar
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['M1', 'M2', 'M2+M1', 'No_scale']

rect_acc = [0.559, 0.457, 0.532, 0.414]
rect_std = [0.007, 0.011, 0.015, 0.015]

x_1 = 2*np.arange(len(rect_acc))
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 6))
plt.bar(x_1, rect_acc, color='#0A81AB')

plt.errorbar(x_1, rect_acc, rect_std, color='black', ls='none', capsize=3)

for x, y in zip(x_1, rect_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy in different scale method', font_title)

plt.xticks(x_1, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,8)
plt.plot([-1.5,8], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_scale.jpg'))
plt.close()



#%% roi diff
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')


labels = ['vis1', 'vis2', 'pmm', 'vmm', 'visual_all']

rect_acc = [0.233, 0.538, 0.278, 0.223, 0.557]
rect_std = [0.007, 0.016, 0.013, 0.015, 0.008]

x_1 = 2*np.arange(len(rect_acc))
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 6))
plt.bar(x_1, rect_acc, color='#0A81AB')

plt.errorbar(x_1, rect_acc, rect_std, color='black', ls='none', capsize=3)

for x, y in zip(x_1, rect_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy in different ROIs', font_title)

plt.xticks(x_1, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,10)
plt.plot([-1.5,10], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_roi.jpg'))
plt.close()


#%% cv diff

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['group_run', 'group_sess', 'sess1', 'sess2', 'sess3', 'sess4']
rect1_acc = [0.394, 0.383, 0.195, 0.272, 0.269, 0.291]
rect1_std = [0.012, 0.001, 0.001, 0.001, 0.001, 0.001]

rect2_acc = [0.559, 0.548, 0.306, 0.310, 0.378, 0.399]
rect2_std = [0.007, 0.001, 0.001, 0.001, 0.001, 0.001]

x_1 = 2.5*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='single_trial', color='#F54748')
plt.bar(x_2, rect2_acc, label='mean_pattern', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.02, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy in different CV method', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,14)
plt.plot([-1.5,14], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_cv.jpg'))
plt.close()


#%% acc mean pattern 
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['single trial', 'mean_pattern']
rect_acc = [0.394, 0.559]
rect_std = [0.012, 0.007]


x_1 = 1.5*np.arange(len(rect_acc))

font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':12}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect_acc, color='#0A81AB', width=0.6)
plt.errorbar(x_1, rect_acc, rect_std, color='black', ls='none', capsize=3)

for x, y in zip(x_1, rect_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy in single trial and mean pattern', font_title)

plt.xticks(x_1, labels, fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1,2.5)
plt.plot([-1, 2.5], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_mean_pattern.jpg'))
plt.close()

#%% acc mean pattern 
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['single trial', 'mean_pattern']
rect_acc = [0.099, 0.098]
rect_std = [0.001, 0.001]


x_1 = 1.5*np.arange(len(rect_acc))

font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':12}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect_acc, color='#0A81AB', width=0.6)
plt.errorbar(x_1, rect_acc, rect_std, color='black', ls='none', capsize=3)

for x, y in zip(x_1, rect_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy using Lasso', font_title)

plt.xticks(x_1, labels, fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.3)

ax.set_xlim(-1,2.5)
plt.plot([-1, 2.5], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_lasso.jpg'))
plt.close()

#%% num prior acc comparasion bar
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['sub-02', 'sub-03']
rect1_acc = [0.337, 0.488]
rect1_std = [0.004, 0.009]

rect2_acc = [0.397, 0.594]
rect2_std = [0.005, 0.010]

x_1 = 2.5*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='single_trial', color='#F54748')
plt.bar(x_2, rect2_acc, label='mean_pattern', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.02, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy using whole brain', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,5)
plt.plot([-1.5,5], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_whole_brain.jpg'))
plt.close()


#%% roi diff

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['V1','V2','V3','V4','V8', 'FFC','VVC']
rect1_acc = [0.194, 0.176, 0.215, 0.239, 0.261, 0.289, 0.307]
rect1_std = [0.011, 0.009, 0.015, 0.015, 0.009, 0.010, 0.012]

rect2_acc = [0.179, 0.198, 0.286, 0.475, 0.298, 0.400, 0.352, ]
rect2_std = [0.010, 0.011, 0.015, 0.011, 0.008, 0.011, 0.011, ]

x_1 = 2.5*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':8}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='sub-02', color='#F54748')
plt.bar(x_2, rect2_acc, label='sub-03', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.02, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_title)
ax.set_title('Accuracy in different ROI', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.6)

ax.set_xlim(-1.5,17)
plt.plot([-1.5,17], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_roi.jpg'))
plt.close()

#%% denoised diff
# including beta & tvalue, denoised, wm

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['beta', 'tvalue', 'beta-denoised','beta-denoised-wm','tvalue-denoised-wm' ]
rect1_acc = [0.358, 0.364, 0.299, 0.314, 0.315]
rect1_std = [0.006, 0.005, 0.007, 0.004, 0.007]

rect2_acc = [0.572, 0.575, 0.508, 0.506, 0.519]
rect2_std = [0.009, 0.008, 0.007, 0.011, 0.011]

x_1 = 3.5*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':8}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='single-trial', color='#F54748')
plt.bar(x_2, rect2_acc, label='mean-pattern', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.01, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.01, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_title)
ax.set_title('Accuracy in different response estimate', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,16)
plt.plot([-1.5,16], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_denoised.jpg'))
plt.close()

#%% model diff
# LDA and logistic

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['Logistic','LDA', 'SVM']
rect1_acc = [0.488, 0.523, 0.499]
rect1_std = [0.009, 0.017, 0.011]

rect2_acc = [0.594, 0.615, 0.594]
rect2_std = [0.010, 0.010, 0.010]


x_1 = 3*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':8}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='sub-02', color='#F54748')
plt.bar(x_2, rect2_acc, label='sub-03', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.01, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.01, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_title)
ax.set_title('Accuracy in different model', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,8)
plt.plot([-1.5,8], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_model2.jpg'))
plt.close()


#%% highly responsive
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['active', 'discrim', 'stability']
rect1_acc = [0.390, 0.469, 0.606]
rect1_std = [0.009, 0.014, 0.006]

rect2_acc = [0.523, 0.615, 0.652]
rect2_std = [0.017, 0.010, 0.006]

x_1 = 2.5*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='sub-02', color='#F54748')
plt.bar(x_2, rect2_acc, label='sub-03', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.02, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.02, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy in different voxel selection method', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.7)

ax.set_xlim(-1.5,7)
plt.plot([-1.5,7], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_voxel_selection.jpg'))
plt.close()

#%% binary classification in different mean times

import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['single_trial','mean_2','mean_3','mean_4',
          'mean_5','mean_6','mean_7','mean_8']
rect1_acc = [0.771, 0.778, 0.858, 0.890, 0.921, 0.925, 0.949, 0.950]
rect1_std = [0.002, 0.003, 0.004, 0.003, 0.004, 0.006, 0.005, 0.007]

rect2_acc = [0.799, 0.797, 0.882, 0.906, 0.936, 0.951, 0.966, 0.975]
rect2_std = [0.002, 0.003, 0.004, 0.003, 0.003, 0.004, 0.006, 0.001]

x_1 = 3*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':20}
font_other = {'family': 'arial', 'weight': 'bold', 'size':8}
font_legend = {'family': 'arial', 'weight': 'bold', 'size':14}

plt.figure(figsize=(15, 10))

plt.bar(x_1, rect1_acc, label='sub-02', color='#F54748')
plt.bar(x_2, rect2_acc, label='sub-03', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.008, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.008, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_title)
ax.set_title('Accuracy of different mean times in binary classification', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=14)
plt.yticks(fontproperties='arial', weight='bold', size=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.set_ylim(0.5,1)
ax.set_xlim(-2,24)

## make twin axes
ax2 = ax.twinx()
test_sample = [288, 144, 96, 72, 57, 48, 41, 36]
ax2.plot((x_1 + x_2)/2, test_sample, ls='--', marker='o', color='#002651',
         label='n_sample in test set',)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_linewidth(1.5)
ax2.set_yticklabels(labels=np.linspace(0,300,7, dtype=int), fontdict=font_legend)
ax2.set_ylabel('Num', font_title, )

# merge legend in differnet axes
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines+lines2, labels+labels2, loc='upper left', 
           bbox_to_anchor=(0.01, 0.93), prop=font_legend)

plt.savefig(pjoin(out_path, 'acc_binary.jpg'))
plt.close()

#%% 10 subject decoding result
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-08', 'sub-09', 'sub-10']
rect_acc = [0.601, 0.606, 0.652, 0.712, 0.654, 0.582, 0.673, 0.555, 0.544,]
rect_std = [0.005, 0.006, 0.006, 0.008, 0.007, 0.010, 0.006, 0.009, 0.010,]


x_1 = 1.5*np.arange(len(rect_acc))

font_title = {'family': 'arial', 'weight': 'bold', 'size':16}
font_other = {'family': 'arial', 'weight': 'bold', 'size':12}
font_text = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect_acc, color='#0A81AB', width=0.6)
plt.errorbar(x_1, rect_acc, rect_std, color='black', ls='none', capsize=3)

for x, y in zip(x_1, rect_acc):
    plt.text(x, y+0.02, y, font_text, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy within subject', font_title)

plt.xticks(x_1, labels, fontproperties='arial', weight='bold', size=12)
plt.yticks(fontproperties='arial', weight='bold', size=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,0.8)

ax.set_xlim(-1, 13)
plt.plot([-1, 13], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_within_sub.jpg'))
plt.close()

#%% 10 sub decoding result asscioate with cv method
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

main_path = '/nfs/m1/BrainImageNet/Analysis_results/'
out_path = pjoin(main_path, 'imagenet_decoding', 'results')

labels = ['group_run', 'group_sess', 'group_sub']
rect1_acc = [0.279, 0.280, 0.242]
rect1_std = [0.001, 0.002, 0.001]

rect2_acc = [0.449, 0.887, 0.922]
rect2_std = [0.003, 0.012, 0.001]

x_1 = 2.5*np.arange(len(rect1_acc))
x_2 = x_1 + 0.8
font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
font_other = {'family': 'arial', 'weight': 'bold', 'size':10}

plt.figure(figsize=(10, 7))
plt.bar(x_1, rect1_acc, label='single_trial', color='#F54748')
plt.bar(x_2, rect2_acc, label='mean_pattern', color='#0A81AB')

plt.errorbar(x_1, rect1_acc, rect1_std, color='black', ls='none', capsize=3)
plt.errorbar(x_2, rect2_acc, rect2_std, color='black', ls='none', capsize=3)

for x, y, a, b in zip(x_1, rect1_acc, x_2, rect2_acc):
    plt.text(x, y+0.015, y, font_other, ha='center', va='bottom')
    plt.text(a, b+0.015, b, font_other, ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax = plt.gca()
ax.set_ylabel('Accuracy', font_other)
ax.set_title('Accuracy across subject using different cv', font_title)

plt.xticks((x_1 + x_2)/2, labels,
           fontproperties='arial', weight='bold', size=10)
plt.yticks(fontproperties='arial', weight='bold', size=10)
plt.legend(prop=font_other, loc='best')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

ax.set_ylim(0,1.0)

ax.set_xlim(-1.5,7)
plt.plot([-1.5,7], [0.1, 0.1], ls='--', color='gray', lw=1.5)

plt.savefig(pjoin(out_path, 'acc_across_cv.jpg'))
plt.close()

