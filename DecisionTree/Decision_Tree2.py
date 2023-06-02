import pandas
from numpy import random
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, roc_auc_score, auc

import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import sys, os
df = pandas.read_csv(sys.argv[1], sep = "\t")
#df = pandas.read_csv("data",sep="\t")

def get_random_as_train( df , t_t_ratio = 0.8 ):
    random_list = []
    random.seed(5)
    while float(len(random_list)/len(df)) <= t_t_ratio:
        random_new = random.randint(0,len(df)-1)
        if random_new not in random_list:
            random_list.append(random_new)
    return random_list

all_list = [i for i in range(len(df))]
train_list = get_random_as_train(df,0.8)
test_list = [i for i in all_list if i not in train_list]

features = [ 'Mycobacterium',  'Complex',  'Complex_RPM',   'Ratio' ]
x_train = df.loc[ train_list, features ]
y_train = df.loc[ train_list, 'result' ] 
df_train = df.loc[train_list]

x_test = df.loc[test_list, features]
y_test = df.loc[test_list, 'result']
df_test = df.loc[test_list]

#X = df[features]
#y = df['result']

dtree = DecisionTreeClassifier(random_state=10)
dtree = dtree.fit(x_train, y_train)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features) #data is to generate picture

graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')
img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)

#plt.show()
#print (dtree.predict([[2935,94,165.8248995,3.202725724]]))
#df['predict'] = dtree.predict([[X]])
#print (df['predict'])
#df['predict'] = [ dtree.predict([[ df['Mycobacterium'][i], df['Complex'][i], df['Complex_RPM'][i], df['Ratio'][i] ]])[0]  for i in range(len(df)) ]

predictions = dtree.predict_proba(x_test)
df_test['prediction'] = predictions[:,1].astype(int)
fpr, tpr, _ = roc_curve(y_test, predictions[:,1])
roc_auc = auc(fpr, tpr)
f, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], ls="--", c=".3")

#plt.clf()
plt.plot(fpr, tpr, linewidth=2, label = 'ROC of Decision Tree', color = 'green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (AUC=' + str(round(roc_auc,2)) + ")")
plt.ylim(0,1.05)
plt.xlim(0,1.05)
#plt.legend(loc=4)
#plt.show()
plt.savefig('decision.roc.png')

#df.to_csv('output.xls', sep='\t')
df_train.to_csv('train.db.xls', sep='\t')
df_test.to_csv('test.db.xls',sep="\t")
#print(df)
