#used for plotting graphs
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors

s = pd.read_csv("score_q3.csv",header=None)
g = pd.read_csv("ground_truth.csv",header=None,skiprows=2)

#Fscore for query1
q1 = f1_score(s[11].tolist(),g[11].tolist(),average='macro')
print(q1)

#Fscore for query2
g1 = g.loc[g[11].isin([1,2])]
frames = g1[0].tolist()
s1 = s.loc[s[0].isin(frames)]
print(len(frames),len(s1[0]))
q2 = f1_score(s1[12].tolist()+s1[13].tolist(), g1[12].tolist()+g1[13].tolist(), average='macro')
print(q2)

#Fscore for query3
black = f1_score(s1[1].tolist() + s1[6].tolist(),g1[1].tolist() + g1[6].tolist(),average='macro')
silver = f1_score(s1[2].tolist() + s1[7].tolist(),g1[2].tolist() + g1[7].tolist(),average='macro')
red = f1_score(s1[3].tolist() + s1[8].tolist(),g1[3].tolist() + g1[8].tolist(),average='macro')
white = f1_score(s1[4].tolist() + s1[9].tolist(),g1[4].tolist() + g1[9].tolist(),average='macro')
blue = f1_score(s1[5].tolist() + s1[10].tolist(),g1[5].tolist() + g1[10].tolist(),average='macro')
q3 = (black+silver+red+white+blue)/5
print(q3)

#Total f1score
total = f1_score(s1[1].tolist() + s1[2].tolist() + s1[3].tolist() + s1[4].tolist() + s1[5].tolist() + s1[6].tolist() + s1[7].tolist() + s1[8].tolist() + s1[9].tolist() + s1[10].tolist(),g1[1].tolist() + g1[2].tolist() + g1[3].tolist() + g1[4].tolist() + g1[5].tolist() + g1[6].tolist() + g1[7].tolist() + g1[8].tolist() + g1[9].tolist() + g1[10].tolist(), average='macro')
print(total)

labels = ["Q1","Q2","Q3 Black","Q3 Silver","Q3 Red","Q3 White","Q3 Blue","Q3 Avg","Overall"]
plt.bar(labels,[q1,q2,black,silver,red,white,blue,q3,total],color = ["red","green","blue","blue","blue","blue","blue","blue","black"],width=0.25)
plt.xticks(rotation=30)
plt.xlabel("Queries")
plt.ylabel("F1 Score")
plt.show()
