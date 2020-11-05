#for graph values

import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.3

#averages
q1=[0.135675392,0.136059384,0.137762112]
q2=[0.139048793,0.156846338,0.176873993]
q3=[0.144458626,0.179167418,0.18723335]

r1 = np.arange(len(q1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, q1, color='tab:blue', width=barWidth, edgecolor='white', label='0 Car')
plt.bar(r2, q2, color='tab:pink', width=barWidth, edgecolor='white', label='1 Car')
plt.bar(r3, q3, color='tab:green', width=barWidth, edgecolor='white', label='2 Cars')

plt.ylim(0.08)
plt.xlabel('Cars per Frame', fontweight='bold')
plt.ylabel('Execution Time(secs)', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(q1))], ['Query1', 'Query(1+2)', 'Query(1+2+3)'])

plt.legend()
plt.show()

t1 = 1495/203.0126459
t2 = 1495/215.123863
t3 = 1495/229.5697784
pl = [t1,t2,t3]
plt.ylim(5,8)
plt.bar(['Query1', 'Query(1+2)', 'Query(1+2+3)'], pl, width=barWidth)
#plt.xlabel('Cars per Frame', fontweight='bold')
plt.ylabel('Throughput(fps)', fontweight='bold')
plt.show()
 
