import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

nc=pd.read_csv('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_1/homework-1-albertzhangchi/task3/Nicholas_Cage.csv')
spd=pd.read_csv('/Users/albertzhang/Desktop/18spring/Applied_ML/HW/HW_1/homework-1-albertzhangchi/task3/Swimming_pool_drownings.csv')
years = nc['years'].tolist()
ncData = nc['Nicholas Cage'].tolist()
spdData = spd['Swimming pool drownings'].tolist()
plt.figure(figsize=(11,3))
ax1 = plt.gca()
line1, = ax1.plot(years,spdData,'o-',c='r')
ax2 = ax1.twinx()
line2, = ax2.plot(years,ncData,'o-',c='k')
plt.legend((line1,line2),('Swimming pool drownings','Nicholas Cage'))
ax1.set_ylabel('Swimming pool drownings')
ax1.set_ylim([80,140])
ax2.set_ylabel('Nicholas Cage')
ax2.set_ylim([0,6])
plt.title('Number of people who drowned by falling into a pool correlates with Films Nicolas Cage appeared in' )


plt.savefig('task31.png')
plt.show()





