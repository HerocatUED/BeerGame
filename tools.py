import matplotlib.pyplot as plt

# printed by main.py
actions_group1 = [0]
counts_group1 = [5050]

actions_group2 = [0,1,2,8]
counts_group2 = [1690, 1635, 1625,  100]

fig, ax = plt.subplots()

ax.bar(actions_group1, counts_group1, width=0.4, label='DQN', align='center')
ax.bar([x + 0.4 for x in actions_group2], counts_group2, width=0.4, label='BS', align='center')

ax.set_xlabel('Actions')
ax.set_ylabel('Counts')
ax.set_title('Histogram of Retailer')
ax.legend()

plt.savefig('logs/histogram.png')
