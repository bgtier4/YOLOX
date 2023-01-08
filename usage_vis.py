# for each line starting with %CPU:
# loop through the lines until next %CPU line (there may be 3 MAX and the first is always 0%?)
# add each line to an array, add a time counter to an extra array
# graph each array against the time array
import sys
from matplotlib import pyplot as plt
import numpy as np
import csv

cpu_log = open('logs/cpu.txt', 'r')
lines = cpu_log.readlines()

time_log = open('logs/time.txt')
times = time_log.readlines()

i = 0
p1 = []
p2 = []
t = 0
t_arr = []
# print('len(lines) = ', len(lines))
while i < len(lines):
    t_arr.append(int(times[t]))
    t += 1
    if lines[i] == "%CPU\n":
        i += 2 # go past %CPU line and 0.0 line
    else:
        print("SOMETHING FAILED!")
        sys.exit()
    
    if i >= len(lines):
        break
    
    p1.append(float(lines[i]))
    i += 1
    if i >= len(lines) or lines[i] == "%CPU\n":
        p2.append(0)
        continue
    p2.append(float(lines[i]))
    i += 1

gpu_util = []
with open('logs/gpu.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')

    for row in plots:
        if row[0][0] == "#":
            continue
        csv_row = [x for x in row[0].split()]
        gpu_util.append(int(csv_row[4]))

print('len(p1) = ', len(p1))
print('len(p2) = ', len(p2))
print('len(t_arr) = ', len(t_arr))

# print('p1 = ', p1)
# print('p2 = ', p2)
# print('t_arr = ', t_arr)
plt.subplot(2,1,1)
plt.plot(t_arr, p1, label='p1')
plt.plot(t_arr, p2, label='p2')
plt.plot(t_arr, np.array(p1)+np.array(p2), label='sum of processes')
plt.xlabel('t')
plt.ylabel('Util %')
plt.title('CPU utilization')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t_arr, gpu_util)
plt.xlabel('t')
plt.ylabel('Util %')
plt.title('GPU utilization')
plt.legend()
# plt.xticks(rotation=90, fontsize='x-small')
# ax = plt.gca()
# for label in ax.get_xaxis().get_ticklabels()[::2]:
#     label.set_visible(False)
plt.tight_layout()

plt.show()
plt.savefig('graphs/usage_out.png')