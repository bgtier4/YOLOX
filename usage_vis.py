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
    if t < len(times):
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
temp = []
mem_usage = []
with open('logs/gpu.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')

    for row in plots:
        if row[0][0] == "#":
            continue
        csv_row = [x for x in row[0].split()]
        gpu_util.append(int(csv_row[4]))
        temp.append(int(csv_row[2]))
        mem_usage.append(int(csv_row[5]))

print('len(p1) = ', len(p1))
print('len(p2) = ', len(p2))
print('len(t_arr) = ', len(t_arr))

# NOW GET MEMORY USAGE
mem_usage = []
TOTAL_MEMORY = 6144 # MiB, constant value

gpu_mem_log = open('logs/gpu_mem.txt', 'r')
lines = gpu_mem_log.readlines()

for line in lines:
    mem_usage.append(int(line)/6144)

# print('p1 = ', p1)
# print('p2 = ', p2)
# print('t_arr = ', t_arr)
# FIRST GRAPH CPU AND GPU USAGE
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

# NOW GRAPH GPU TEMPERATURE AND MEMORY USAGE
plt.clf()
plt.subplot(2,1,1)
plt.plot(t_arr, temp)
plt.xlabel('t')
plt.ylabel('Temperature')
plt.title('GPU temperature')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t_arr, mem_usage)
plt.xlabel('t')
plt.ylabel('Usage %')
plt.title('GPU memory usage')
plt.legend()
# plt.xticks(rotation=90, fontsize='x-small')
# ax = plt.gca()
# for label in ax.get_xaxis().get_ticklabels()[::2]:
#     label.set_visible(False)
plt.tight_layout()

plt.show()
plt.savefig('graphs/extra_out.png')