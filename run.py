import time
import random
import math
# if __name__ == '__main__':
#     with open('aa.txt', 'w', encoding='utf-8') as f:
#         f.write('aa')

# print(4/3, math.ceil(4/3))

tic = time.process_time()
print('start_time', tic)
sorted([random.random() for i in range(1000000)])
toc = time.process_time()

time_dif = (toc-tic)

print('Time usage:{} s'.format(time_dif))