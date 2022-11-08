__author__ = 'dk'

import time
import csv
import pandas as pd

time_start = time.time()
from flowcontainer.extractor import extract

result = extract(r"D:\dataset\benign data/18-10-19.pcap", filter='tcp or udp', extension=[''])
counter=0


with open("d:/18-10-19.csv", 'w',newline='') as f:

    for key in result:
        ### The return vlaue result is a dict, the key is a tuple (filename,procotol,stream_id)
        ### and the value is an Flow object, user can access Flow object as flowcontainer.flows.Flow's attributes refer.
        counter=counter+1
        value = result[key]
        #输出的元素
        arraynum=len(value.ip_lengths)

        #label='Normal'
        label2=0
        ipnum = 0
        element = [counter, value.src, value.dst,value.sport, value.dport, value.ip_lengths[ipnum],
                      value.ip_timestamps[ipnum],value.time_start,value.time_end,label2]  #lable1 ,value.payload_lengths[ipnum//2],value.payload_timestamps[ipnum//2]]
        ipnum = ipnum + 1
        arraynum = arraynum - 1
        writer = csv.writer(f)
        writer.writerow(element)



      #  writer = csv.writer(f)
      # writer.writerow(element)
        print("success !!!! 1")




        print('Flow {0} info:'.format(key))
        ## access ip src
        print('src ip:', value.src)
        ## access ip dst
        print('dst ip:', value.dst)
        ## access srcport
        print('sport:', value.sport)
        ## access_dstport
        print('dport:', value.dport)
        ## access payload packet lengths
        print('payload lengths :', value.payload_lengths)
        ## access payload packet timestamps sequence:
        print('payload timestamps:', value.payload_timestamps)
        ## access ip packet lengths, (including packets with zero payload, and ip header)
        print('ip packets lengths:', value.ip_lengths)
        ## access ip packet timestamp sequence, (including packets with zero payload)
        print('ip packets timestamps:', value.ip_timestamps)

        ## access default lengths sequence, the default length sequences is the payload lengths sequences
        print('default length sequence:', value.lengths)
        ## access default timestamp sequence, the default timestamp sequence is the payload timestamp sequences
        print('default timestamp sequence:', value.timestamps)

        ##access sni of the flow if any else empty str
        print('extension:', value.extension)

print(len(result))

time_end = time.time()
print('time cost', time_end - time_start, 's')
