__author__ = 'dk'

import time
import csv
import pandas as pd

time_start = time.time()
from flowcontainer.extractor import extract

result = extract(r"D:\dataset\Attack & Benign Data/18-10-24.pcap", filter='tcp or udp', extension=[''])
counter=0
with open("d:/18-10-24.csv", 'w',newline='') as f:

    for key in result:
        ### The return vlaue result is a dict, the key is a tuple (filename,procotol,stream_id)
        ### and the value is an Flow object, user can access Flow object as flowcontainer.flows.Flow's attributes refer.
        counter=counter+1
        value = result[key]
        #输出的元素
        arraynum=len(value.ip_lengths)
        ipnum=0
        label='Normal'
        label2=0

       # ip_onehot = pd.get_dummies(value.src)
       # print(pd.get_dummies(value.src))


        while arraynum > 0:
           if(1540300287<=value.ip_timestamps[ipnum]<=1540300887):
              label='UdpDevice1W2D'
              label2 =1
           elif (1540300897<=value.ip_timestamps[ipnum]<=1540301498):
               label='UdpDevice10W2D'
               label2 = 1
           elif (1540301508<= value.ip_timestamps[ipnum] <=	1540302108):
               label='UdpDevice100W2D'
               label2 = 1
           elif (1540305374 <= value.ip_timestamps[ipnum] <= 1540305974):
               label='UdpDevice1W2D'
               label2 = 1
           elif (1540305984 <= value.ip_timestamps[ipnum] <= 1540306584):
               label = 'UdpDevice10W2D'
               label2 = 1
           elif (1540306594<= value.ip_timestamps[ipnum] <= 1540307194):
               label = 'UdpDevice100W2D'
               label2 = 1
           elif (1540309095 <= value.ip_timestamps[ipnum] <= 1540309695):
               label = 'TcpSynReflection1W2D2W'
               label2 = 1
           elif (1540310295 <= value.ip_timestamps[ipnum] <= 1540310895):
               label = 'TcpSynReflection10W2D2W'
               label2 = 1
           elif (1540311495 <= value.ip_timestamps[ipnum] <= 1540312095):
               label = 'TcpSynReflection100W2D2W'
               label2 = 1
           elif (1540313470 <= value.ip_timestamps[ipnum] <= 1540314070):
               label = 'Ssdp1W2D2W'
               label2 = 1
           elif (1540314670 <= value.ip_timestamps[ipnum] <= 1540315270):
               label = 'Ssdp10W2D2W'
               label2 = 1
           elif (1540315870 <= value.ip_timestamps[ipnum] <= 1540316470):
               label = 'Ssdp100W2D2W'
               label2 = 1


           element = [#counter, value.src, value.dst,
                      value.sport, value.dport, value.ip_lengths[ipnum],
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
