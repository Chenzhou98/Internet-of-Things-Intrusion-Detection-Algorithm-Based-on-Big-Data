#Introduction
flowcontainer is a basic network traffic information extraction library based on python3 to facilitate the analysis of network traffic. Given a pcap file, the library will extract the relevant information of all pcap streams, including the stream source port, source IP address, destination IP address, destination port, length sequence of IP packets, arrival time series of IP data sets, payload sequence and corresponding payload arrival time series, and other extended information. The library filters IP packets, and those packets with tcp/udp payload not 0 are counted in the payload sequence. The tool is simple to use, extensibility and high reusability.

#Original Blog address
[flowcontainer: network traffic characteristic information extraction based on python3 library] (https://blog.csdn.net/jmh1996/article/details/107148871)
url: https://blog.csdn.net/jmh1996/article/details/107148871
