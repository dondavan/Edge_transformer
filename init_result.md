# setting
BERT based uncased, 12 layer
1 code book, with 128 center

clustered model using int8_t for label, float32_t for center.
original model using float32_t for all weights and computing


# weight
overall model weights, gathered through du:
clustered: 100832 (kb), each layer(excepet embedding):6380 (kb), embedding: 23284(kb)
ori: 400632 (kb), each layer(excepet embedding):25400 (kb), embedding: 93120(kb)

runtime memory access:
bert matrix multiplication(MM) mostly happened in attention liner(Q,K,V 3 MM) and feed forward layer(FF1,FF2 2 MM):
attention liner(Q,K,V) MM right hand matrix size: 768*768
feed forward layer(FF1,FF2), FF1: 3072*768, FF2:768*3072


ori:
Q,K,V: 768*768 = 589824 float32_t memory access
    589824*4 = 2359296 Byte memory access(Q,K,V each)
FF1,FF2: FF1: 3072*768 =  FF2:768*3072 = 2359296 float32_t memory access
    FF1 = FF2 = 3072*768 * 4= 9437184 Byte memory access

clustered + segmented/tile memory access: 
Q,K,V: 768*768 = 589824 int8_t memory access, plus extra 128 float32_t code book mapping
    589824*1 = 589824 Byte memory access(Q,K,V each)
FF1,FF2: FF1: 3072*768 =  FF2:768*3072 = 2359296 int8_t memory access, plus extra 128 float32_t code book mapping
    FF1 = FF2 = 3072*768 * 1= 2359296 Byte memory access


Code book memory is frequently accessed, way better cache performance, right now I don't have runtime memory perf result, so I will treated as no extra memory access.
MM alone clustered version saves ~= 4 times memory access:
ori:            2359296*3 + 9437184*2
                          /                     = 4
cluster:        589824*3  + 2359296*2 

# latency:
CPU,4 thread
input token length: 16

clustered: 0.7427922499999999 (s)
ori:        1.508237(s)