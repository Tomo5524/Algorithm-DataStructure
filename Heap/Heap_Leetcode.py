# Leetcode
from collections import Counter
import heapq

# 621. Task Scheduler


["A","A","A","B","B","B"] n = 2
["A","B","C","A","B","C","D","E","F","G"] n  = 2
["A","A","A"] n = 2
["A","A","A","B"] n = 2
["A","A","A","B","B","C","C"] n = 2
["A","B","C","A","B","C"] n  =2 # no idle time




# 703. Kth Largest Element in a Stream

class KthLargest:
    """
    Create a pq - keep it only having the k-largest elements by popping off small elements.
    With only k elements, the smallest item (self.heap[0]) will always be the kth largest.

    If a new value is bigger than the smallest, it should be added into your heap.
    If it's bigger than the smallest (that are already the kth largest), i
    it will certainly be within the kth largest of the stream.
    """

    # time complexity O(n)

    def __init__(self, nums, k):
        # algorithm, remove small items (after heapifying,) till length of nums == k
        # then first element is always kth largest
        self.subnum = nums
        self.k = k
        heapq.heapify(self.subnum)  # if not hepify this, small element would be 4 and will result in 2,5,8
        # make it same length as k so the first element is always the third largest
        while len(self.subnum) > k:
            heapq.heappop(self.subnum)
            # front value will always be the largest

    def add(self, val):
        # when nothing added, else statement will pop smallest and length is short by 1 so keep it maintained
        # when length is off, need to maintain heap so push  val
        if len(self.subnum) < self.k:
            heapq.heappush(self.subnum, val)
            # heappushpop(heap, ele) :- This function combines the functioning of both push and
            # pop operations in one statement,
            # increasing efficiency. Heap order is maintained after this operation.

        else:
            # when input array is empty, and if statement above(line29)
            # dosent exist, it will cause error
            # as this line has nothing to pop
            heapq.heappushpop(self.subnum, val)
            # heapq.heapreplace(self.subnum,val)

        return self.subnum[0]


meow = [4, 5, 8, 2]
k = 3
kthLargest = KthLargest(meow, k)
print('kthLargest')
print(kthLargest.add(3))
print(kthLargest.add(5))
print(kthLargest.add(10))
print(kthLargest.add(9))
print(kthLargest.add(4))

# ["KthLargest","add","add","add","add","add"]
# [[1,[]],[-3],[-2],[-4],[0],[4]]

meow1 = []
k1 = 1
kthLargest = KthLargest(meow1, k1)
print('kthLargest')
print(kthLargest.add(-3))
print(kthLargest.add(-2))
print(kthLargest.add(-4))
print(kthLargest.add(0))
print(kthLargest.add(4))

# 347. Top K Frequent Elements

def TopKFrequentElements(nums,k):
    """algorithms
        1, create hash and find out the frequency of each element by its value
        2, using keyword only argument, get key
        3,
    """

    c = Counter(nums) # {-1: 2, 2: 2, 4: 1, 1: 1, 3: 1}
    # val is key, counter is value
    #print(c)
    # this keyword only argument will get largets values as keys are paired with values
    # key=c.get will grab its values
    # c.get points to values as c is all keys which are attacthed to values
    # ,c.keys() will get all values as that is how to get values (always by key)
    return heapq.nlargest(k, c, key=c.get)  # get keys based on values

    # wont work for edge case, [-1,-1], it will return 2 instead of 1
    # return heapq.nlargest(k, c.values(), key=c.get)  # get keys based on values

meow = [4, 1, -1, 2, -1, 2, 3]  # k =2
k_test = 2
meow1 = [1]
k1 = 1
meow2 = [1, 2]
k2 = 2
meow3 = [3, 0, 1, 0]
k3 = 1  # doesnt work for heapq.nlargest(k,c.values(),key=c.get)
meow4 = [1, 1, 1, 2, 2, 3]
k4 = 2
nums = [-1,-1]
k = 1

print(TopKFrequentElements(meow, k_test))
print(TopKFrequentElements(meow1, k1))
print(TopKFrequentElements(meow2, k2))
print(TopKFrequentElements(meow3, k3))
print(TopKFrequentElements(meow4, k4))
print(TopKFrequentElements(nums,k))
#arr = sorted(dic, key=dic.get, reverse=True)

# 692. Top K Frequent Words

def topKFrequent(arr,k):
    """Given a non-empty list of words, return the k most frequent elements.
    If two words have the same frequency, then the word with the lower alphabetical order comes first.
    algorithm
    1, creat heap and and push count and word to heap
    2, heap will maintain smallest count at the beginning and word as second element
        if count of two words are same (i and love), i will push it to the beginning
    3, genius!!! by making counter minus, heappop maintain heap invariance so smaller element will always be at the beginning
    """
    # time complexity O(nlogn)
    # brute force

    # arr.sort()  # sort so it is now in alphabetical order
    # word = Counter(arr)
    # return heapq.nlargest(k, word, key=word.get)


    # time complexity O(nlogk)
    # how to sort words with O(nlogk)
    # you dont sort words, make freq negative so it will be at the beginning of heap
    count = Counter(arr)
    #heap = [(-freq, word) for word, freq in count.items()]
    heap = []
    for word,freq in count.items():
        # since frequent is -, top most frequent words are always places at the beginning
        ### Genius!!!
        heapq.heappush(heap,(-freq,word))

    #heapq.heapify(heap)
    res = []
    for i in range(k):
        # heappop maintain heap invarient
        # so coding was the last value, but becomes the beginning since it is smaller
        res.append(heapq.heappop(heap)[1])

    return res

meow = ["i", "love", "leetcode", "i", "love", "coding"]
k = 2
meow1 = ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"]
k1 = 4
meow2 =["i", "love", "leetcode", "i", "love", "coding"]
k2 = 3

print()
print(topKFrequent(meow,k))
print(topKFrequent(meow1,k1))
print(topKFrequent(meow2,k2))

# 373. Find K Pairs with Smallest Sums

def kSmallestPairs(n1,n2,k):
        """
        algorithm
        1, since it is SORTED, start off with first value n1[0] and n2[0]
        2, and visit adjacent nodes and heappush it to heap
            heap always holds the smallest one at the beginning
        1, create heap mark n1[i] and n2[i] visited
        2, push value, and visited locations to heap
        3, pop heap and push n1[i+1],n2[i] and n[i],n2[i+1]
        4, keep going until len(res) == k
        5, when working with whiteboard, write a graph 2d array and track visited locations

        """
        # how to get value 1,1 and 2,2
        visited = set()
        res = []
        heap = []
        visited.add((0, 0))  # first values are already visited
        heapq.heappush(heap, (n1[0] + n2[0], 0, 0))  # get sum of first values from both lists

        # why heap needed
        # n1 = [1,1,2], n2 = [1,2,3], k = 10
        # it will be out of boundary as there is no heap but still tyring to pop
        # while len(res) < k or heap: or will keep going until there is nothig in heap
        # so and it correct
        while len(res) < k and heap:
            val,n1_idx,n2_idx = heapq.heappop(heap)  # heappop always spits out the smallest value
            res.append([n1[n1_idx], n2[n2_idx]])  # since the given input is sorted
            # first values are the smallest

            # without visited, it will visit the same locations
            # e,g, [1, 1, 2], [1, 2, 3], k = 10
            # it will keep visiting the same location
            # and wont visit every location
            if n1_idx + 1 < len(n1) and (n1_idx + 1, n2_idx) not in visited:
                # push value, and visited locations to heap
                heapq.heappush(heap, (n1[n1_idx + 1] + n2[n2_idx], n1_idx + 1, n2_idx,))
                visited.add((n1_idx + 1, n2_idx))

            if n2_idx + 1 < len(n2) and (n1_idx, n2_idx +1) not in visited:
                # push value, and visited locations to heap
                heapq.heappush(heap, (n1[n1_idx] + n2[n2_idx + 1], n1_idx, n2_idx + 1,))
                visited.add((n1_idx, n2_idx + 1))

        return res

n1 = [1, 7, 11]
n2 = [2, 4, 6]
k = 3
meow1 = [1, 1, 2]
meow2 = [1, 2, 3]
k1 = 2
meow3 = [1, 1, 2]
meow4 = [1, 2, 3]
k2 = 10 # [[1,1],[1,1],[2,1],[1,2],[1,2],[2,2],[1,3],[1,3],[2,3]]
nums1 = [1,2]
nums2 = [3]
k3 = 3
print(kSmallestPairs(n1, n2, k))
print(kSmallestPairs(meow1, meow2, k1))
print(kSmallestPairs(meow3, meow4, k2))
print(kSmallestPairs(nums1, nums2, k3))
