# selection sort
from collections import defaultdict
import heapq
def Selectionsort(A):

    # cnt = 0
    # for i in range(len(arr)):
    #     cur_min = i
    #     for j in range(i,len(arr)):
    #         if arr[cur_min] > arr[j]:
    #             cur_min = j
    #
    #     if cur_min != i:
    #         arr[i],arr[cur_min] = arr[cur_min],arr[i]
    #         cnt += 1
    #
    # return cnt

    # for duplicates
    heap = list(A)
    heapq.heapify(heap)
    dic = defaultdict(list)
    for idx, el in enumerate(A):
        dic[el].append(idx)

    cnt = 0
    # values denote index and we swap values to update index
    for i in range(len(A)-1):
        cur_idx = dic[A[i]].pop()
        cur_val = A[i]
        min_val = heapq.heappop(heap)
        if cur_val != min_val:
            min_idx = dic[min_val].pop()
            dic[cur_val],dic[min_val] = min_idx,cur_idx
            dic[cur_idx].append(min_idx)
            cnt+=1

    return cnt


test = [5,2,4,6,1,3]
print(Selectionsort(test))