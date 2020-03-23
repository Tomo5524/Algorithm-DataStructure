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

# https://leetcode.com/discuss/interview-question/346621/Google-or-Phone-Screen-or-Min-swaps-to-sort-array
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
            dic[cur_val].append(min_idx)
            dic[min_val].append(cur_idx)
            # swap values in original arrays
            A[cur_idx],A[min_idx] = min_val,cur_val
            cnt+=1

    return cnt


test = [5,2,4,6,1,3]
print(Selectionsort(test))


def merge_sort(arr):
    # time complexity O(nlogn)
    if len(arr) > 1:  # there is just one element, recursion stops
        mid = len(arr) // 2  # Finding the mid of the array
        L = arr[:mid]  # Dividing the array elements
        R = arr[mid:]  # into 2 halves

        merge_sort(L)  # Sorting the first half
        merge_sort(R)  # Sorting the second half

        i = j = k = 0

        # swap values based on arr
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

    return arr


test = [5,8,2,6,3,7,1,9]
test1 = [10, 15, 4, 20, 1]
test2 = [23, 92, 67, 54, 4, 42, 16, 36, 93, 57]
test3 = [8,5,6,2,7,3]

print("merge")
print(merge_sort(test))
print(merge_sort(test1))
print(merge_sort(test2))
print(merge_sort(test3))
print()


def Bubble_Sort(nums):

    # time complexity O(n2)
    for i in range(len(nums)):
        swap = False
        for j in range(i+1,len(nums)):
            if nums[i] > nums[j]:
                 nums[i],nums[j] = nums[j],nums[i]
                 swap = True

        if not swap:
            break

    return nums

test = [5,8,2,6,3,7,1,9]
test1 = [10, 15, 4, 20, 1]
test2 = [23, 92, 67, 54, 4, 42, 16, 36, 93, 57]
test3 = [5,2,4,6,1,3]

print(Bubble_Sort(test))
print(Bubble_Sort(test1))
print(Bubble_Sort(test2))
print(Bubble_Sort(test3))



























