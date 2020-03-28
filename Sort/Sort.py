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
print()


# insert

def insertion_sort(arr):
    # time complexity O(nlogn)
    # always compare current index to the one index before current one
    # start from index 1 to compare current index and one idx before
    for i in range(1,len(arr)):
        # if current idx smaller than one index before, it is not sorted.
        if arr[i] < arr[i-1]:
            j = i
            while j and arr[j-1] > arr[j]:
                arr[j-1], arr[j] = arr[j], arr[j-1]
                j -= 1

    return arr

test = [17, 41, 5, 22, 54, 6, 29, 3, 13]
test1 = [8,5,6,2,7,3]
test2 = [1,2,3,4,5]
test3 = [5,4,3,2,1]
test4 = [5,8,2,6,3,7,1,9]
test5 = [10, 15, 4, 20, 1]
test6 = [23, 92, 67, 54, 4, 42, 16, 36, 93, 57]
test7 = [64, 34, 25, 12, 22, 11, 90,2]
test8 = [1]
test9 = [1,2]
test10 = [2,1]

print('insertion_sort')
print(insertion_sort(test))
print(insertion_sort(test1))
print(insertion_sort(test2))
print(insertion_sort(test3))
print(insertion_sort(test4))
print(insertion_sort(test5))
print(insertion_sort(test6))
print(insertion_sort(test7))
print(insertion_sort(test8))
print(insertion_sort(test9))
print(insertion_sort(test10))
print()

def partition(arr,left,right):
    # time complexity O(nlogn)
    # no auxiliary space

    l = left
    pivot = arr[right]  # pivot is rightmost

    # go over from left to right
    for r in range(left, right):

        ### if pivot is greater than current value
        if arr[r] <= pivot:  # if value smaller than pivot found,
            # swap that value and the value greater than pivot at rightmost (i)

            arr[r], arr[l] = arr[l], arr[r]
            l += 1
    arr[l], arr[right] = arr[right], arr[l]
    return l

def quick_sort(arr, low, high):
    if low < high:  # if there are values to be sorted
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)  # if rigthmost value is the greatest, pi return len(arr)
        quick_sort(arr, pi + 1, high)

    return arr

test = [17,41,5,22,54,6,29,3,13]
test1 = [8,5,6,2,7,3]
test2 = [1,2,3,4,5]
test3 = [5,4,3,2,1]
test4 = [5,8,2,6,3,7,1,9]
test5 = [10, 15, 4, 20, 1]
test6 = [23, 92, 67, 54, 4, 42, 16, 36, 93, 57]

print("quick_sort")
print(quick_sort(test,0,len(test)-1))
print(quick_sort(test1,0,len(test1)-1))
print(quick_sort(test2,0,len(test2)-1))
print(quick_sort(test3,0,len(test3)-1))
print(quick_sort(test4,0,len(test4)-1))
print(quick_sort(test5,0,len(test5)-1))
print(quick_sort(test6,0,len(test6)-1))
print()

def merge_sort(arr):
    # # time complexity O(nlogn)

    # divide array until it can't
    if len(arr) > 1:
        mid = len(arr) // 2
        L,R = arr[:mid],arr[mid:]
        merge_sort(L)
        merge_sort(R)
        # index for each array
        l = r = k = 0
        while l < len(L) and r < len(R):
            # if current element in R array is bigger than current element in L
            # array[k] gets whichever smaller
            if L[l] < R[r]:
                arr[k] = L[l]
                l += 1
            else:
                arr[k] = R[r]
                r += 1

            k += 1

        # check if there is any element left
        while l < len(L):
            arr[k] = L[l]
            l += 1
            k += 1

        while r < len(R):
            arr[k] = R[r]
            r += 1
            k += 1

        return arr

test = [5,8,2,6,3,7,1,9]
test1 = [10, 15, 4, 20, 1]
test2 = [23, 92, 67, 54, 4, 42, 16, 36, 93, 57]
test3 = [8,5,6,2,7,3]
test4 = [1,2,4,5,6,7,8,3]

print("merge")
# print(merge_sort(test))
# print(merge_sort(test1))
# print(merge_sort(test2))
print(merge_sort(test3))
print(merge_sort(test4))
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



























