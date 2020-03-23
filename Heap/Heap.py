from heapq import heappush, heappop, heapify
import heapq
"""
import heapq

heappop O(log n) - pop and return the smallest element from heap
heappush O(log n)- push the value item onto the heap, maintaining
            heap invarient
heapify O(h) = O(n) - transform list into heap, in place, in linear time
heapsort is O(nlogn) 
heapq.nlargest 
heapq.nsmallest, O(log(n) * m), heapq.nlargest(n, it), where it is an iterable with m elements
"""

# https://leetcode.com/problems/kth-largest-element-in-an-array/discuss/206454/3-Python-solutions-with-Space-and-Time-complexities

#
# li = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
# li1 = [1, 9, 8, 2, 3, 10, 14, 7, 16, 4]
# heapq.heapify(li)
# heapq.heapify(li1)
# print ("The created heap is : ", li)
# print ("The created heap is : ", li1)
# print (list(li))

""""""


def build_min_heap(arr):
    # iterates the nodes except the leaf nodes
    # We don’t need to apply min_heapify to the items of indices after n/2+1,
    # which are all the leaf nodes.
    for i in reversed(range(len(arr) // 2)):
        # since it is reversed, it is going up from bottom which is badass
        minheapify(arr, i)

    return arr

def minheapify(arr, root):
    left = (2 * root) + 1
    right = (2 * root) + 2

    smallest = root

    # compair root and left child and if it is smaller
    # update root
    if left < len(arr) and arr[smallest] > arr[left]:
        smallest = left

    # length compairson should come first
    # otherwise arr[right] is 10 and len(arr) is up to 9
    # if arr[smallest] > arr[right] and right < len(arr): # cause error
    # cuz it does not check if it is out of bounrdy or not
    if right < len(arr) and arr[smallest] > arr[right]:
        smallest = right

    # if swap needs here
    if smallest != root:
        arr[root], arr[smallest] = arr[smallest], arr[root]
        # without this, in this case [1,9,8,2,3,10,14,7,16,4]
        # 9 and 7 never swap
        # this makes sure current subtree satisfies the heap property
        minheapify(arr, smallest)

    return arr

test = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
test1 = [1, 9, 8, 2, 3, 10, 14, 7, 16, 4]
print('min heap')
print("before min heap: ", test)
print("before min heap: ", test1)
print()
print("after min heap: ", build_min_heap(test))
print("after min heap: ", build_min_heap(test1))

def heapsort(iterable):
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]

print()
print("heapsort")
print(heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0]))



# initializing list
li = [5, 7, 9, 1, 3]
# li = [[1, 2], [1, 4], [1, 6], [7, 2], [7, 4], [7, 6], [11, 2], [11, 4], [11, 6]]
# li = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
# li = [5, 7, 9, 4, 3]

# using heapify to convert list into heap
print('heapify')
heapq.heapify(li)
print(li)
print()
# print(heapq.heapify(li)) # doesnt work for some reason


# initializing list
li1 = [6, 7, 9, 4, 3, 5, 8, 10, 1]

# using heapify() to convert list into heap
heapq.heapify(li1)
print(li1)
# using nlargest to print 3 largest numbers
# prints 10, 9 and 8
print("The 3 largest numbers in list are : ", end="")
print(heapq.nlargest(3, li1))

# using nsmallest to print 3 smallest numbers
# prints 1, 3 and 4
print("The 3 smallest numbers in list are : ", end="")
print(heapq.nsmallest(3, li1))
print(heapq.nsmallest(3, li1)[-1])