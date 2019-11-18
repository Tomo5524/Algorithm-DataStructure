#Stack LeetCode

# 11/14/2019

# 739. Daily Temperatures
def dailyTemperatures(T):

    # optimized solution
    """
    algorithm
    1, loop in descending order because we are trying to find the next occurrence of a warmer temperature
        so we just need to remember previous numbers
    2, create stack and put index into stack
    3, if we see smaller value than the value on the top of stack,
        we want to find the distance between warmer value(top stack) and current temperature
    """
    stack = []
    res = [0] * len(T)
    for i in range(len(T)-1,-1,-1):
        # if stack exists and top stack is smaller than current temperature, pop the previous
        # so current warmest tempearture will be on top of stack
        # duplicates mean it is not warmer so should update stack when running into duplicates
        while stack and T[stack[-1]] <= T[i]:
            stack.pop()

        # if stack exists, meaning, we still have the warmest temperature so sar
        # so update result
        if stack:
            res[i] = stack[-1] - i

        #  We'll keep a stack of indices
        stack.append(i)

    return res

    # brute force
    """
    check each value and find out shortest distance between current value and any value greater than that
    """
    res = []
    for i in range(len(T)):
        cur_lowest = float("inf")
        for j in range(i,len(T)):
            if T[i] < T[j]:
                cur_lowest = min(cur_lowest,j-i)

        if cur_lowest != float('inf'):
            res.append(cur_lowest)

        # if cur_val is the largest, get 0
        else:
            res.append(0)

    return res


print(dailyTemperatures([89,62,70,58,47,47,46,76,100,70])) # [8, 1, 5, 4, 3, 2, 1, 1, 0, 0]
print(dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73])) # [1, 1, 4, 2, 1, 1, 0, 0]
print(dailyTemperatures([1,1,2,2,3,3])) # [2, 1, 2, 1, 0, 0]
print(dailyTemperatures([5,4,2,1,3,6])) # [5, 4, 2, 1, 1, 0]
print(dailyTemperatures([9,5,4,2,1,3,6])) # [0, 5, 4, 2, 1, 1, 0]
print()



# 10/27/2019

# 239. Sliding Window Maximum
# # return each max element from each subarray with length of k
from collections import deque
def maxSlidingWindow(nums,k):

    """
    key is to how to store value and how to pop it
    algorithm
    1, create queue, when running into greater value than last element in queue,
        pop element in queue
    2, front element (index) in queue should be biggest all the time
        so when encountering a larger element, pop the previous biggest
    3, last element in queue is the smallest so when we see greater value than that, pop it
    3, we manipulate index, so store index in queue
    """
    # time complexity O(n)
    if not nums: return None
    res = []
    q = deque()
    # first get the first k values
    for i in range(k):
        # if we see greater value than the previous value, pop previous value
        ### should be while for edge case, [4,3,11]
        while q and nums[q[-1]] < nums[i]:
            q.pop()

        q.append(i)

    for j in range(k,len(nums)):
        # front value is always the biggest value so get it
        res.append(nums[q[0]])
        ### if front value is out of k boundary, pop it
        if j - k >= q[0]:
            q.popleft()

        ### nums[q[-1]] denotes smallest value so if current value is greater, pop it
        while q and nums[q[-1]] < nums[j]:
            q.pop()

        q.append(j)

    res.append(nums[q[0]])
    return res


meow = [1,3,1,2,0,5] # [3, 3, 2, 5]
k = 3
meow1 = [1,3,-1,-3,5,3,6,7] # [3, 3, 5, 5, 6, 7]
k1 = 3
meow2 = [4,3,11] # [11]
k2 = 3
meow3 = []
k3 = 0
print(maxSlidingWindow(meow,k))
print(maxSlidingWindow(meow1,k1))
print(maxSlidingWindow(meow2,k2))
print(maxSlidingWindow(meow3,k3))


# 844. Backspace String Compare
"""Edge case is when ## are in a row,

"""

def backspaceCompare(S, T):
    """
    algorithm
    1, append anything other than "#" to stack
    2, if current value is "#", pop last element from stack
    3, compare length of both strings
    :param S:
    :param T:
    :return:
    """
    if not S or not T: return None
    s1 = []
    s2 = []
    for i in range(len(S)):
        if s1 and S[i] == "#":

            s1.pop()

        else:
            if S[i] != "#": # w/o this, when running inot #s in a row, # will go into stack and will cause error
                            # edge case for "y#fo##f , "y#f#o##f"
                s1.append(S[i])

    for i in range(len(T)):
        if s2 and T[i] == "#":

            s2.pop()

        else:
            if T[i] != "#":
                s2.append(T[i])

    return s1 == s2

meow = "y#fo##f"
meow1 = "y#f#o##f"
meow2 = "ab#c"
meow3 = "ad#c"
meow4 = "ab##"
meow5= "c#d#"
meow6 = "a##c"
meow7 = "#a#c"
meow8 = "a#c"
meow9 = "b"
print(backspaceCompare(meow,meow1))
print(backspaceCompare(meow2,meow3))
print(backspaceCompare(meow4,meow5))
print(backspaceCompare(meow6,meow7))
print(backspaceCompare(meow8,meow9))

# 232. Implement Queue using Stacks

class myQueue:
    """ 1, newly arrived element goes into s1
        2, when it pops or peeks, move it all to s2
        in which the first value should be returned
        cuz queue is last in first out

    """

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []

    def push(self, x):
        """
        Push element x to the back of queue.
        """
        self.s1.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.peek():
            return self.s2.pop()

    def peek(self):
        """
        Get the front element of s2
        if both stacks are empty, return None
        if s2 is not empty, the last value in it is queue's first value
        """
        if not self.empty():
            if not self.s2: # if s2 exists, it will just return the top of it, in test case2, 2 is already in queue so return it
                while self.s1:
                    self.s2.append(self.s1.pop())

            return self.s2[-1]

        return None

    def empty(self):
        """
        Returns whether the queue is empty.
        """
        return self.s1 == [] and self.s2 == []


# ["MyQueue","push","empty"]
# [[],[1],[]]

MyQueue = myQueue()
MyQueue.push(1)
print(MyQueue.empty())

MyQueue = myQueue() # test case 2
MyQueue.push(1)
MyQueue.push(2)
print(MyQueue.pop())
MyQueue.push(3)
MyQueue.push(4)
print(MyQueue.pop())
print(MyQueue.pop())
print(MyQueue.pop())
print(MyQueue.pop())

MyQueue = myQueue()
MyQueue.push(1)
MyQueue.push(2)
print(MyQueue.peek())
print(MyQueue.pop())
print(MyQueue.empty())

# 1172. Dinner Plate Stacks
import heapq

class StackPlates:
    def __init__(self, capacity):
        self.c = capacity
        self.q = []
        self.stacks = []

    def push(self, val):
        # when stack reaches capacity, heap spits out index of current stack
        while self.q and self.q[0] < len(self.stacks) and len(self.stacks[self.q[0]]) == self.c:
            heapq.heappop(self.q)
        # update index of current stack by appending len of stack
        if not self.q:
            stack_num = len(self.stacks)
            heapq.heappush(self.q, stack_num)
        # creates new subrrray as previous stack reaches capacity
        if self.q[0] == len(self.stacks):
            self.stacks.append([])
        self.stacks[self.q[0]].append(val)

    def pop(self):
        # delete last empty list
        while self.stacks and not self.stacks[-1]: # when there is no stack, it will cause error, after third pop,
                                    # there is no value, and will become self.stack = [],
                                    # after poppping that, error happens
            self.stacks.pop()
        return self.popAtStack(len(self.stacks) - 1)

    def popAtStack(self, index):
        # when there is no stack, index -1 will be passed in and 0 < index will be false
        # to check that, len(stack) should be greater than that
        if 0 <= index and self.stacks[index]:
            # update heappush so the first value in heap represents the index of non-full stack
            heapq.heappush(self.q, index)
            return self.stacks[index].pop()
        return -1

# ["DinnerPlates","push","push","push","push","push","popAtStack","push","push","popAtStack","popAtStack","pop","pop","pop","pop","pop"]
# [[2],[1],[2],[3],[4],[5],[0],[20],[21],[0],[2],[],[],[],[],[]]
s = StackPlates(2)
s.push(1)
s.push(2)
s.push(3)
s.push(4)
s.push(5)
print(s.popAtStack(0))
s.push(20)
s.push(21)
print(s.popAtStack(0))
print(s.popAtStack(0))
print(s.popAtStack(0))
print(s.popAtStack(2))
print(s.pop())
print(s.pop())
print(s.pop())
print(s.pop())
print(s.pop())

#["DinnerPlates","push","push","push","popAtStack","pop","pop"]
# [[1],[1],[2],[3],[1],[],[]]
# [null,null,null,null,2,3,1]
# when len of stack is 2, should check if len(stack) >= capacity, cuz len is greater, not just equal
s = StackPlates(1)
s.push(1)
s.push(2)
s.push(3)
print(s.popAtStack(1))
print(s.pop())
print(s.pop())

# one problem, empty list created after value is popped which is a waste
# class StackPlates:
#     def __init__(self, capacity):
#         self.list = [[]]
#         self.current = 0
#         self.capacity = capacity
#
#     def push(self, data):  # there is no edge case as number of stacks is not fixed
#         flag = True
#         for i in range(len(self.list)):
#             if len(self.list[i]) != self.capacity:
#                 self.list[i].append(data)
#                 flag = False
#                 break
#
#         if flag:
#             self.list.append([data])
#
#     def pop(self):
#         # edge case
#         if self.isEmpty():
#             print('empty you dummy')
#             return
#         val = self.list[self.current].pop()  # value is popped so it decrements automatically
#
#         if self.isEmpty() and self.current != 0:
#             # delete current sublist
#             # when there is no value, if current decrements, it becomes -1
#             # so currnet != 0 is edge case
#             self.clean_emptylist()  # before decrement cuz current shuld be the one to delete
#             self.current -= 1
#
#         return val
#
#     def clean_emptylist(self):
#         if self.isEmpty():
#             self.list.pop()
#
#     def isEmpty(self):
#         return len(self.list[self.current]) == 0
#
#     def popAtStack(self, index):
#         if not self.list:
#             return None
#         return self.list[index].pop()


##["DinnerPlates","push","push","push","push","push","popAtStack","push","push","popAtStack","popAtStack","pop","pop","pop","pop","pop"]

# # [null,null,null,null,null,null,2,null,null,1,20,21,5,4,3,-1]
# s = StackPlates(2)
# s.push(1)
# s.push(2)
# s.push(3)
# s.push(4)
# s.push(5)
# print(s.popAtStack(0))
# s.push(20)
# s.push(21)
# print(s.popAtStack(0))
# print(s.popAtStack(2))
# print(s.pop())
# print(s.pop())
# print(s.pop())
# print(s.pop())
# print(s.pop())

# 682. Baseball Game
def calPoints(points):
    """ algorithm
        1, create stack, calculate based on each operation
        2, if no operation happens, cur val is integer and just append it to stack
        3, get sum of stack and return it

    # edge case, negative number # doesnt matter if it is positive or negrative just append
    :param points:
    :return: sum of stack
    """
    stack = []
    for p in points:

        if p == "C":
            stack.pop()

        elif p == "D":
            stack.append(stack[-1]*2)

        elif p == "+":
            stack.append(stack[-1]+stack[-2])

        # if p == "D": if it is just if, this else goes through and causes error
        # when current value is not integer, error base 10 """
        else: # if this eles takes place, the current value is promised to be integer
                # and doesnt matter if it is negative or not so just append it
            stack.append(int(p))

    return sum(stack)

meow =  ["5","2","C","D","+"]
meow1 = ["5","-2","4","C","D","9","+","+"]
print(calPoints(meow))
print(calPoints(meow1))

# 1021. Remove Outermost Parentheses
def removeOuterParentheses(s):
    """ algorithm
        1, knowing input cab be either empty (""), "(" + A + ")",
            or A + B, where A and B are valid parentheses strings,
        2, create counter, increment each time and if it is, dont do anything
            as cnt 0 means outermost parenthesis
        3, check closed parenthesis first and decrement it,
            because, if open parenthesis is checked first, it just returns the same thing as original input
            as cnt is never 0
        4, edge case for when "()()", no parenthesis inside so output is ""
    """
    cnt = 0
    res = ''
    for char in s:

        if char == ")":
            cnt -=1
        if cnt != 0:
            res += char
        if char == "(":
            cnt +=1

    return res

meow = "(()())(())"
meow1 = "(()())(())(()(()))"
meow2 = "()()" # no parenthesises are inside of each parenthesis so remove outermost parenthesis
print(removeOuterParentheses(meow))
print(removeOuterParentheses(meow1))
print(removeOuterParentheses(meow2))




# 155. Min Stack
class MinStack:
    """ algorithm
        1, create stack under constructor
        2, this stack holds 2 values in each subarray
        3, in each sub list, first value is value that is passed in and second value is min value
        4, keep the track of minvalue by min(self.stack[-1][1],val)

"""

    def __init__(self):
        self.stack = []

    def push(self, x):

        if not self.stack:  # get the first value, otherwise causes error when comparing using stack[-1]
            self.stack.append((x, x))
        else:
            self.stack.append((x,min(self.getMin(),x)))

    def pop(self):
        if self.isEmpty():
            return "Underflow"
        popped = self.stack.pop()
        return popped[0]

    def top(self):
        if self.isEmpty():
            return "Underflow"

        return self.stack[-1][0]

    def getMin(self):
        if self.isEmpty():
            return "Underflow"
        return self.stack[-1][1]

    def isEmpty(self):
        return self.stack == []

s = MinStack()
s.push(0)
s.push(1)
s.push(0)
print(s.getMin())
print(s.top())
print(s.pop())
print(s.top())
print(s.pop())
print(s.getMin())

a = MinStack()
print('test case 2')
a.push(5)
a.push(2)
print(a.top()) # 2
print(a.pop()) # 2
a.push(3)
a.push(4)
print(a.getMin()) # 3
print(a.pop()) # 4
print(a.pop()) # 4
print(a.top()) # 5
print(a.getMin()) # 5
print(a.pop()) # 5
print(a.getMin())
print(a.pop())
a.push(1000)
print(a.getMin())

# daily coding problem 19

# return each max element from each subarray with length of k
from collections import deque
def maxKsubarray(arr,k):

    # brute force time complexity O(2)
##    for i in range(k):
##        max_ele = 0
##        for j in range(i,i+k):
##            max_ele = max(max_ele,arr[j])
##        print(max_ele)
##
##    print(max_ele)        

    """ algorithm
        1, create queue, when running into greater value than last element in queue, 
            pop element in queue
        2, front element (index) in queue should be biggest
            so when encountering a larger element, pop the previous biggest
        3, 

"""
    q = deque()
    res = []
    for i in range(k):
        while q and arr[i] > arr[q[-1]]:
            # this while loop kees the biggest in front
            q.pop()
        q.append(i)

    # take care of elements after index k
    for i in range(k,len(arr)):
        res.append(arr[q[0]])
        while q and q[0] <= i - k:
            # this while loops pops out the element that is out of k range
            q.popleft() # front value is out of range of k, e,g 10

        while q and arr[q[-1]] < arr[i]:
            #this while loops pops out the element that is less than current element (arr[i])
            q.pop()

        q.append(i)

    res.append(arr[q[0]])
    return res

meow = [10,5,2,7,8,7] # [10, 7, 8, 8]
meow1 = [5,10,2,7,8,7] # [10, 10, 8, 8]
meow2 = [1,2,3,4,5,6] # [3, 4, 5, 6]
meow3 = [6,5,4,3,2,1] # [6, 5, 4, 3]
print(maxKsubarray(meow,3))
print(maxKsubarray(meow1,3))
print(maxKsubarray(meow2,3))
print(maxKsubarray(meow3,3))

# Leetcode 20. Valid Parentheses

def ValidParentheses(p):
    """ algorithm
        1, create dictionary, key is open braces and values are closed braces
        2, create stack and append ope braces to it 
        3, if current value is a pair of last value in stack, pop it from stack

"""
    dic = {"(":")", "{":"}", "[":"]"}
    stack = []
    for i in p:
        
        if i not in dic.values():
            stack.append(i)
        
        else:
            # find match
            if stack and dic[stack[-1]] == i:
                stack.pop()
            else:
                return False

    return stack == []

##    # my work # 
##    stack = []
##    # it will always be closed with one of these items
##    # all i need to validate is close as open is already in stack
##    close = [")","}","]"]
##    for i in s:
##        if stack and i in close: # needs stack to get first value otherwise it will cause error "){"
##            # if right match, stack[-1] should be open
##            valid = stack[-1] + i
##            if valid == "()" or valid == "{}" or valid == "[]":
##                stack.pop()
##            # "(])" in this case, 
##            # the line below will append ']' and becomes false
##            # otherwise becomes true
##            else:
##                stack.append(i)
##        else:
##            stack.append(i)
##            
##    return False if stack else True
    
meow = ""
meow1 = "(){}[]"
meow2 = "(]"
meow3 = "({})"
meow4 = "(])"
meow5 ="){"
meow6 ="]"

print(ValidParentheses(meow))
print(ValidParentheses(meow1))
print(ValidParentheses(meow2))
print(ValidParentheses(meow3))
print(ValidParentheses(meow4))
print(ValidParentheses(meow5))
print(ValidParentheses(meow6))
