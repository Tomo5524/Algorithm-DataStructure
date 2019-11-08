# Stack and Queue CTCI round 2

# 3.1 Three in One: Describe how you could use a single array to implement three stacks.

class threeStacks:

    """algorithm
        1, number of stacks is fixed, create a single list and size tracker 
        2, push and pop according to top of each sublist
        3,


"""
    
    def __init__(self,capacity):
        self.num_of_stack = 3 # fixed
        self.arr = [None for i in range(self.num_of_stack * capacity)]
        self.len_tracker = [0] * self.num_of_stack # represent size of each list
        self.capacity = capacity

    def Push(self,data,stacknum):
        if self.IsFull(stacknum):
            raise Exception('Stack Overflow　みゃおーーー ')
        self.len_tracker[stacknum] += 1
        self.arr[self.top(stacknum)] = data
        
        
    def Pop(self,stacknum):
        if self.IsEmpty(stacknum):
            raise Exception('Stack Underflow')
        val = self.arr[self.top(stacknum)]
        self.len_tracker[stacknum] -= 1
        return val

    def IsEmpty(self,stacknum):
        return self.len_tracker[stacknum] == 0

    def IsFull(self,stacknum):
        return self.len_tracker[stacknum] == self.capacity

    def Peek(self,stacknum):
        return self.arr[self.top(stacknum)]

    def top(self,stacknum): # top of each sublist
        sublist = self.capacity * stacknum # find sublist (stacknum)
        return sublist + self.len_tracker[stacknum] - 1 # -1 cuz of adjustment of index
        
                  

##s = threeStacks(2)
##print(s.IsEmpty(1))
##s.Push(3, 1)
##print(s.Peek(1))
##print(s.IsEmpty(1))
##s.Push(2, 1)
##print(s.Peek(1))
##print(s.Pop(1))
##print(s.Peek(1))
##s.Push(3, 1)
##s.Push(0,2)
##s.Push(1,2)
##print(s.Peek(1))
##print(s.Pop(2))
##print(s.Pop(2))
##s.Push(5,0)
##print(s.Pop(0))
##




# CTCI 3.2 Stack Min

"""algorithm
    1, create stack and anoter list that keeps the track of lowest value
    2, the lowest tracker's last element is always the lowest which is O(1) by tracker[-1]
    3, keep updadaitng the tracker by when pop value in stack, if it is the last value in tracker
    4, if so, pop minitracker


"""

class Stack:
    def __init__(self):
        self.stack = []
        self.mini_tracker = []

    def push(self,data):
        if len(self.mini_tracker) == 0:
            self.mini_tracker.append(data)

        if self.mini_tracker[-1] > data:
            self.mini_tracker.append(data)

        self.stack.append(data)

    def pop(self):
        if self.isEmpty():
            print('no value')
            return
        val = self.stack.pop()
        if val == self.mini_tracker[-1]: # update the tracker
                                # otherwise, lowest value will be same even after it get popped
            self.mini_tracker.pop()
        return val

    def mini(self):
        if self.isminiEmpty(): # edge case
            print('no value')
            return
        
        return self.mini_tracker[-1]

    def isminiEmpty(self):
        return len(self.mini_tracker) == 0

    def isEmpty(self):
        return len(self.stack) == 0

# ["MinStack","push","push","push","getMin","pop","getMin"]
# [[],[0],[1],[0],[],[],[]]

s = Stack()
s.push(0)
s.push(1)
s.push(0)
print(s.mini())
print(s.pop())
print(s.pop())
print(s.mini())

#print(s.mini())
##s = Stack()
##s.push(5)
##s.push(2)
##print(s.pop())
##s.push(3)
##s.push(4)
##print(s.mini())
##print(s.pop())
##print(s.pop())
##print(s.mini())
##print(s.pop())
##print(s.mini())
##print(s.pop())
##s.push(1000)
##print(s.mini())


# CTCI 3.3 Stack of Plates
"""algorithm
        1, using a single list, implement stack of plates (nested list)
        2, when currect stack exceeds capacity, new sublist is created
        3, push val to stack [current] 
        4, and pop val in stack[current] if there is no val, move back to previous sublist
        5, if current sublist is empty (current !=0) delete the subllist


"""


# one problem, empty list created after value is popped which is a waste
class StackPlates:
    def __init__(self,capacity):
        self.list = [[]]
        self.current = 0
        self.capacity = capacity

    def push(self,data): # there is no edge case as number of stacks is not fixed
        
        if len(self.list[self.current]) == self.capacity:
            #if not self.list[self.current+1]:
            # prevent from recreating new list
                #self.list.append([])
            self.list.append([])
            self.current += 1

        self.list[self.current].append(data)

    def pop(self):
        # edge case
        if self.isEmpty():
            print('empty you dummy')
            return
        val = self.list[self.current].pop() # value is popped so it decrements automatically
        
        if self.isEmpty() and self.current != 0:
            # delete current sublist
            # when there is no value, if current decrements, it becomes -1
            # so currnet != 0 is edge case
            self.clean_emptylist() # before decrement cuz current shuld be the one to delete
            self.current -= 1
            
            
        return val

    def clean_emptylist(self):
        if self.isEmpty():
            self.list.pop()

    def isEmpty(self):
        return len(self.list[self.current]) == 0


##s = StackPlates(2)
##s.push(1)
##s.push(2)
##print(s.pop())
##s.push(3)
##s.push(4)
##print(s.pop())
##print(s.pop())
##print(s.pop())
##print(s.pop())
##s.push(5)
##s.push(6)
##s.push(7)
##print(s.pop())
##print(s.pop())



# CTCI 3.4 Queue via Stacks

# create queue using two stacks

"""algorithm
    1, s1 is used as stack (LIFO)
    2, pop top value into s2 so popped valus goes to the bottom of stack
    3, pop the top value of s2

"""

class stack:

    def __init__(self):
        self.s1 = []
        self.s2 = []


    def push(self,data):
        self.s1.append(data)

    def pop(self):
        if self.s2: # if there is value in s2 that is already queued
            return self.s2.pop()

        if self.s1:
            while self.s1:
                self.s2.append(self.s1.pop())
                
            return self.s2.pop()

        return None # edge case


    # another way to implementate this 
    
##    def __init__(self):
##        self.s1 = []
##        self.s2 = []
##
##    def push(self,data):
##        while len(self.s1) != 0:
##            self.s2.append(self.s1.pop())
##            #self.s2.pop()
##
##        self.s1.append(data)
##
##        while len(self.s2)!=0:
##            self.s1.append(self.s2.pop())
##            #self.s1.pop()
##
##    def pop(self):
##        # edge case
##        if len(self.s1) == 0:
##            print('not popping enough')
##            return
##        
##        return self.s1.pop()

##s = stack()
##s.push(1)
##s.push(2)
##print(s.pop()) 
##s.push(3)
##s.push(4)
##print(s.pop()) 
##print(s.pop())
##print(s.pop())
##print(s.pop())        


# CTCI 3.5 Sort Stack
    

def sort_stack(s):
    temp_stack = Stack()
 
    while not s.is_empty():
        # while s: s still exists even len is 0 and peek causes error cuz none type has no value
        temp_value = s.pop()
 
        if not temp_stack.is_empty() and temp_stack.peek() > temp_value:
            while not temp_stack.is_empty() and temp_stack.peek() > temp_value:
                s.push(temp_stack.pop())
         
        temp_stack.push(temp_value)
 
    while not temp_stack.is_empty():
        s.push(temp_stack.pop())
 
    return s
 
class Stack:
     
    def __init__(self):
        self.stack = []

    # this method coverts instance to string so can be pritned
    def __str__(self): # pass all the metohds around
        return self.stack.__str__()
 
    def push(self, value):
        self.stack.append(value)
 
    def pop(self):
         
        return self.stack.pop()
 
    def peek(self):
        if not self.stack:
            return None
 
        return self.stack[-1]
 
    def is_empty(self):
        return len(self.stack) == 0

val = Stack()
val.push(5)
val.push(3)
val.push(1)
val.push(4)
print(sort_stack(val))



# CTCI 3.6 Animal Shelter

"""algorithm
    1, create 2 queues (cats and dogs) and timestamp
    2, deque will be based on this timestamp
    3, pay attention to edge cases

"""

from collections import deque 

class AnimalShelter:
    def __init__(self):
        self.cats = deque([])
        self.dogs = deque([])
        self.timestamp = 1 # starts off with 1 
        
    def push(self,animal):
        # no edge case #
        if animal == "Cat":
            self.cats.append((animal,self.timestamp))
            self.timestamp+=1
        elif animal == "Dog":
            self.dogs.append((animal,self.timestamp))
            self.timestamp+=1
        else:
            print('bro do you know English?')

    def dequeueAny(self):
        # edge case # 
        if self.isCatsEmpty() and self.isDogsEmpty():
            return None
        if self.isCatsEmpty():
            return self.dogs.popleft()
        if self.isDogsEmpty():
            return self.cats.popleft()

        # when self.peek called it returns tuple ('Cat',1)
        # so this deque is based on timestamp
        # [1] is absolutely critical here
        if self.peekCat()[1] < self.peekDog()[1]:
            return self.cats.popleft()
        else:
            return self.dogs.popleft()
        
    def dequeueCat(self):
        # edge case
        if self.isCatsEmpty():
            return None
        return self.cats.popleft()
    
      
    def peekCat(self):
        # edge case
        if self.isCatsEmpty():
            return None
        return self.cats[0]
        # just [0] cuz cats is tuple
        # cats = ((cat,0)(cat,1))
    
    def dequeueDog(self):
        if self.isDogsEmpty():
            return None
        return self.dogs.popleft()


    def peekDog(self):
        if self.isDogsEmpty():
            return None
        return self.dogs[0]

    def isCatsEmpty(self):
        return len(self.cats) == 0
        

    def isDogsEmpty(self):
        return len(self.dogs) == 0
   
    def p(self):
        print(self.dogs)
        print(self.cats)

a = AnimalShelter()
a.push("Dog")
a.push("Cat")
a.push("Dog")
a.push("Cat")
a.push("Cat")
a.push("meow")
print(a.dequeueCat())
print(a.dequeueAny())
#print(a.peekCat())
#print(a.peekDog())
print(a.dequeueDog())

print(a.dequeueAny())
print(a.peekCat())
a.p()



    # my work using linkedlist # 
##class Node:
##    def __init__(self,data):
##       
##        self.data = data
##        self.next = None
##        self.pre = None
##        
##class AnimalShelter:
##    def __init__(self):
##        #self.shelter = []
##        self.head = None
##        self.dog_front = self.cat_front = None
##        self.dog_rear = self.cat_rear = None
##        self.front = self.rear = None 
##
##    def enqueue(self,data):
##        temp = Node(data)
##        if self.rear == None:
##            self.front = temp
##        else:
##            self.rear.next = temp
##            
##        self.rear = temp
##        
##    def dequeueAny(self):
##        if self.isEmpty():
##            print('empty')
##            return
##        if self.front:
##            temp = self.front
##            next_node = self.front.next
##            self.front = next_node
##            return temp.data
##
##
##    def dequeueCat(self):
##        if self.isEmpty():
##            print('empty')
##            return
##        cur = self.front
##        #next_first = self.fornt
##        while cur:
##            if cur.data == "cat":
##                temp = cur
##                cur.next = cur.next.next
##                #self.front = next_node
##                return temp.data
##
##            cur = cur.next
##        
##   # how to deal with a situation where the first node is the target one 
##    def dequeueDog(self):
##        if self.isEmpty():
##            print('empty')
##            return
##        cur = self.front
##        #next_first = self.fornt
##        while cur:
##            if cur.data == "dog":
##                temp = cur
##                cur.next = cur.next.next
##                #self.front = next_node
##                return temp.data
##
##            cur = cur.next
##
##    def isEmpty(self):
##        return self.front == None
##
##a = AnimalShelter()
##a.enqueue("cat")
##a.enqueue("cat")
##a.enqueue("dog")
##a.enqueue("dog")
##a.enqueue("dog")
##print(a.dequeueCat())
##print(a.dequeueCat())
##print(a.dequeueDog())
##print(a.dequeueAny())
##print(a.dequeueAny())
##print(a.dequeueAny())

