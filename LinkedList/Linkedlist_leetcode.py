from Linkedlist_node_library import Node,convert_to_list,reverse
from collections import deque

# 234. Palindrome Linked List
class LinkedList:
    def __init__(self):
        self.head = None

    def isPalindrome(self):
        """
        key is to get middle value
        algorithm
        1, create fast and slow pointers to get middle point
        2, reverse from middle to the end
        3, get current that points to the front and compare second half and current
        """
        fast = slow = self.head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # this slow reverse from middle to the end
        second_half = reverse(slow)
        cur = self.head
        while second_half:
            if cur.val != second_half.val:
                return False

            cur = cur.next
            second_half = second_half.next

        return True

llist = LinkedList()
llist.head = Node(1)
llist.head.next = Node(2)
llist.head.next.next = Node(2)
llist.head.next.next.next = Node(1)
print(llist.isPalindrome())

# 109. Convert Sorted List to Binary Search Tree
        
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.left = None
        self.right = None

class LinkedList:
    def __init__(self):
        self.head = None

    def sortedListToBST(self):

        # Form an array out of the given linked list and then
        # use the array to form the BST.
        values = convert_to_list(self.head)

        # l and r represent the start and end of the given array
        def convertListToBST(l, r):

            if l > r:
                return 
            mid = (l+r) // 2
            root = TreeNode(values[mid])
            root.left = convertListToBST(l, mid-1)
            root.right = convertListToBST(mid+1, r)

            return root


        root = convertListToBST(0,len(values)-1)
        return levelOrder(root)

def levelOrder(root):

    q = deque([root])
    res = []
    while q:
        cur = q.popleft()
        res.append(cur.val)
        if cur.left:
            q.append(cur.left)
        if cur.right:
            q.append(cur.right)

    return res
    
        
##def sortedListToBST(self):
##    """ algorithm
##        1, create 2 pointers, slow and fast
##        2, find middle, and create bst tree
##        3, 
##
##"""
##    
##    if not head: return None
##    
##    slow = fast = head
##    while fast and fast.next:
##        fast = fast.next.next
##        slow = slow.next
##
##    root = Node

# [-10,-3,0,5,9]
llist = LinkedList()
llist.head = Node(-10)
llist.head.next = Node(-3)
llist.head.next.next = Node(0)
llist.head.next.next.next = Node(5)
llist.head.next.next.next.next = Node(9)
print(llist.sortedListToBST())

# 2. Add Two Numbers
class LinkedList:
    def __init__(self):
        self.head = None

    def AddTwoNumbers(self,l2):
        """ algorithm
            1, create dummy, and create cur that points to dummy
            2, use carry as sum. carry gets cur1 and cur2 values
                if it comes out greater than 10, get carry over value
            3, cur gets cur.next

"""
        dummy = cur = Node(0)
        cur1 = self.head
        cur2 = l2.head
        carry = 0

        # carry needed in while statement
        # edge case for when input finishes with carry left. [5],[5] = [0,1]
        while cur1 or cur2 or carry: # carry will still grab leftovers even cur1 and cur2 dont exist
            if cur1:
                carry += cur1.val
                cur1 = cur1.next

            if cur2:
                carry += cur2.val
                cur2 = cur2.next
            
            # here carry is cut down to 1
            cur.next = Node(carry%10) # gets the last digit
            # 7 % 10 = 7, 17 % 10 = 7
            # 242 % 10 = 2, 242 % 100 = 42, 202 % 100 = 2

            cur = cur.next
            carry = carry//10 # if grater than 10, grab 1st digit (1), if less, grab 0
            # 2 // 10 = 0, 10 // 10 = 1
            
            # temp.next = cur1 or cur2
        return convert_to_list(reverse(dummy.next))

llist = LinkedList()
llist1 = LinkedList()
llist.head = Node(7)
llist.head.next = Node(1)
llist.head.next.next = Node(6)
llist1.head = Node(5)
llist1.head.next = Node(9)
llist1.head.next.next = Node(2)

llist2 = LinkedList()
llist3 = LinkedList()
llist2.head = Node(2)
llist2.head.next = Node(4)
llist2.head.next.next = Node(3)
llist3.head = Node(5)
llist3.head.next = Node(6)
llist3.head.next.next = Node(4)

llist4 = LinkedList()
llist5 = LinkedList()
llist4.head = Node(9)
llist4.head.next = Node(8)
llist5.head = Node(1) # [9,0]

llist6 = LinkedList()
llist7 = LinkedList()
llist6.head = Node(5)
llist7.head = Node(5) # [0,1]

llist8 = LinkedList()
llist9 = LinkedList()
llist8.head = Node(9)
llist8.head.next = Node(9)
llist9.head = Node(1) # [1,0,0]


# print(llist.AddTwoNumbers(llist1))
# print(llist2.AddTwoNumbers(llist3))
# print(llist4.AddTwoNumbers(llist5))
# print(llist6.AddTwoNumbers(llist7))
# print(llist8.AddTwoNumbers(llist9))


# 82. Remove Duplicates from Sorted List II
class LinkedList:
    def __init__(self):
        self.head = None

    def delete_duplicates(self):
        """ algorithm
            1, create dummy,
            2, create pointer (pre) that keeps track of previous val of cur
            3, cur checks duplicates and connect previous to non duplicates
            4, return dummny.next to exclude dummy

"""
        dummy = Node(0)
        dummy.next = self.head
        pre = dummy
        cur = self.head
        while cur and cur.next:
            # when cur.next doesnt exist, loop will terminate, (after 5)
            # so no error 
            # cur alone is for edge case where input is []
            if cur.val == cur.next.val:
                while cur.next and cur.val == cur.next.val:
                    # if there is no cur.next, and input ends with duplicates, # [1,1]
                    # cur.next.val will cause error as cur.next is None
                    cur = cur.next
                    
                # head is at the last val of current duplicates
                pre.next = cur.next
                
            else:
                # slower
                # pre.next is alive so this one easier
                # if cur.val and cur.next.val are not same
                # prev = prev.next
                # to connect pointer when cur.val and cur.next val are not the same
                # this line should be in else
                pre = pre.next
            cur = cur.next
                 
        # ignore first val (0)
        # so should be dummy.next
        return convert_to_list(dummy.next)

# edge case for when input ends with duplicates

llist = LinkedList()
llist.head = Node(1)
llist.head.next = Node(1)
llist.head.next.next = Node(3)
llist.head.next.next.next = Node(3)
llist.head.next.next.next.next = Node(3)
llist.head.next.next.next.next.next = Node(4)
llist.head.next.next.next.next.next.next = Node(5)
llist.head.next.next.next.next.next.next.next = Node(5)
#llist.head.next.next.next.next.next.next.next = Node(2)

# edge case for when itputs only has duplicates

llist1 = LinkedList()
llist1.head = Node(1)
llist1.head.next = Node(1) # []

llist2 = LinkedList()
llist2.head = Node([]) # [[]]

llist3 = LinkedList()
llist3.head = Node(1)
llist3.head.next = Node(1)
llist3.head.next.next = Node(1)
llist3.head.next.next.next = Node(2)
llist3.head.next.next.next.next = Node(3) # [2,3]

print(llist.delete_duplicates())
print(llist1.delete_duplicates())
print(llist2.delete_duplicates())
print(llist3.delete_duplicates())

# 142. Linked List Cycle II

class LinkedList:
    def __init__(self):
        self.head = None

    def detectCycle(self):

        """ algorithm
            1, create 2 pointers, fast and slow
            1, when fast and slow collide
            2, bring slow back to the beginning and move slow and fast by one
            3, when they collide, that is the intersection

"""
        # returns value not index

        # edge case for when there is no value #
        if not self.head: return None
        slow = fast = self.head
        while fast and fast.next:
            # if only fast, fast.next.next causes error when there is only value
            # if only fast.next, when there is no value, it will cause error
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                slow = self.head
                while slow:
                    if slow == fast:
                        return slow.val

                    slow = slow.next
                    fast = fast.next
                    # [1,2], 0, think about this case
                    # slow and fast move toghether 
                    # and collide at 1 not 0
        return None
        
llist = LinkedList()
llist.head = Node("grinding")
llist.head.next = Node("negativity")
llist.head.next.next = Node("good ass vibes")
llist.head.next.next.next = Node("depression")
llist.head.next.next.next.next = Node("grinding")
llist.head.next.next.next.next.next = Node("dream")
llist.head.next.next.next.next.next.next = llist.head
#llist.head.next.next.next.next.next.next = Node("hussle")
print(llist.detectCycle())

llist1 = LinkedList()
llist1.head = Node(3)
llist1.head.next = Node(2)
llist1.head.next.next = Node(0)
llist1.head.next.next.next = Node(-4)
llist1.head.next.next.next.next = llist1.head.next
print(llist1.detectCycle())  # 2

llist2 = LinkedList()
llist2.head = Node(1)
llist2.head.next = Node(2)
llist2.head.next.next = llist2.head
print(llist2.detectCycle()) # 1

# test case 3
llist3 = LinkedList()
llist3.head = Node(1)
print(llist3.detectCycle()) # none

llist4 = LinkedList()
llist4.head = Node([])
print(llist4.detectCycle())

# 86. Partition List

class LinkedList:
    def __init__(self):
        self.head = None
    
    """ algorithm
            1, create 4 pointes to track less and greater values
            2, create current pointer and next pointer
            3, current pointer.next should be none so it can connect to a less greater value
            4, after that connect less end and greater beginning
            
        """
    # 1->4->3->2->5->2
    # if every iteration, pointer doesn't get killed, 5 still connects to 2
    # and cause infine loop

    def partition(self,t):
            if not self.head: return None
            
            less_front = less_end = None
            greater_euqal_front = greater_euqal_end = None
            cur = self.head
            while cur:
                next_node = cur.next 
                cur.next = None
                if cur.val < t:
                    if not less_front:
                        less_front = cur
                    else:
                        less_end.next = cur
                        
                    less_end = cur
                    
                else:
                    if not greater_euqal_front:
                        greater_euqal_front = cur
                    else:
                        greater_euqal_end.next = cur
                        
                    greater_euqal_end = cur
                    
                cur = next_node

##            less_end.next = greater_euqal_front
##            return convert_to_list(less_front) 'NoneType' object has no attribute 'next'

            ### edge case for single node ###
            # input = [1], t = 0, shoud return [1]
            # but lines above causes error as less_end is none (not exist)
            # so check if it exists and if so, connect it to greater line
            # if not, just return greater line
            
            if less_end:
                less_end.next = greater_euqal_front
                return convert_to_list(less_front)
                
            else:
                return convert_to_list(greater_euqal_front)

llist = LinkedList()
llist.head = Node(28)
llist.head.next = Node(15)
llist.head.next.next = Node(6)
llist.head.next.next.next = Node(9)
llist.head.next.next.next.next = Node(1)
llist.head.next.next.next.next.next = Node(2)
llist.head.next.next.next.next.next.next = Node(27)
llist.head.next.next.next.next.next.next.next = Node(2)

# 1->4->3->2->5->2
# if every iteration, pointer doesn't get killed, 5 still connects to 2
# and cause infine loop
llist1 = LinkedList()
llist1.head = Node(1)
llist1.head.next = Node(4)
llist1.head.next.next = Node(3)
llist1.head.next.next.next = Node(2)
llist1.head.next.next.next.next = Node(5)
llist1.head.next.next.next.next.next = Node(2)

llist2 = LinkedList()
llist2.head = Node(1)

print(llist.partition(6))
print(llist1.partition(3))
print(llist2.partition(0))
    

# 876. Middle of the Linked List

class Node:
    def __init__(self,val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        
    def middleNode(self):
            if not self.head: return None
            slow = fast = self.head
            while fast and fast.next:
                # fast must come first as after 5,
                # as when it points to none, fast.next causes error
                
                # if only fast.next, [1,2,3,4,5,6], in this case, after five, 
                # fast points to none and none has no next which causes this error
                # if only fast, after 5, fast.next.next dose not exist and causes error
                
                fast = fast.next.next  
                slow = slow.next
                
            return slow.val

test = LinkedList()
test.head = Node(1)
test.head.next = Node(1)
test.head.next.next = Node(2)
print(test.middleNode())

# 83. Remove Duplicates from Sorted List

class LinkedList:
    def __init__(self):
        self.head = None

    def deleteDuplicates(self):
        
        if not self.head: return None
        
        cur = self.head
        while cur:
            while cur.next and cur.next.val == cur.val:
                # cur doesn't move, and cur always points to the origival val
                # where loop starts,
                # e,g, 1->1->1->4->3->, when comparing all 1s,
                # cur.val stays at the beginning and compares each value
                # by moving just 1 (cur.next = cur.next.next) which validates each val
                
                
                # when it reaches the end of value,
                # w/o cur.next, cur.next.val will register and cause an error
                # as there is no cur.next
                # which is why cur.next is needed
                # it is needed when comparing cur.val and cur.next .val
                
                cur.next = cur.next.next     # skip duplicated node
            cur = cur.next     # not duplicate of current node, move to next node
            
        return convert_to_list(self.head)

# test case
test = LinkedList()
test.head = Node(1)
test.head.next = Node(1)
test.head.next.next = Node(2) #[1,2]

test1 = LinkedList()
test1.head = Node(1)
test1.head.next = Node(1)
test1.head.next.next = Node(1) # [1]

test2 = LinkedList()
test2.head = Node(1)
test2.head.next = Node(1)
test2.head.next.next = Node(1)
test2.head.next.next.next = Node(3)
test2.head.next.next.next.next = Node(3)
test2.head.next.next.next.next.next = Node(3)
test2.head.next.next.next.next.next.next = Node(4)
test2.head.next.next.next.next.next.next.next = Node(5)

test3 = LinkedList()
test3.head = Node(1)
test3.head.next = Node(1)
test3.head.next.next = Node(2)
test3.head.next.next.next = Node(3)
test3.head.next.next.next.next = Node(3)

print(test.deleteDuplicates())
print(test1.deleteDuplicates())
print(test2.deleteDuplicates())
print(test3.deleteDuplicates())


class LinkedList:
    def __init__(self):
        self.head = None

# merge sorted lists

    def merge_lists(self,head1, head2):
        """algorithm
            1, create temp node
            2, and connect it to smaller value
            3, if list1 is smaller, move list1 value to the right
            4, other move list2
            

"""
         
        cur_s1 = self.head
        cur_s2 = head2.head
        temp = Node(float('-inf'))
        #temp = Node(0)
        temp_f = temp
        while cur_s1 and cur_s2:
            if cur_s1.val <= cur_s2.val:
                temp.next = cur_s1
                temp = cur_s1
                cur_s1 = cur_s1.next
            else:
                temp.next =  cur_s2
                temp = cur_s2
                cur_s2 = cur_s2.next

        temp.next = cur_s1 or cur_s2

        res = []
        head = temp_f.next
        while head:
            res.append(head.val)
            head = head.next

        return res

        
geek = LinkedList()
geek.head =Node(1)
geek.head.next =Node(5)
geek.head.next.next =Node(9)

geek1 = LinkedList()
geek1.head =Node(2)
geek1.head.next =Node(4)
geek1.head.next.next =Node(6)

print(geek.merge_lists(geek,geek1))


# 328 Odd Even Linked List

class LinkedList:
    def __init__(self):
        self.head = None

    def oddEvenList(self):
            
            cur = self.head
            odd = even = odd_e = even_e = None
            cnt = 1
            while cur:
                next_node = cur.next
                cur.next = None
                if cnt % 2 != 0:
                    if odd == None:
                        odd = cur
                    else:
                        odd_e.next = cur 
                    odd_e = cur
                        
                else:
                    if even == None:
                        even = cur
                    else:
                        even_e.next = cur  
                    even_e = cur
                    
                cur = next_node
                cnt+=1   
            
            odd_e.next = even

            res =[]
            while odd:
                res.append(odd.val)
                odd = odd.next
            return res

llist = LinkedList()
llist.head = Node(28)
llist.head.next = Node(15)
llist.head.next.next = Node(6)
llist.head.next.next.next = Node(9)
llist.head.next.next.next.next = Node(1)
llist.head.next.next.next.next.next = Node(2)
llist.head.next.next.next.next.next.next = Node(27)
llist.head.next.next.next.next.next.next.next = Node(2)

print(llist.oddEvenList())


