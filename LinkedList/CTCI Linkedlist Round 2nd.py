# CTCI Linkedlist round 2nd 6/14/2019

class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
        self.prev = None

class LinkedList:
    def __init__(self):
        self.head = None
        

    def display(self):
        res = []
        cur = self.head
        while cur:
            res.append(cur.data)
            cur = cur.next
        print(res)

    def length(self,node):
        cnt = 0
        cur = node.head
        while cur:
            cur = cur.next
            cnt +=1
        return cnt

    def doublypush(self,data):
        new_node = Node(data)
        new_node.next = self.head
        new_node.prev = None
        if self.head: # if there is existing head, connect its prev to new_node
            self.head.prev = new_node
        self.head = new_node # new_node becomes head

    def reverseHelper(self,node):
        # no head so create pointer instead of head
        pre_node = None
        next_node = None
        while node:
            next_node = node.next
            node.next = pre_node
            pre_node = node
            node = next_node # node becomes None as next_node is node at the end 
            
        self.head = pre_node # pre_node points to the last value (the front cue it is reversed)
        return node

        
        

    def reverse_recursion_helper(self,node):
        # when node gets last val, and there is no head.next, node will end up with the last val
        # you need head here cuz when it gets to the last val, there is no next
        # so havig only head.next causes error
        if not head or not head.next:
            return head
        node = self.reverseList(head.next)
        # head is the val before the last val (head[-2])
        # head.next.next = head, points to itself
        head.next.next = head
        # cut off link
        head.next = None
        # node always poits to the last val as it will be the head in reversed LL
        return node
        


       
    # CTCI 2.1 
    def removeDuplicates(self):
        """algorithm
            1, traverse to create dic and count each value
            2, traverse again to eliminate duplicates 
            3, if a values is stored in dic more than twice, get rid of it


"""

        # remove duplicates and output will not have no duplicates at all
        dummy = Node(0)
        prev = dummy
        cur = self.head
        # when head.next doesnt exist, loop will terminate, (after 5)
        # so no error 
        # cur alone is for when input is []
        while cur and cur.next:
            if cur.next and cur.val == cur.next.val:
                while cur.val == cur.next.val:
                    cur = cur.next
                    
                # cur is at the last val of current duplicates
                prev.next = cur.next
            else:
                # pre.next is alive so this one easier
                # if cur.val and cur.next.val are not same
                # prev = prev.next
                # to connect pointer when cur.val and cur.next val are not the same
                # this line should be in else
                prev = prev.next

            # move cur cuz it is still at the last val of duplicates
            cur = cur.next
            
            
        
        
        
        # assumign LL is sorted
        cur = self.head
        while cur.next:
            if cur.data == cur.next.data:
                cur.next = cur.next.next
                # cur doest not move so it points to the same node
                # unitl it runs into a new value
            else:
                cur = cur.next

##        dic = {}
##        cur = self.head
##        while cur:
##            if cur.data not in dic:
##                dic[cur.data] = 1
##            else:
##                dic[cur.data] +=1
##            cur = cur.next
##
##        cur = self.head
##        while cur:
##            if dic[cur.data] > 1:
##                cur.next = cur.next.next
##                dic[cur.data] = 1 # if deleted value does not get updated, it will keep deleting and no vavlue that has duplicates appears
##            else:
##                cur = cur.next

##        cur = self.head
##        if cur == None:
##            return
##        while cur.next != None:
##            if cur.data == cur.next.data:
##                new = cur.next.next
##                cur.next = None
##                cur.next = new
##            else:
##                cur = cur.next


    # CTCI 2.2
    def returnKthToLast(self,k):
        """ algorithm
            1, move p1 kth times
            2, after steps 1, move both of them unitl pi hits the end 

"""
        p1 = p2 = self.head
        cnt = 0 
        while p1 and cnt < k:
            p1 = p1.next
            cnt +=1

        while p1:
            p1 = p1.next
            p2 = p2.next

        return p2.data


    # CTCI 2.3
    def deleteMiddle(self,node):
        length = self.length(node)
        mid = length // 2
        print(mid)
        cnt = 0
        cur = node.head
        pre = node.head
        while cur:
            if cnt == mid:
                pre.next = cur.next.next
                cur = None
            else:
                pre = cur
                cur = cur.next
            cnt += 1

    # CTCI 2.4
    def partition(self,x):
        
        """ algorithm
            1, create 4 pointes to track less and greater values
            2, create current pointer and next pointer
            3, current pointer.next should be none so it can connect to a less greater value
            4, after that connect less end and greater beginning
            5,
        """

        ## cur poiner should be killled every time
        ## cuz it stays, 

        Less_front = None
        Less_end = None
        greater_or_equal_front = None
        greater_or_equal_end = None
        cur = self.head
        res = []
        while cur:
            next_node = cur.next # store temp  
            cur.next = None # kill pointer so can be connected to the right node
            if cur.data < x:
                if Less_front == None: # creat head of Less_front
                    Less_front = cur
                    Less_end = cur 
                else:
                    Less_end.next = cur # front end points to new values
                    Less_end = cur # keep the end of list to new values
            else:
                if greater_or_equal_front == None:
                    greater_or_equal_front = cur
                    greater_or_equal_end = cur
                else:
                    greater_or_equal_end.next = cur
                    greater_or_equal_end = cur

            cur = next_node # temp is created so it can move on
            #cur = cur.next # doesnt work here cuz it is alredy None

        Less_end.next = greater_or_equal_front
                    
        
        while Less_front:
            res.append(Less_front.data)
            Less_front = Less_front.next

        return res

    def oddEvenList(self):
            
            cur = self.head
            odd = None
            even = None
            odd_e = None
            even_e = None
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
            
            while odd:
                print(odd.data)
                odd = odd.next
            

llist = LinkedList()
llist.head = Node(28)
llist.head.next = Node(15)
llist.head.next.next = Node(6)
llist.head.next.next.next = Node(9)
llist.head.next.next.next.next = Node(1)
llist.head.next.next.next.next.next = Node(2)
llist.head.next.next.next.next.next.next = Node(27)
llist.head.next.next.next.next.next.next.next = Node(2)
##
###llist.display()
###llist.removeDuplicates()
##llist.display()
print(llist.returnKthToLast(3)) # 2

###llist.deleteMiddle(llist)
#llist.display()
##
print(llist.partition(6))
#llist.oddEvenList()
##llist.display()

class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
        self.prev = None

class LinkedList:
    def __init__(self):
        self.head = None

    def reverseHelper(self,node):
        # no head so create pointer instead of head
        pre_node = None
        next_node = None
        while node:
            next_node = node.next
            node.next = pre_node
            pre_node = node
            node = next_node # node becomes None as next_node is node at the end 
            
        self.head = pre_node # pre_node points to the last value (the front cue it is reversed)

        
   # CTCI 2.5
    def sumlists(self,node1, node2):
        """
        algorithm
        1, add two nodes py position
        2, carry goes to next positoin
        3, create another linkedlist for sum
        4, reverse sum

        """
        # reverse list
##        reversed_node1 = self.reverse(node1)
##        reversed_node2 = self.reverse(node2)
        

        cur1 = node1.head
        cur2 = node2.head
        # dummy will stay at the beginning of the list
        dummy = cur = Node(0)
        carry = 0
        while carry or cur1 or cur2:
            if cur1:
                # get sum by using carry
                carry += cur1.data
                cur1 = cur1.next
            if cur2:
                carry += cur2.data
                cur2 = cur2.next

            # get last digit of sum
            cur.next = Node(carry%10)
            # this is the list to return 
            cur = cur.next
            # get carry over value 
            carry = carry // 10

        res = []
        reverse_res = self.reverseHelper(dummy.next)
        while reverse_res:
            #res.append(reverse_res.data)
            reverse_res = reverse_res.next        
        
##        temp_front = None
##        temp = None # this node needs to be created 
##        prev_node = None # pointer
##        carry = 0
##        while cur1 or cur2: # this should be or here cuz when one list is longer the other one
##                            # if it is or, becomes true so loop keeps going
##
##            #cur1_data = 0 if cur1.data == None else cur1.data # if cur1 is short it will vanish
##            # so cur1.data is not none, doest not even exist
##                                                            
##            cur1_data = 0 if cur1 == None else cur1.data # if there is no value in cur1, returns 0
##            cur2_data = 0 if cur2 == None else cur2.data # # if there is no value in cur1, returns 0
##            
##            Sum = carry + cur1_data + cur2_data
##            carry = 0 if Sum < 10 else 1
##            Sum = Sum % 10 if Sum >= 10 else Sum
##
##            temp = Node(Sum)
##            if temp_front == None:
##                temp_front = temp
##                #prev_node = temp
##            else:
##                prev_node.next = temp
##
##            prev_node = temp
##            # without if statement 
##            #cur1 = cur1.next # if cur1 is short, crushes 
##            #cur2 = cur2.next # if cur2 is short, curshes
##
##            if cur1 != None: # if cur1 is longer than cur2, cur1's pointer moves on 
##                cur1 = cur1.next
##
##            if cur2 != None: # # if cur2 is longer than cur1, cur2's pointer moves on 
##                cur2 = cur2.next
##            
##        if carry > 0: # if there is carry over value, create new node with the value and connect 
##            prev_node.next = Node(carry)
##        #res = []
##        #cur_temp = temp.head
##        reverse_res = self.reverseHelper(temp_front)
##        while reverse_res:
##            #res.append(reverse_res.data)
##            reverse_res = reverse_res.next
##        #print(res)

    ## test case for 2.5 #
llist = LinkedList()
llist1 = LinkedList()
##llist.head = Node(2)
##llist.head.next = Node(4)
##llist.head.next.next = Node(3)
##llist1.head = Node(5)
##llist1.head.next = Node(6)
##llist1.head.next.next = Node(4)
llist.head = Node(7)
llist.head.next = Node(1)
llist.head.next.next = Node(6)
llist1.head = Node(5)
llist1.head.next = Node(9)
llist1.head.next.next = Node(2)
llist.sumlists(llist,llist1)
#llist.display()

##
##
##    
##
##
##    ## CTCI 2.6 ##
##    def palindrome(self,node):
##        cur = node.head
##        reverse_list = self.reverseHeadHelper(node)
##        #cur = node.head node gets updated so cur should be at the beginning,
##        # other wise same words are compaired
##        while cur and reverse_list:
##            if cur.data != reverse_list.data:
##                return False
##            reverse_list = reverse_list.next
##            cur = cur.next
##        return True
##       ## test case ##           
####l = LinkedList()
####l.head = Node('a')
####l.head.next = Node('b')
####l.head.next.next = Node('c')
####print(l.palindrome(l))
##
##        
##
##
##    ## CTCI 2.7 ## 
##    def intersection(self,node1,node2):
##        """algorithm
##            1, find the difference
##            2, move longer list based on difference so both lists are positioned same
##            3, move both pointers until node1.data = node2.data
##            """
##        res = []
##        node1_len = self.length(node1) # node1_len just points to head that does not contain data or next
##        node2_len = self.length(node2)
##        # head is just an instance and does not point to anything
##        cur1 = node1.head # cur1 gets what head points to (whole list)
##        cur2 = node2.head # cur2 gets what head points to (whole list)
##        
##        for i in range(abs(node1_len - node2_len)):
##            if node1_len < node2_len:
##                cur2 = cur2.next
##            else:
##                cur1 = cur1.next
##        while cur1 or cur2:
##            if cur1 == cur2:
##                res.append(cur1.data)
##            cur2 = cur2.next
##            cur1 = cur1.next
##            return True
##        #return res
##        return False
##            
##                
##
##            ##### test case for CTCI 2.7 #####
####llist = LinkedList()
####llist1 = LinkedList()
####llist.head = Node(7)
####llist.head.next = Node(1)
####llist.head.next.next = Node(6)
####llist.head.next.next.next = Node(23)
####llist.head.next.next.next.next = Node(55)
####
####
####llist1.head = Node(5)
####llist1.head.next = Node(9)
####llist1.head.next.next = Node(2)
####llist1.head.next  = llist.head.next.next.next
####llist1.head.next.next  = llist.head.next.next.next.next
####
####print(llist.intersection(llist,llist1))
##      
##
## 
##    ## CTCI 2.8 ##
##    def LoopDetection(self,node):
##        """algorithm
##            1, create 2 pointes 
##            2, if 2 pointers collide, return True
##            
##            """
##        slow_p = self.head
##        fast_p = self.head
##        while slow_p and fast_p and fast_p.next:
##            # if there is no fast_p.next,
              # where there is no loop it returns an error instead of False
##            slow_p = slow_p.next
              # wanna move slow not node 
##            fast_p = fast_p.next.next
              # move fast not node  
##            if slow_p == fast_p:
##                return True
##        return False
##            
##        
##llist = LinkedList()
##
##llist.head = Node("grinding")
##llist.head.next = Node("negativity")
##llist.head.next.next = Node("good ass vibes")
##llist.head.next.next.next = Node("depression")
##llist.head.next.next.next.next = Node("grinding")
##llist.head.next.next.next.next.next = Node("dream")
##llist.head.next.next.next.next.next.next = llist.head
###llist.head.next.next.next.next.next.next = Node("hussle")
##
##
##print(llist.LoopDetection(llist))
##
##
##
##
##
##
##
##
##
####llist.doublypush(1)
####llist.doublypush(2)
####llist.doublypush(3)
####llist.doublypush(4)
####llist.doublypush(5)
##
####llist.display()
###llist.reverse(llist)
###llist.display()
##
###llist.display()
##
###llist.sumlists(llist,llist1)
###llist.intersection(llist,llist1)
##
###llist.display()
