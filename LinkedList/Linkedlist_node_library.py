
class Node:
    def __init__(self,val):
        self.val = val
        self.next = None
        self.prev = None

##    def __str__(self): # pass all the metohds around
##        return self.convert_to_list(head).__str__()
        
def convert_to_list(head):
    cur = head
    res = []
    while cur:
        res.append(cur.val)
        cur = cur.next
    return res

def reverse(head):
   res = []
   prev = None
   while head:
       next_node = head.next
       head.next = prev
       prev = head
       head = next_node

   return prev

# def reverse(head):
# #def recursion_reverse(head):
#     """ algorithm
#         1, iterate all the way down, stop at the last val by 'if head.next'
#         2, pointer points to itself by head.next.next = head
#         3, kill hea
#
# """
#
#     # base case
#     if not head:
#         return None
#     
#     newHead = head
#     if head.next:
#         newHead = reverse(head.next)
#         head.next.next = head
#     head.next = None
#
#     return newHead
    
# head = Node(-10)
# head.next = Node(-3)
# head.next.next = Node(0)
# head.next.next.next = Node(5)
# head.next.next.next.next = Node(9)
# print(reverse(head))

##from collections import deque
##class Node:
##    def __init__(self, val):
##        self.val = val
##        self.next = None
##
##class TreeNode:
##    def __init__(self, val):
##        self.val = val
##        self.next = None
##        self.left = None
##        self.right = None
##
##class LinkedList:
##    def __init__(self):
##        self.head = None
##
##    def sortedListToBST(self):
##
##        # Form an array out of the given linked list and then
##        # use the array to form the BST.
##        values = convert_to_list(self.head)
##
##        # l and r represent the start and end of the given array
##        def convertListToBST(l, r):
##
##            if l > r:
##                return 
##            mid = (l+r) // 2
##            root = TreeNode(values[mid])
##            root.left = convertListToBST(l, mid-1)
##            root.right = convertListToBST(mid+1, r)
##            
##            return root
##
##        res = convertListToBST(0,len(values)-1)
##        return levelOrder(res)
##        
##def convert_to_list(head):
##    cur = head
##    res = []
##    while cur:
##        res.append(cur.val)
##        cur = cur.next
##    return res
##    
##def levelOrder(root):
##
##    q = deque([root])
##    res = []
##    while q:
##        cur = q.popleft()
##        res.append(cur.val)
##        if cur.left:
##            q.append(cur.left)
##        if cur.right:
##            q.append(cur.right)
##
##    return res
##
### [-10,-3,0,5,9]
##llist = LinkedList()
##llist.head = Node(-10)
##llist.head.next = Node(-3)
##llist.head.next.next = Node(0)
##llist.head.next.next.next = Node(5)
##llist.head.next.next.next.next = Node(9)
##print(llist.sortedListToBST())
