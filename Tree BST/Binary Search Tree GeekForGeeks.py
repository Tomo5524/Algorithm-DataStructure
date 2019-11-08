# Binary Search Tree

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def insert(root,node):
    """traverse tree and when you hit root that is none, insert
        if insert data is greater than node go to left otherwise go right

    

    """
    
    if root == None:
        root = node
    if root.data > node.data:
        if root.left:
            insert(root.left,node)
        else:
            root.left = node
    else:
        if root.right:
            insert(root.right,node)
        else:
            root.right = node
                   
def inorder(root):
    if root == None:
        return

    inorder(root.left)
    print(root.data,end = ' ')
    inorder(root.right)

def search(root,key):
    if root.data ==key:
        print('found')
        return root.data

    if key > root.data:
        return search(root.right,key)

    else:
        return search(root.left,key)

def delete(root,key):
    """algorithm
        1, Node to be deleted is leaf: Simply remove from the tree.
        2,  Node to be deleted has only one child:
            Copy the child to the node and delete the child
        3,  Node to be deleted has two children:
            Find inorder successor of the node. Copy contents of the inorder successor to the node
            and delete the inorder successor.

"""
    # base case
    if root == None: return root

    # determine which tree the key lies in

    if key < root.data:
        # variable is root.left cuz when there is only one child,
        # root.left gets new node, e.g. [5,3,2] 3 only has one child
        # root(5).left gets 2 
        root.left = delete(root.left,key)

    elif key > root.data:
        root.right = delete(root.right,key)

    # when current root is key
    # Node to be deleted has only one child:
    # Copy the child to the node and delete the child
    else:
        if root.left == None:
            temp = root.right
            root = None
            return temp

        if root.right == None:
            temp = root.left
            root = None
            return temp

        # Node with two children: Get the inorder successor 
        # (smallest in the right subtree)
        # when there are 2 children, pick smallst in right subtree
        # which is after one move to right, leftmost child
        temp = smallest(root.right)
        

        root.data = temp.data

        root.right = delete(root.right,temp.data)


    return root

def smallest(root):
    while root.left:
        root = root.left

    return root




# Find the node with minimum value in a Binary Search Tree

def findmin(root):
    """algorithm
        1, smallest is alwas left most at the bottom
        2,
        3,

"""
    if root == None: return

    while root.left:
        root = root.left

    return root.data
    




r = Node(5)
insert(r,Node(3)) 
insert(r,Node(2)) 
insert(r,Node(4)) 
insert(r,Node(7)) 
insert(r,Node(6)) 
insert(r,Node(8))
insert(r,Node(1)) 
# Print inoder traversal of the BST
#print(findmin(r))

##inorder(r)
##print(search(r,6))
###delete(r,2)
###inorder(r)
##delete(r,3)
###print()
##inorder(r)


##class pre_suc:
##    def findpre_suc(self,root):
##        self.pre = None
##        self.suc = None
##
##    def pre_succ_inorder(self,root,key):
##        if root == None: return
##
##        if root.data == key:
##            if root.left:
##                temp = root.left
##                while temp.right:
##                   temp = temp.right
##                self.pre = temp
##                
##            if root.right:
##                temp = root.right
##                while temp.left:
##                    temp = temp.left
##                return temp
##        if root.data > key:
##            self.pre = root
##            self.pre_succ_inorder(root.left,key)
##
##        else:
##            self.suc = root
##            self.pre_succ_inorder(root.right,key)
##
##
##        if self.pre and self.suc:
##            print('pre',self.pre.data)
##            print('suc',self.suc.data)
##            return 
##        if self.pre:
##            print('pre',self.pre.data)
##            return
##        if self.suc:
##            print('suc',self.suc.data)
##            return
##        return None

##r = Node(5)
##insert(r,Node(3)) 
##insert(r,Node(2)) 
##insert(r,Node(4)) 
##insert(r,Node(7)) 
##insert(r,Node(6)) 
##insert(r,Node(8))
##s = pre_suc()
##print(s.pre_succ_inorder(r,4))
        

### Inorder predecessor and successor for a given key in BST


def find_p_s(root, key,pre,suc):
    """algorithm
        1, traverse tree with inorder
        2, append all smaller value update it  
        3, the first grater value appened than key is successor 

"""
    if root == None: return

    find_p_s(root.left, key,pre,suc)

    if root.data < key:
        # get predecessor
        pre[0] = root

    elif root.data > key:
        # first suceccor will be appended
        if not suc[0]:
            suc[0] = root

        if suc[0] and suc[0].data > root.data:
            suc[0] = root
            
    find_p_s(root.right, key,pre,suc)
            

r = Node(5)
insert(r,Node(3)) 
insert(r,Node(2)) 
insert(r,Node(4)) 
insert(r,Node(7)) 
insert(r,Node(6)) 
insert(r,Node(8))
#print(find_p_s(r, 6))
pre = [None]
suc = [None]
find_p_s(r, 5, pre, suc)
##if(pre[0]) : print(pre[0].data, end = "")
##if(suc[0]) : print("", suc[0].data)

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def insert(root,node):
    if root == None:
        root = node
    if root.data > node.data:
        if root.left:
            insert(root.left,node)
        else:
            root.left = node
    else:
        if root.right:
            insert(root.right,node)
        else:
            root.right = node
                   
def inorder(root):
    if root == None: return
    inorder(root.left)
    print(root.data,end = ' ')
    inorder(root.right)

from collections import deque

# Check if a binary tree is BST or not

#def isBST(root):
def isBST(root, left = None, right = None):
    """algorithm
        1, traverse tree and if current node is less than left child, false
        2, if current node is greater than right child, false


"""
  
    if root == None: return True

    if left != None and root.data < left.data:
        return False
    if right != None and root.data > right.data:
        return False

    return isBST(root.left, left, root) and isBST(root.right, root, right)


# Lowest Common Ancestor in a Binary Search Tree.

def lca(root,t1,t2):
    if root == None: return None

    if root.data > t1 and root.data > t2:
        return lca(root.left,t1,t2)

    if root.data < t1 and root.data < t2:
        return lca(root.right,t1,t2)

    return root.data

def kthmin(root,k):
    
    res = []
    stack = [root]
    cur_root = root
    while stack:
        while cur_root.left:
            stack.append(cur_root.left)
            cur_root = cur_root.left
            
        popped = stack.pop()
        res.append(popped.data)
        if popped.right:
            #cur_root = popped.right
            #cur_root = cur_root.right
            # make 2 moves to the right
            cur_root = popped
            stack.append(cur_root.right)
            cur_root = cur_root.right
      
    print(res)
    return res[k]


root = Node(3)  
root.left = Node(2)  
root.right = Node(5)  
root.right.left = Node(4)  
root.right.right = Node(6)  
###root.right.left.left = newNode(40) 
#print(isBST(root))
#print(lca(root,4,6))
print(kthmin(root,2))

# Inorder Successor in Binary Search Tree

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None

def insert(root,node):
    if root == None:
        root = node
    if root.data > node.data:
        if root.left:
            insert(root.left,node)
        else:
            temp = node
            root.left = node
            temp.parent = root
            
            
    else:
        if root.right:
            insert(root.right,node)
        else:
            temp = node
            root.right = node
            temp.parent = root
            
            

def inorderSuc(root,key):
    """algorithm
        1, when inserting, create parent pointer so node can travel up
        ,2
        3,

"""
    # base case
    if root == None: return

    

    if root.data == key:
        # if target'node right exists,
        # suc is smallest value in the right subtree( leftmost value)
        if root.right:
            temp = root.right
            while temp.left:
                temp = temp.left
            return temp.data

        if root.right == None:
            if root.parent:
                temp = root.parent
                # if parent's left child is target,
                # the parent node is sucessor 
                if temp.left == root:
                    return temp.data
                # if target is not paretns's left child,
                # travese all the way up
                while temp.parent:
                    temp = temp.parent
                return temp.data         
    
    return inorderSuc(root.left,key) or inorderSuc(root.right,key)
    
def inorder(root):
    stack = []
    res = []

    while stack or root:
        if root:
            stack.append(root)
            root = root.left
        else:
            popped = stack.pop()
            res.append(popped.data)
            root = popped.right

    return res

def findKth(root,kth,cnt):
    stack = []
    res = []

    while stack or root:
        if root:
            stack.append(root)
            root = root.left
        else:
            popped = stack.pop()
            res.append(popped.data)
            cnt += 1
            if cnt == kth:
                return popped.data
            
            root = popped.right

    return res


# Merge two BSTs with limited extra space
# print the elements of both BSTs in sorted form.

def merge(r1,r2):
    """algorithm
        1, traverse both trees and convert them into sorted lists
        2, create BST out of these sorted lists
        3, travese sorted tree with inorder

"""
    res = []
    root1 = helpInorder(r1)
    root2 = helpInorder(r2)
    #print(root1)
    #print(root2)

    while root1 and root2:
        if root1 < root2:
            popped = root1.pop(0)
            res.append(popped)
        else:
            popped = root2.pop(0)
            res.append(popped)

    while root1:
        popped = root1.pop(0)
        res.append(popped)
    while root2:
        popped = root2.pop(0)
        res.append(popped)
    
    return res

def helpInorder(root):
    
    if root == None: return

    cur = root
    stack = []
    res = []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        popped = stack.pop()
        res.append(popped.data)
        root = popped.right

    return res

# Floor and Ceil from a BST

# floor round down
# celi rounds up

def celi(root,Input):
    if root == None: return -1

    if root.data == Input:
        return root.data

    if root.data < Input:
        return celi(root.right,Input)

    val = celi(root.left,Input)
    return val if val >= Input else root.data
    
        
root = Node(20)
insert(root, Node(8))
insert(root, Node(22))
insert(root, Node(4))
insert(root, Node(12))
insert(root, Node(10))   
insert(root, Node(14))
insert(root, Node(21))
insert(root, Node(16))
insert(root, Node(15))
insert(root, Node(13))
insert(root, Node(9))
print(inorderSuc(root,22))
#for i in range(25):
    #print(i,celi(root,i))
##root1 = Node(1)
##insert(root1, Node(9))
##insert(root1, Node(25))
##insert(root,Node(11))
##insert(root,Node(15))
#print(inorderSuc(root,12))
#print(findKth(root,3,0))
#print(inorder(root)
#print(merge(root,root1))


# leetcode
# Sorted Linked List to Balanced BST


class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
        self.prev = None

class TreeNode:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

class LinkedList:
    def __init__(self):
        self.head = None

    def sortedListToBST(self, head):
        # using slicing
        head = self.head
        if not head:
            head = self.head
        if not head.next:
            return TreeNode(head.data)
      
        slow, fast = head, head.next.next
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # tmp points to root
        tmp = slow.next
        # cut down the left child
        slow.next = None
        
        
##        print(loc)
##        front,end = head[:loc],head[loc:]
##        print(front)
##        print(end)
        # create helper function
        # to create 2 trees and merge them
        root = Node(tmp.data)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(tmp.next)
        return root

def inorder(root):
    stack = []
    res = []

    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        popped = stack.pop()
        res.append(popped.data)
        root = popped.right

    return res

llist = LinkedList()
llist.head = Node(-10)
llist.head.next = Node(-3)
llist.head.next.next = Node(0)
llist.head.next.next.next = Node(5)
llist.head.next.next.next.next = Node(9)
#llist.display()
root = llist.sortedListToBST(llist)
print(inorder(root))



