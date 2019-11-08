 ### Binary Tree Data Structure GFG
##
##class Node:
##    def __init__(self,val):
##        self.left = None
##        self.right = None
##        self.val = val
##
### create root 
##root = Node(1) 
##''' following is the tree after above statement 
##        1 
##      /   \ 
##     None  None'''
##  
##root.left      = Node(2); 
##root.right     = Node(3); 
##    
##''' 2 and 3 become left and right children of 1 
##           1 
##         /   \ 
##        2      3 
##     /    \    /  \ 
##   None None None None'''
##  
##  
##root.left.left  = Node(4); 
##'''4 becomes left chil d of 2 
##           1 
##       /       \  
##      2          3 
##    /   \       /  \ 
##   4    None  None  None 
##  /  \ 
##None None'''

##root = Node(1)
##root.left = Node(2)
##root.right = Node(3)
##root.left.left = Node(4)

# Level Order

# to check how many levels on both sides
# print values based on level from left to right

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        
# Function to  print level order traversal of tree 
def LevelOrder(root):
    h = height(root) # return the larger subtree
    for i in range(1,h+1): # starts with 1 cuz bottom node starts with 1
        displayGivenOrder(root,i)
        

#Print nodes at a given level 
def displayGivenOrder(root,level):
    # if level match with i then print
    if root == None: return

    if level == 1:
        print(root.data)

    elif level > 1:
        (displayGivenOrder(root.left,level-1))
        # if level 2, i starts with 2 so  by keep decrementing, it hits 1
        (displayGivenOrder(root.right,level-1))

def height(root):
    # find the height of the tree 
    if root == None: return 0

    else:
        left_tree = height(root.left)
        right_tree = height(root.right)

        #Use the larger one
        # deepest lever returns 1 and next level returns + 1
        if left_tree < right_tree:
            return right_tree + 1
        else:
            return left_tree + 1


root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5)
#LevelOrder(root)



# # Level Order using Queue

from collections import deque

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def queue(root):
    if root == None:
        return

    queue = deque([])
    queue.append(root) # get the top root

    while queue:
        
        # queue.popleft() root does not move
        root = queue.popleft() # this root is inheariated 
        print(root.data)

        if root.left != None:
            queue.append(root.left)
        if root.right != None:
            queue.append(root.right)

        


root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5)
#queue(root)


##from collections import deque
##class Node:
##    def __init__(self,data):
##        self.left = None
##        self.right = None
##        self.data = data
##
##def inorder(root):
##    # base case
##    if root == None: return
##
##    inorder(root.left)
##    print(root.data,end=' ')
##    inorder(root.right)
##    
##
##def insert(root,key):
##    
##    """algorithm
##        1, traverse the tree until i find a node whose either left or child is empty
##        2, and insert new node to that missng spot
##        3,
##
##"""
##    
##    q = deque([])
##    q.append(root)
##    while len(q) != 0:
##         
##        root = q.popleft() # current root pops on which next move is based
##        if root.left: # if left child exists, move on 
##            q.append(root.left) # enque that child
##        else: # if left child is missing, insert new node
##            root.left = Node(key)
##            break # base case
##            
##        if root.right:
##            q.append(root.right) # if left child exists, move on
##
##        else: # if right child is missing, insert new node
##            root.right = Node(key)
##            break # base case
##        
##
##root = Node(10)  
##root.left = Node(11)  
##root.left.left = Node(7)  
##root.right = Node(9)  
##root.right.left = Node(15)  
##root.right.right = Node(8)  
##
##print("Inorder traversal before insertion:") 
##inorder(root)  
##
##key = 12
##insert(root, 12)
##print()
##print("Inorder traversal after insertion:") 
##inorder(root)


# deletion in Binary Tree


#from collections import deque

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def inorder(root):
    if root== None:
        return
    inorder(root.left)
    print(root.data,end=' ')
    inorder(root.right)

def findmostRight(root,deepest): #
    q = deque([])
    q.append(root)
    while len(q) != 0:
        # dont want root to move
        root = q.popleft()

        # find most right
        if root.right:
            if root.right.data == deepest:
                root.right = None
            else:
                q.append(root.right)
                

        if root.left:
            if root.left.data == deepest:
                root.left = None
            else:
                q.append(root.left)
           
    
        
def findDelete(root,target):
    ## update the deleted node


    # base case
    if root == None: return None
    # if top root is the target
    if root.right == None and root.left == None:
        if root.data == target:
            return None
        else:
            return root

    target_node = None
    q = deque([])
    q.append(root)
    while len(q) != 0:
        # create temp so root dose not move
        temp = q.popleft()
        if temp.data == target:
            target_node = temp

        if temp.left:
            q.append(temp.left)

        if temp.right: 
            q.append(temp.right)

    if target_node:
        deepest = temp.data # the last value that is enqued
        findmostRight(root,deepest)
        target_node.data = deepest
        
    return root # return here so passed into inrder
    


##root = Node(10) 
##root.left = Node(11) 
##root.left.left = Node(7) 
##root.left.right = Node(12) 
##root.right = Node(9) 
##root.right.left = Node(15) 
##root.right.right = Node(8) 
##print("The tree before the deletion:") 
##inorder(root)
##root = findDelete(root, 11) 
##print() 
##print("The tree after the deletion;") 
##inorder(root)   
            


# AVL Tree


##class Node:
##    # A utility function to create a new node 
##    def __init__(self,key):
##        self.key = key
##        self.left = None
##        self.right = None
##        self.height = None
##        
##
##class AVL_tree:
##    def Insert(self,root,key):
##        if root == None:
##            root = Node(key)
##
##        if key > root.key:
##            insert(root.right,key)
##        else:
##            insert(root.left,key)
##            
##        return root
##
##    def preorder(self,root):
##        
##        print(root.data)
##        preorder(root.left)
##        preorder(root.right)
##            
##
##myTree = AVL_tree() 
##root = None
##  
##root = myTree.Insert(root, 10) 
##root = myTree.Insert(root, 20) 
##root = myTree.Insert(root, 30) 
##
##print("Preorder traversal of the", 
##      "constructed AVL tree is") 
##myTree.preOrder(root) 


# 6/29 / 2019

# Symmetric Tree (Mirror Image of itself)

class Node:
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None
    """ For two trees to be mirror images, the following three 
        conditions must be true 
        1 - Their root node's key must be same 
        2 - left subtree of left tree and right subtree 
          of the right tree have to be mirror images 
        3 - right subtree of left tree and left subtree 
           of right tree have to be mirror images 
    """

def mirror(root1,root2):
    # edge case
    # If both trees are empty, then they are mirror images
    if root1 == None and root2 == None:
        return True
    if root1 != None and root2 != None:
        if root1.key == root2.key:
            return mirror(root1.left,root2.right) # return when false
            return mirror(root1.right,root2.left)
            return True
    return False
    
def symmetric(root):
    return mirror(root,root) # duplicate trees and pass them in 
    
root = Node(1) 
root.left = Node(2) 
root.right = Node(2) 
root.left.left = Node(3) 
root.left.right = Node(4) 
root.right.left = Node(4) 
root.right.right = Node(3)
#root.right.right.right = Node(8)
#print(symmetric(root))

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def printPreorder(root):
    if root != None:
        
        print(root.data,end=' ')
        printPreorder(root.left)
        printPreorder(root.right)
        
def printInorder(root):
    if root != None:
        
        printInorder(root.left)
        print(root.data,end=' ')
        printInorder(root.right)

def printPostorder(root):
    if root != None:
        printPostorder(root.left)
        printPostorder(root.right)
        print(root.data,end=' ')
        
    


root = Node(1) 
root.left      = Node(2) 
root.right     = Node(3) 
root.left.left  = Node(4) 
root.left.right  = Node(5)
##
##print("Preorder traversal of binary tree is")
##printPreorder(root) 
##  
##print("\nInorder traversal of binary tree is")
##printInorder(root) 
##  
##print("\nPostorder traversal of binary tree is")
##printPostorder(root) 
##        


# Inorder Tree Traversal Iteratively
### DFS

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def inorder(root): # using stack without recursion

    stack = []
    cur_node = root
    stack.append(cur_node)
    while stack:
        while cur_node.left:
            stack.append(cur_node.left)
            cur_node = cur_node.left
            
        popped = stack.pop()
        print(popped.data)

        if popped.right:
            cur_node = popped
            stack.append(cur_node.right)
            cur_node = cur_node.right

    return 'stack empty'

root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5)
print(inorder(root),'a')






# Inorder Successor of a node in Binary Tree
# Inorder Successor of a node in binary tree is the next node



class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def inorderSuccessor(root,node):
    # use stack
    if root == None: return

    s = []
    s.append(root)
    cur = root
    prev_data = None
    while len(s) != 0:
        while cur.left:
            s.append(cur.left)
            cur = cur.left

        popped = s.pop()
        if prev_data == node: return popped.data

        prev_data = popped.data

        while popped.right:
            s.append(popped.right)
            #cur = popped.right # popped.right points to the same thing
            popped = popped.right
    return None

        
    
root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5)
#print(inorderSuccessor(root,1))



# Find n-th node of inorder traversal
# not completed

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def find_nth_node(root,Nth_node):

    if root == None: return

    stack = []
    res = []
    stack.append(root)
    cnt = 0
    while len(stack) != 0:
        while root.left:
            stack.append(root.left)
            root = root.left

        popped = stack.pop()
        res.append(popped.data)
        if cnt == Nth_node:
            return popped.data

        cnt +=1

        #while popped.right:
        if popped.right:
            stack.append(popped.right)
            popped = popped.right

        root = popped
        
    print(res)
    return None


root = Node(7) 
root.left = Node(2) 
root.right = Node(3) 
root.right.left = Node(8) 
root.right.right = Node(5)
#print(find_nth_node(root,3))

## DFS

##class Node:
##    def __init__(self,data):
##        self.left = None
##        self.right = None
##        self.data = data
##
##def preorderDFS(root):
##    "root->left->right"
##   
##    
##
##def postOrder(root):
##    # left->right->root
##    
##
##            
####def inorderDFS(root):
####    # left->root->right
##    
##    
##root = Node(1)
##root.left = Node(2)
##root.right = Node(3)
##root.left.left = Node(4)
##root.left.right = Node(5)
##root.right.left = Node(6)
##root.right.right = Node(7)
##
##print(preorderDFS(root))
    


##class Node:
##    # A utility function to create a new node 
##    def __init__(self,data):
##        self.data = data
##        self.left = None
##        self.right = None
##        
##
##def postorder(root,Nth_Node):
##   
##    cnt = [0]
##    if root != None:
##        postorder(root.left,Nth_Node)
##        postorder(root.right,Nth_Node)
##        cnt[0] += 1
##        if cnt[0] == Nth_Node:
##            return root.data
##
##
##root = Node(25)  
##root.left = Node(20)  
##root.right = Node(30)  
##root.left.left = Node(18)  
##root.left.right = Node(22)  
##root.right.left = Node(24)  
##root.right.right = Node(32)  
##  
##print(postorder(root,2))   

# Check for Children Sum Property in a Binary Tree

# data value must be equal to sum of data values in left and right children.
# Consider data value as 0 for NULL children. Below tree is an example

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def Sumcheck(root):

    """algorithm,
        1, traverse tree using level order (queue)
        2, check if node's sum is equal to its children
        3, if root is none return 0


"""
    if root == None: return 

    q = deque([])
    q.append(root)
    while len(q) != 0:
        popped = q.popleft()
        if popped.left != None or popped.right != None:
            
            if popped.left:
                q.append(popped.left)
                left = popped.left.data
            else:
                left = 0
                
            if popped.right: # check if they exisit or not
                q.append(popped.right)
                right = popped.right.data
            else:
                right = 0

            if (left + right) != popped.data:
                return False
    return True


root = Node(10)  
root.left = Node(8)  
root.right = Node(2)  
root.left.left = Node(3)  
root.left.right = Node(5)  
root.right.right = Node(2)
root.left.right.left = Node(3)
root.left.right.right = Node(2)
#print(Sumcheck(root))


# LAC in Binary Tree
# Lowest Common Ancestor in a Binary Tree

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None



def lca(root,n1,n2):

    """algorithm
        1, traverse tree with recursion 
        2, if one of the keys found, flag it,
        3, 
        4,

"""
    # recursion
    # store the target node
    # base case
    if root == None: return

    # if one of the targets values found
    if root.data == n1 or root.data == n2:
        return root.data

    # whatever that is returned here is the target node
    ########
    # finds root's left and right children 
    target_in_left_tree = lca(root.left,n1,n2)
    tartget_in_right_tree = lca(root.right,n1,n2)

    if target_in_left_tree and tartget_in_right_tree:
        return root.data

##    else:
##        return target_in_left_tree or tartget_in_right_tree

    # if traverse all the way here, keys are not in same
    # without this, when target found it is never stored,
    # as soon as target found, it will go to the other subtree and return nothing
    return target_in_left_tree if target_in_left_tree else tartget_in_right_tree
    


root = Node(1) 
root.left = Node(2) 
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
root.left.left = Node(4) 
root.left.right = Node(5) 

n1 = 2
n2 = 66
## assuming both vlaues are in tree
# otherwise it dosent work, as 4 is not checekd cuz they are in the same subtree
  
print("LCA of" ,n1 ,"and" ,n2, "is", lca(root, n1, n2,))


# Data Structure 14, 17, 21, 22


# the Maximum Depth or Height of a Tree

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def maxDepth(root):
    if root == None: return 0

    left_tree = maxDepth(root.left)
    right_tree = maxDepth(root.right)

    if left_tree < right_tree:
        return right_tree + 1
    else:
        return left_tree + 1

    
root = Node(1) 
root.left = Node(2) 
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
root.left.left = Node(4) 
root.left.right = Node(5)
root.left.right.left = Node(5)
root.left.right.left.right = Node(5)
root.left.right.left.right.left = Node(5)
root.left.right.left.right.left.right = Node(5)

#print(maxDepth(root))

# Connect nodes at same level

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        #self.p = None
        self.next_right = None

def connectLevel(root):

    # This approach works only for Complete Binary Trees
    """algorithm
        1, traverse the tree using level order
        2, 
        3,
        
"""
    if root == None: return

    if root.left:
        root.left.next_right = root.right

    if root.right:
        if root.next_right:
            # why does next_right exist?
            #it is same as child, so now root has 3 child left, right and right_next
            root.right.next_right = root.next_right.left

        else:
            root.right.next_right = None

    connectLevel(root.left)
    connectLevel(root.right)
    
     
    

##root = Node(1) 
##root.left = Node(2) 
##root.right = Node(3)
##root.right.left = Node(6)
##root.right.right = Node(7)
##root.right.right.right = Node(11)
##root.right.right.left = Node(12)
##root.left.left = Node(4) 
##root.left.right = Node(5)
##root.left.left.left = Node(9)
##root.left.left.right = Node(10)
##connectLevel(root)  
##if root.right.next_right: 
##    print(root.right.next_right.data)
##if root.left.right.next_right: 
##    print(root.left.right.next_right.data)
##if root.left.left.right.next_right: 
##    print(root.left.left.right.next_right.data)
####if root.left.right.p: 
####    print(root.left.right.p.data)
##else:
##    print('None')
##
##

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        #self.p = None
        self.next_right = None

def getRight(root):
    # edge case for when it is not perfect tree
    # left.left.left can get right.right.right as current node (1 level above) are connected
    temp = root.next_right
    while temp:
        if temp.left:
            return temp.left
        if temp.right:
            return temp.right
        temp = temp.next_right
    return None

"""algorithm
        1, traverse the tree using level order
        2, traverse all the values on the same level
         3, traverse last leaves
"""


def connectLevel(root):
    
    # iterative solution
    if root == None: return

    while root:
        temp_root = root
        while temp_root:
            if temp_root.left:
                if temp_root.right:
                    temp_root.left.next_right = temp_root.right

                else:
                    temp_root.left.next_right = getRight(temp_root)

            if temp_root.right:
                # right child gets to another child in another subtree
                temp_root.right.next_right = getRight(temp_root)

            temp_root = temp_root.next_right
            #########temp_root  =  temp_root.next_right
            # this temp_root traverse all nodes on the current level
            #temp_root.left.next_right = getRight(root)
            # if theres no children, move over to get_right(next value on the same level)

        # after traversing all values on the level
        if root.left:
            root = root.left

        # elif so if stament above true, this elif will not occur
        elif root.right:
            root = root.right
        # if there is no child,
        # root is alwasy connected on the same level
        # traverse all the leaves in the end
        else:
            root = getRight(root)



root = Node(1) 
root.left = Node(2) 
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
root.right.left.left = Node(131)
root.right.right.right = Node(12)
root.right.right.left = Node(11)
root.left.left = Node(4) 
root.left.right = Node(5)
root.left.right.right = Node(88)
root.left.left.left = Node(9)
root.left.left.right = Node(10)
##connectLevel(root)
##
##if root.right.left:
##    print(root.right.left.next_right.data)
##else:
##    print('None')
##if root.right.next_right: 
##    print(root.right.next_right.data)
##else:
##    print('None')
##if root.left.right.next_right: 
##    print(root.left.right.next_right.data)
##else:
##    print('None')
##if root.left.left.right.next_right: 
##    print(root.left.left.right.next_right.data)
##else:
##    print('None')
##if root.right.right.left.next_right:
##    print(root.right.right.left.next_right.data)
##else:
##    print('None')
##if root.left.right.p: 
##    print(root.left.right.p.data)


## Check if a binary tree is subtree of another binary tree | Set 1



class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        
def checkHelper(root1,root2):
    if root1 == None and root2 == None:
        return True

    if root1 == None or root2 == None:
        return False
    
    
    if root1.data == root2.data:
        return checkHelper(root1.left,root2.left) or checkHelper(root1.right,root2.right)

def subTree(root1,root2):
    if not root1 or not root2:
        return

    if checkHelper(root1,root2):
        # when false nothing is returnd which is why
        # return Subtree below
        return True

    return subTree(root1.left,root2) or subTree(root1.right,root2)
    


T = Node(26) 
T.right = Node(3) 
T.right.right  = Node(3)
T.right.left  = Node(9)
T.right.left.right  = Node(0) 
T.left = Node(10) 
T.left.left = Node(4) 
T.left.left.right = Node(30) 
T.left.right = Node(6) 
  
""" TREE 2 
     Construct the following tree 
          10 
        /    \ 
      4      6 
       \ 
        30 
    """

S = Node(3) 
S.right  = Node(3)
S.left  = Node(9)
#S.left.right  = Node(9) 

##S = Node(10) 
##S.right = Node(6) 
##S.left = Node(4) 
##S.left.right = Node(30)
print(subTree(T,S))



#find maximum path

class Node:
    def __init__(self,data):
        self.left = None
        self.right = None
        self.data = data

class FindbiggestPath:
    def returnSum(self,root):
        self.Sum = float('-inf')
        self.findPath(root)
        return self.Sum

    def findPath(self,root):
        if root == None: return 0

        left_tree = self.findPath(root.left)
        right_tree = self.findPath(root.right)

        if left_tree or right_tree:
##        left_tree = left_tree if left_tree > 0  else 0
##        right_tree = right_tree if right_tree > 0 else 0

        # get subtree's sum
        # it will return the biggest sum and the path already seenis greater,
        # the current path will be passed up
            self.Sum = max(self.Sum, (left_tree + right_tree) + root.data)

        return max(left_tree,right_tree) + root.data

        
root = Node(-10)
root.left = Node(2) 
root.right   = Node(10) 
root.left.left  = Node(20)
root.left.right = Node(1)
root.right.right = Node(-25)
root.right.right.left   = Node(3) 
root.right.right.right  = Node(4)
s = FindbiggestPath()
print(s.returnSum(root))




