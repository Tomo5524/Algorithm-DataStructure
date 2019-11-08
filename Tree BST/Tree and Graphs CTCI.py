# CTCI 4 Tree and Graphs


# # Level Order using Queue

from collections import deque

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

## BFS Level Order Traversal # #

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

def queue(root):
    if root == None:
        return

    level = deque([])
    level.append(root) # get the top root
    res = []
    while level:
        new_level = []
        cur_level_nodes = []
        # queue.popleft() root does not move
        #root = queue.popleft() # this root is inheariated
        for node in level:
            cur_level_nodes.append(node.data)
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
                
        res.append(cur_level_nodes)
        level = new_level

    return res
            
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

print(queue(root))


## DFS Interatvie Inorder traversal ##

## you can apply this to so many problems

def preorder(root):
    "root->left->right"
    # recursive 
##    print(preorder(root.data))
##    preorder(root.data)
##    preorder(root.data)
 
    # key is to get right child first 
    stack = [root]
    res = []
    while stack:
        root = stack.pop()
        res.append(root.data)
        if root.right:
            stack.append(root.right)
        if root.left:
            stack.append(root.left)

    return res
        


def inorder(root):

    # recursive
##    inorder(root.left)
##    print(inorder.data)
##    inorder(root.right)
    
    if not root: return None
    
    stack = []
    res = []

    # when it reaches the original root (top),
    # there is no val in stack
    # that is why root is needed here
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        
        popped = stack.pop()
        res.append(popped.data)
        root = popped.right

    return res

def postOrder(root):
    # left->right->root
    # recusive
##    postOrder(root.left)
##    postOrder(root.right)
##    print(postOrder(root.data))

    # key is, in preorder fashion, get left child first
    # and reverse output 
    stack = [root]
    res = []
    while stack:
        root = stack.pop()
        res.append(root.data)

        if root.left:
            stack.append(root.left)
        if root.right:
            stack.append(root.right)

    return res[::-1]
            
   

root = Node(3)  
root.left = Node(2)
root.left.left = Node(1)
root.right = Node(5)  
root.right.left = Node(4)  
root.right.right = Node(6)
print("inorder, let->root->right")
print(inorder(root))
print("Preorder, root->left->right")
print(preorder(root))
print("postOrder, left->right->root")
print(postOrder(root))

## Daily coding problem 36 

##    def inorder(root,cnt):
##        if not root or cnt == 2:
##            return
##        if root.right:
##            return inorder(root.right,cnt)   
##        cnt += 1
##        if cnt == 2:
##            return root.val   
##        if root.left:
##            return inorder(root.left,cnt)  
##    cnt = 0
##    return inorder(root,cnt)

        ## answer above, when it hits the leaf, it will return the value and
        ## cnt will never be 2

def second_largest(root):
    # Python's demented scoping rules, we have to wrap count and val in a list.
    
    def inorder(root):
        # algorithm
        # if there is a righht leaf, parent is the target
        # if not, the left child is the target
        # 1, inorder from right, and when it hits leaf, increment counter

        if not root or cnt[0] == 2:
            return

        if root.right:
            inorder(root.right)

        cnt[0] += 1

        if cnt[0] == 2:
            res.append(root.data)
            return

        if root.left:
            inorder(root.left)

    cnt = [0]
    res = []
    inorder(root)
    #return inorder(root)
    return res[0]
    

root = Node(3)  
root.left = Node(2)  
root.right = Node(5)  
root.right.left = Node(4)  
root.right.right = Node(6)  
###root.right.left.left = newNode(40) 
#print(isBST(root))
#print(lca(root,4,6))
print(second_largest(root))

root1 = Node(5)  
root1.left = Node(2)  
root1.right = Node(9)  
root1.right.left = Node(8)  
root1.right.left.left = Node(6)
print(second_largest(root1))    


class Node:
    def __init__(self,data):
        self.data = data
        self.left = self.right = None

def buildTree(preorder, inorder):

    """
    1, Keep pushing the nodes from the preorder into a stack
        (and keep making the tree by adding nodes to the left of the previous node)
        until the top of the stack matches the inorder.
    2, At this point, pop the top of the stack until the top does not equal inorder
        (keep a flag to note that you have made a pop).
    3, Repeat 1 and 2 until preorder is empty.
        The key point is that whenever the flag is set,
        insert a node to the right and reset the flag.

"""
    if not preorder:
        return None

    # get root
    # root is always at the beginning of preorder
    root = Node(preorder[0])
    stack = []
    stack.append(root)
    
    pre = 1 # preorder index cnt, starts off with 1 as root is already in stack
    ino = 0 # inorder index cnt
    while pre < len(preorder):
        curr = Node(preorder[pre])
        pre += 1 # pre increments as new node from preorder went into stack
        # updatae prvious value
        prev = None # set flag false (None)
        # after 3 is spit out there is no stack as cur is 20 but there is no val in stack
        # this line casues error: while stack[-1].val == inorder[inor]:
        while stack and stack[-1].data == inorder[ino]:
        
        # if curr is right subtree of prev  
        # stack spit all the values till 
        # original value, prev
        # prev gets the value that went into stack before
            prev = stack.pop()
            ino += 1
        if prev:
            prev.right = curr
        else:
            stack[-1].left = curr
            
        stack.append(curr)
    return root
        
p = [3,9,20,15,7]
i = [9,3,15,20,7]
print(buildTree(p,i)) # convert it to str

# leetcode
# 112. Path Sum

class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def hasPathSum(root,t):
    # time complexity O(n)
    
    # edge case
    # []
    # [1],t=0
    if not root: return False
    
    
    
    # subtracting is the best when finding if sum can be made with given values
    t -= root.data
    
    # in case when sum path found before leaf
    if t == 0:
        return True
        
    # when reaching a leaf
    if not root.left and not root.right:
        return t==0
    
    return hasPathSum(root.left, t) or hasPathSum(root.right, t)



root = Node(1) 
root.left = Node(2)
root.left.left = Node(4) 
root.left.right = Node(5) 
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
t = 3

print(hasPathSum(root,t))

# 4.2 minimal tree

class Node:
    def __init__(self,data):
        self.data = data
        self.left = self.right = None

def minDepth(root):

    # BFS 
        # in this case bfs is fater
        # first node is depth 1
        if not root: return 0
        depth = 1
        queue = deque([(depth,root)])
        while queue:
            depth,cur_node = queue.popleft()
            #children = [root.left,root.right]
            if cur_node:
                if not cur_node.left and not cur_node.right:
                    return depth
                # even root.left is none 
                # it will get none as root exists
                queue.append((depth+1,cur_node.left))
                queue.append((depth+1,cur_node.right))
                
        # DFS
                
        if not root: return 0
        stack = [(1,root)]
        md = float('inf')
        while stack:
            depth,root = stack.pop()
            if root:
                # if no children,update min_depth
                if not root.left and not root.right:
                    md = min(md,depth)
                    
                stack.append((depth+1,root.left))
                stack.append((depth+1,root.right))
                
        return md
    
##    left_depth  = left_height(root)
##    right_depth  = right_height(root)
    

##def left_height(root):
##    if root == None: return 0
##
##    left_h = left_height(root.left)
##    
##    return left_h + 1
##
##def right_height(root):
##    if root == None: return 0
##
##    right_h = right_height(root.right)
##
##    return right_h + 1

root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4)
root.left.left.left = Node(8)

root.left.right = Node(5) 
print(minDepth(root))

# better approach level order


# 4.3 List of Depths


class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        #self.p = None
        self.next_right = None

def connectLevel(root):
    """"""
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
                # this is not else statement
                # for edge case where there are no children of currnet node
                temp_root.right.next_right = getRight(temp_root)

            temp_root = temp_root.next_right

        if root.left:
            root = root.left
            
        elif root.right:
            root = root.right

        else:
            root = getRight(root)

def getRight(root):
    temp = root.next_right
    while temp:
        if temp.left:
            return temp.left
        if temp.right:
            return temp.right
        temp = temp.next_right

    return temp


root = Node(1) 
root.left = Node(2)
root.left.left = Node(4) 
root.left.right = Node(5)
root.left.left.left = Node(9)
root.left.left.right = Node(10)
#root.left.right.right = Node(14)
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
root.right.right.left = Node(11)
root.right.right.right = Node(12)
root.right.left.left = Node(15)
connectLevel(root)
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


# 4.4 Check Balanced

def checkBalanced(root):
    """algorithm
        1, get heights from bot trees
        2, comapair them and if abs differnece is greater than 1, false
        3,

"""
    if root == None:
        return 0 

    left_height = leftHeight(root)
    right_height = rightHeight(root)

    return True if abs(left_height - right_height) <= 1 else False
def leftHeight(root):
    if root == None:
        return 0
    left = leftHeight(root.left)

    return left+1
    
def rightHeight(root):
    if root == None:
        return 0
    right = leftHeight(root.right)

    return right+1

root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
##root.left.left = Node(4)
##root.left.left.left = Node(4)
##root.left.left.left.left = Node(4)
##root.left.left.left.left.left = Node(4)
##root.left.left.left.left.left.left = Node(4)
##root.left.left.left.left.left.left.left = Node(4)
##root.left.left.left.left.left.left.left.left = Node(4)
root.left.right = Node(5)
#print(checkBalanced(root))


# 4.5 Validate BST

def isBST(root):

    """algorithms
        1, left child should be always less than current node
        2, right child should be always greater than current node
        3, traverse tree and if left child exists and violates the rule above, false
        4, same for right 


"""
    #base case
    if root == None: return True

    # if left child exists and 
    if root.left and root.left.data > root.data: return False
    
    if root.right and root.right.data < root.data: return False

    return isBST(root.left) and isBST(root.right)

    """algorithm
        1, create stack and put all values into list (res) with inorder 
        2, check if is sorted
        3,

"""
    # my work #
##    tree = inorder(root)
##
##    return tree == sorted(tree)
##    
##def inorder(root):
##    stack = []
##    res = []
##
##    while stack or root:
##        if root:
##            stack.append(root)
##            root = root.left
##        else:
##            popped = stack.pop()
##            res.append(popped.data)
##            root = popped.right
##
##    return res
    
    # time complexity O(n) space complexity  O(1)
    

root = Node(3)  
root.left = Node(2)
#root.left.left = Node(100)  
root.right = Node(5)  
root.right.left = Node(4)  
root.right.right = Node(6)
#print(isBST(root))

## test case ##
# [10,5,15,null,null,6,20]
# [1,1]

# 7/8/2019

# 4.6 Successor
# Write an algorithm to find the "next" node (i.e., in-order successor) in a binary search tree.


class Node:
    def __init__(self,data):
        self.data = data
        self.left = self.right = None
        self.parent = None

def insert(root, node):
    if root == None:
        root = node

    if root.data > node.data:
        if root.left:
            insert(root.left, node)

        else:
            root.left = node
            root.left.parent =root
    else:
        if root.right:
            insert(root.right, node)
        else:
            root.right = node
            root.right.parent =root


def inorderSuc(root,key):
    ##### In interview questios, it is pretty common that tree is balanced 
    """algorithm
        1, if key's right child exists,
        successor is smallest value in the right subtree (leftmost value) 
        2, if key's right does not child exists,
        travel up using parent pointer, if root's parent's left child is the root,
        that root's parent is the successor
        3, other wise, go travel up all the way to the top and find successor
        4, edge case, if key is the largest value, return None

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
                # edge case
                if temp.data < key:
                    return None
                
                return temp.data         
    
    return inorderSuc(root.left,key) or inorderSuc(root.right,key)
### it returns whatever it was returned previously so temp.data is 20
### in 14's loop, it will return 20


root = Node(20)
insert(root, Node(8))
insert(root, Node(22))
insert(root, Node(4))
insert(root, Node(12))
insert(root, Node(10))   
insert(root, Node(14))
insert(root, Node(21))
#print(inorderSuc(root,21))       
#print(root.left.left.parent.data)


# 4.7 Build Order

# graph

# 4.8 First Common Ancestor


class Node:
    # A utility function to create a new node 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None

def fca(root,t1,t2):
    """algorithm
        1, traverse tree with recursion
        2, if t1 or t2 spotted, return 
        3, and return the root of these targets

"""
    if root == None: return

    if root.data == t1 or root.data == t2:
        return root.data

    left_tree = fca(root.left,t1,t2)
    right_tree = fca(root.right,t1,t2)

    if left_tree and right_tree:
        return root.data

    # store the spotted values
    return fca(root.left,t1,t2) or fca(root.right,t1,t2)


root = Node(1) 
root.left = Node(2)
root.left.left = Node(4) 
root.left.right = Node(5) 
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
t1 = 4
t2 = 7
#print(fca(root,t1,t2))

# 4.9 BST Sequences # Bullshit dont even bother

##class Node:
##    # A utility function to create a new node 
##    def __init__(self,data):
##        self.data = data
##        self.left = None
##        self.right = None
##
##def BSTSequences(root):
##    if root == None: return
##
##    stack = []
##    res = []
##    stack.append(root)
##    while stack and root:
##        popped = stack.pop()
##        res.append(popped.data)
##        if popped.right:
##            stack.append(popped.right)
##
##        if popped.left:
##            stack.append(popped.left)
##
##        root = popped
##
##    return res
##  
##
##root = Node(7) 
##root.left = Node(5)
##root.right = Node(9)
##root.left.left = Node(2)
##root.right.right = Node(11)
##print(BSTSequences(root))
##


# 4.10 Check Subtree


def isSubtree(t,s):
    """algorithm
        1, pass current nodes(t,s) in to helper function
        2, which will check if current nodes (t,s) are same,
        3, if not, goes back to main function
        4, keep going until find identical matches
        5, when returning, if I use and, it will not search values in right tree
        as false will be returned and return stametement becomes false and
        loop will terminate

"""
    
    if t == None: return False
    # "if t == None or s == None" if s is not none and t is none
    # loop will keep goind and t will not have left child
    # and become never ending loop
    #

    if s == None: return True

    if identical(t,s):
        # when false nothing is returnd which is why
        # return Subtree below
        return True

    return isSubtree(t.left,s) or isSubtree(t.right,s)
    # return isSubtree(t.left,s) or isSubtree(t.right,s)
    # if it is and not or, right tree will not be checked as left tree return None
    # and loop will be done


def identical(t,s):
    if t == None and s == None: return True

    # without this, bus will be none after 10
    if t == None or s == None: return False

    # if current node's data is identical, keep going
    if t.data == s.data:
        return identical(t.left,s.left) and identical(t.right,s.right)

    
T = Node(26) 
T.right = Node(3) 
T.right.right  = Node(3)
T.right.left  = Node(9)
T.right.left.right  = Node(0) 
T.left = Node(10) 
T.left.left = Node(4) 
T.left.left.left = Node(30) 
T.left.left.right = Node(6) 

S = Node(4) 
S.right = Node(6)
S.left = Node(30)
print(isSubtree(T,S))
# test casse from leetcode
# [3,4,5,1,2,null,null,0]

##############################

# this test case is wrong cuz 3 and 9 are not leaves in main tree
# and it keeps going left or right
# so it is never identical

#########
##T = Node(26) 
##T.right = Node(3) 
##T.right.right  = Node(3)
##T.right.left  = Node(9)
###T.right.left.right  = Node(0) 
##T.left = Node(10) 
##T.left.left = Node(4) 
##T.left.left.right = Node(30) 
##T.left.right = Node(6) 
##
##S = Node(3) 
##S.right  = Node(3)
##S.left  = Node(9)




# 4.12 Paths with Sum

class path:
    """algorithm
        1, traverse tree, keep track of biggest path(tree)
        2,
        3,

"""
    # dont use __init__ cuz it does not return anything
    #def __init__(self,root):
    #TypeError: __init__() should return None, not 'int'
    def returnsum(self,root):
        # initiate infinite number
        # keep tracks of biggest tree
        self.biggestpath = float('-inf')
        self.get_path(root)
        return self.biggestpath

    def get_path(self,root):
        if root == None: return
        
        left_tree = self.get_path(root.left)
        right_tree = self.get_path(root.right)

        # this if statement makes left_tree integer
        # else 0 (digit compairson) makes it integer
        # that is why we do not need left_tree.data
        left_tree = left_tree if left_tree else 0
        right_tree = right_tree if right_tree else 0

        # keeps track of biggest path (tree)
        # this one might be the biggest path
        self.biggestpath = max(self.biggestpath, (left_tree + right_tree) + root.data)
        # this stament will lead to another potential biggest path
        return max(left_tree,right_tree) + root.data

                               
##root = Node(10)
##root.left = Node(2) 
##root.left.left  = Node(-20)
##root.left.right = Node(1)
##root.right  = Node(10)   
##root.right.right = Node(-25)
##root.right.right.left   = Node(3) 
##root.right.right.right  = Node(4)

root = Node(-20)
root.left = Node(-40) 
root.right   = Node(10) 
root.left.left  = Node(90)
root.left.left.left  = Node(110)
root.left.left.right  = Node(50)
root.left.right = Node(1)
root.right.right = Node(-25)
root.right.right.left   = Node(3) 
root.right.right.right  = Node(4)

s = path()
print(s.returnsum(root))



      

