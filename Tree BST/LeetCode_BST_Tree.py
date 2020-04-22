# LeetCode Tree
from tree_library import Node,inorder,levelOrder

# 4/22/2020
# 653. Two Sum IV - Input is a BST
class a:
    def findTarget(self,root, k):
        """
        algorithm
        1, deconstruct tree and create array off of it
        2, apply 2 sum algorithm
        """

        def dfs(self,root):
            if not root: return
            if root.val in self.hashset:
                return True
            self.hashset.add(k - root.val)

            return dfs(self,root.left) or dfs(self,root.right)

        self.hashset = set()
        return True if dfs(self,root) else False

# def findTarget(self, root, k):
#     """
#         algorithm
#         1, deconstruct tree and create array off of it
#         2, apply 2 sum algorithm
#         """
#
#     stack = []
#     res = []
#
#     while stack or root:
#         while root:
#             stack.append(root)
#             root = root.left
#
#         popped = stack.pop()
#         res.append(popped.val)
#         root = popped.right
#
#     # time complexity O(n)
#     dic = {}
#     for i in range(len(res)):
#         if res[i] in dic:
#             return True
#         else:
#             dic[k - res[i]] = i
#
#     return False

    # brute force
    # if len(res) < 1:
    #     return False
    # for i in range(len(res)):
    #     num = i
    #     for j in range(i + 1, len(res)):
    #         if res[i] + res[j] == k:
    #             return True
    #
    # return False

b = a()
root = Node(2)
root.left = Node(1)
root.right = Node(3)
print(b.findTarget(root,4))

c = a()
root2 = Node(5)
root2.left = Node(3)
root2.right = Node(6)
root2.left.left = Node(2)
root2.left.right = Node(4)
root2.right.right = Node(7)
print(c.findTarget(root2,9))

d = a()
root3 = Node(5)
root3.left = Node(3)
root3.right = Node(6)
root3.left.left = Node(2)
root3.left.right = Node(4)
root3.right.right = Node(7)
print(d.findTarget(root3,27))


# 3/22/2020
# 654. Maximum Binary Tree
def constructMaximumBinaryTree(nums):

    #DS used; stack
    # time complexity O(n)

    stack = []
    popped = None
    for num in nums:
        while stack and stack[-1].val < num:
            popped = stack.pop()

        root = Node(num)

        # if current root is less than the biggest so far
        # becomes right child of biggest
        if stack:
            stack[-1].right = root

        # if this popped is true, meaning this one has to become left child of root
        # e,g, 0 has to become left child of 5 as it was originally right child of 6
        if popped:
            root.left = popped

        stack.append(root)
        popped = None

    return levelOrder(stack[0])



    # max_num = max(nums)
    # loc = nums.index(max_num)
    # left, right = nums[:loc], nums[loc+1:]
    # left_root = right_root = root = Node(max_num)
    # biggest = max(left)
    # left.remove(biggest)
    # left_root.left = Node(biggest)
    # left_root = left_root.left
    # while left:
    #     biggest = max(left)
    #     left.remove(biggest)
    #     if biggest < left_root.val:
    #         left_root.left = Node(biggest)
    #         left_root = left_root.left
    #
    #     else:
    #         left_root.right = Node(biggest)
    #         left_root = left_root.right
    #
    # Max = max(right)
    # right.remove(Max)
    # right_root.right = Node(Max)
    # right_root = right_root.right
    # while right:
    #     Max = max(right)
    #     right.remove(Max)
    #     if Max < right_root.val:
    #         right_root.left = Node(Max)
    #         right_root = right_root.left
    #
    #     else:
    #         right_root.right = Node(Max)
    #         right_root = right_root.right
    #
    # return levelOrder(root)

test = [3,2,1,6,0,5]
print(constructMaximumBinaryTree(test))

# 11/13/2019
class findPath:

    # time complexity, O(n)
    def pathSum(self, root, Sum):
    # define global result and path

        def dfs(root, t, cur_path):
            if not root:
                return

            # it is basically 2 sum
            # if old_path is in dic, get its value
            cur_path += root.val
            old_path = cur_path - t
            if old_path in cache:
                self.cnt += cache[old_path]
                # self.cnt += 1
                # this will be
            cache[cur_path] = cache.get(cur_path, 0) + 1

            dfs(root.left, t, cur_path)
            dfs(root.right, t, cur_path)
            # without this, test2 will fail
            cache[cur_path] -= 1

        # 0,1 needs it when cur_path - sum is 0
        cache = {0: 1}
        self.cnt = 0
        dfs(root, Sum, 0)
        return self.cnt

    # brute force
    """
    algoirhtm,
    1, traverse tree with 2 layers
    2, creaate global counter
    """
    def pathSum(self,root, Sum):
        self.cnt = 0
        self.dfs(root, Sum)
        return self.cnt

    def dfs(self,root, Sum):
        # 1st layer of dfs

        if not root:
            return 0

        self.getpath(root, Sum)
        self.dfs(root.left, Sum)
        self.dfs(root.right, Sum)

    def getpath(self,root,Sum):
        # second layer of dfs
        # sum will not inherit any values
        # so node.left(5) can get 2 paths
        if not root:
            return 0

        Sum -= root.val
        if Sum == 0:
            self.cnt += 1

        self.getpath(root.left, Sum)
        self.getpath(root.right, Sum)

f = findPath()
root = Node(10)
root.left = Node(5)
root.left.left = Node(3)
root.left.left.left = Node(3)
root.left.left.right = Node(-2)
root.left.right = Node(2)
root.left.right.right = Node(1)
root.right = Node(-3)
root.right.right = Node(11)
print(f.pathSum(root,8))
f1 = findPath()
root = Node(1)
root.left = Node(-2)
root.right = Node(-3)
print(f1.pathSum(root,-1))
print()

# 11/11/2019
def isSymmetric(root):
    """
    algorithm,
    1, create 2 roots that check left subtree and right subtree
    2, if there is no node, is it symmetric
    3, if one of nodes is none or nodes value are not same, it is invalid

    """
    def dfs(r1, r2):

        if not r1 and not r2:
            return True

        if not r1 or not r2 or r1.val != r2.val:
            return False

        return dfs(r1.left, r2.right) and dfs(r1.right, r2.left)

    if not root:
        return True
    r1 = root
    r2 = root
    if dfs(r1.left, r2.right):
        return True

    return False


root = Node(1)
root.left = Node(2)
root.left.left = Node(3)
root.left.right = Node(4)
root.right = Node(2)
root.right.left = Node(4)
root.right.right = Node(3)
print(isSymmetric(root))
print()

# 543. Diameter of Binary Tree
class Diamiter:
    def diameterOfBinaryTree(self, root):
        """
        basically it is max path without value
        so every node keeps track its longest path
        and return it
        """
        def dfs(root):
            if not root:
                return 0

            left = dfs(root.left)
            right = dfs(root.right)

            # alwasy check if local longest path is greter than global path
            self.res = max(self.res, left + right)

            # when hits leaf, incremetn by 1
            # every node keeps track of local longest path
            return 1 + max(left, right)

        self.res = 0
        dfs(root)
        return self.res

d = Diamiter()
root = Node(1)
root.left = Node(2)
root.left.left = Node(3)
root.left.right = Node(5)
root.right = Node(2)
root.right.left = Node(6)
root.right.right = Node(9)
print(d.diameterOfBinaryTree(root))
print()

# 11/10/2019
def invertTree(root):
    """
    algorithm
    1, key is to flip children of current node
    2, so create current node and flip children by NewNode.left gets current node.right
         and NewNode.right gets current node.left
    3,
    """

    # base case
    if not root:
        return

    new_tree = Node(root.val)

    new_tree.left = invertTree(root.right)
    new_tree.right = invertTree(root.left)

    return new_tree

root = Node(4)
root.left = Node(2)
root.left.left = Node(1)
root.left.right = Node(3)
root.right = Node(7)
root.right.left = Node(6)
root.right.right = Node(9)
#[[4], [7, 2], [9, 6, 3, 1]]
print(levelOrder(invertTree(root)))
print()

# 11/7/2019

# 951. Flip Equivalent Binary Trees

def flipEquiv(root1, root2):

    # Time Complexity: O(min(N_1, N_2))

    # base case
    #  they are equivalent if they are both null.
    if not root1 and not root2:
        return True

    # constraint
    # if one of them is None or values are different, False
    if not root1 or not root2 or root1.val != root2.val:
        return False

    # traverse tree both directinon and different direction
    return (flipEquiv(root1.left, root2.left) and flipEquiv(root1.right, root2.right)  or \
            flipEquiv(root1.left, root2.right) and flipEquiv(root1.right, root2.left))

    # return (flipEquiv(root1.left, root2.left) or flipEquiv(root1.left, root2.right)) and \
    #        (flipEquiv(root1.right, root2.left) or flipEquiv(root1.right, root2.right))

root = Node(1)
root.left = Node(2)
root.left.left = Node(4)
root.left.right = Node(5)
root.left.right.left = Node(7)
root.left.right.right = Node(8)

root.right = Node(3)
root.right.left = Node(6)

root1 = Node(1)
root1.left = Node(3)
root1.left.right = Node(6)

root1.right = Node(2)
root1.right.right = Node(5)
root1.right.left = Node(4)
root1.right.right.right = Node(7)
root1.right.right.left = Node(8)


print(flipEquiv(root,root1))
print()

## DFS Interatvie Inorder traversal ##

## you can apply this to so many problems

def preorder(root): # 3 <- 2 <- 1 <- 5 <- 4 <- 6
    "root->left->right"
    # recursive
    ##    print(preorder(root.val))
    ##    preorder(root.left)
    ##    preorder(root.right)

    # key is to get right child first
    """
        algorithm,
        1, when getting nodes on the next level, 
            first append the right child to stack
            and left child is at the end of stack
        2, so stack pops out left first.
        3 do the step 1 and 2
        # key is to get right child first
        
        """
    # since stack popps, stack will append root before iteration
    stack = [root]
    res = []
    while stack:
        root = stack.pop()
        res.append(root.val)
        # get the right first
        if root.right:
            stack.append(root.right)

        if root.left:
            stack.append(root.left)

    return res

def postOrder(root): # 1 <- 2 <- 4 <- 6 <- 5 <- 3
    "left->right->root"
    # recusive
    ##    postOrder(root.left)
    ##    postOrder(root.right)
    ##    print(postOrder(root.val))

    """
        algorithm
        1, create stack
        2, apppend next level's nodes and pop the last value in stack out
        3, reverse result
    """

    # key is, in preorder fashion, get left child first
    # and reverse output
    stack = [root]
    res = []
    while stack:
        root = stack.pop()
        res.append(root.val)

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
print("inorder, left->root->right")
print(inorder(root))
print("Preorder, root->left->right")
print(preorder(root))
print("postOrder, left->right->root")
print(postOrder(root))

# 104. Maximum Depth of Binary Tree

def maxDepth(root):
    """"""

    return 1 + (max(maxDepth(root.left),maxDepth(root.right))) if root else 0

#     # base case
#     left = findleft(root.left)
#     right = findright(root.right)
#     return max(left,right)
#
# def findleft(root):
#     if not root:
#         return 0
#     left = findleft(root.left)
#     return 1 + left

# def findright(root):
#     if not root:
#         return 0
#     right = findleft(root.right)
#     return 1 + right

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.left.left.left = Node(8)
print(maxDepth(root))


def converttoBST(arr):
    """ algorithm
        1, root will be middle element,
        2, root.left will expand to the left on that middle element
        3, root.right will expand to the left on that middle element

    """
    def buildBST(arr,l,r):

        # if left is greater than right, invalid
        # base case
        if l <= r: # if not equal or greater, it will cause error
                    # does not get both front end and rear end value.
            mid = (l+r) // 2 # get middle
            root = Node(arr[mid]) # middle value becomes root
            root.left = buildBST(arr,l,mid-1)
            root.right = buildBST(arr,mid+1,r)

            return root

    return buildBST(arr,0,len(arr)-1)


test = [-10,-3,0,5,9]
print(converttoBST(test))
print(levelOrder(converttoBST(test)))

# 617. Merge Two Binary Trees
# how to add None and integer
# is tree complete? what type of tree is this
# how to traverse when one value exsits and the other dosent
def mergeTrees(t1,t2):
    """" algorithm
        1, traverse both tree, if both root exist, create new node and add them up
        2, if one of them exist, just simply return it by return t1 or t2

    """

    # my work
    if not t1 and not t2:
        return None

    if not t1:
        return t2

    if not t2:
        return t1

    root = Node(t1.val + t2.val)
    root.left = mergeTrees(t1.left, t2.left)
    root.right = mergeTrees(t1.right, t2.right)

    return root

    # if t1 and t2:
    #     # it is guaranteed both tree node exit so create new node and add them up
    #     root = Node(t1.val + t2.val)
    #     root.left = mergeTrees(t1.left, t2.left)
    #     root.right = mergeTrees(t1.right, t2.right)
    #
    #     return root
    #
    # else:
    #     # if only either t1 or t2 exists,
    #     # or neither of them exist
    #     # and no need ot create new node for it
    #     # as whatever that calls (either root.left or root.right)
    #     # will get either t1 or t2 or None
    #     # so techinically, it does not create new node
    #     # it wil l just point to an existing node
    #     return t1 or t2

    # if not t1:
    #     return t2
    #
    # if not t2:
    #     return t1
    #
    # t1.val += t2.val
    # t1.left = mergeTrees(t1.left, t2.left)
    # t1.right = mergeTrees(t1.right, t2.right)
    #
    # return t1


t1 = Node(1)
t1.left = Node(3)
t1.left.left = Node(5)
t1.right = Node(2)

t2 = Node(2)
t2.left = Node(1)
t2.left.right = Node(4)
t2.right = Node(3)
t2.right.right = Node(7)

print(inorder(mergeTrees(t1,t2)))

# 112. Path Sum
"""Given a binary tree and a sum, determine 
    if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
"""
""" algorithm
        1, traverse tree (inorder), evevry time, subtract target with current node
        2, if target becomes 0, return True
        3, be careful what needs to be found is root to leaf path, not path that has target value
            meaning if target becomes 0 on the way to path, that is not valid

    :param root:
    :param t:
    :return: bool
"""

def hasPathSum(root,t):

    # iterative DFS
    stack = []
    #stack = [root,t]
    while stack or root:
        while root:
            t -= root.val
            stack.append([root,t])
            root = root.left

        node,cur_sum = stack.pop()
        # if popped node dosent have chidren, it is a leaf node and if target is 0, return True
        if not node.left and not node.right and cur_sum == 0:
            return True
        t = cur_sum
        root = node.right

    return False

    # recursive
    # if not root:
    #     return False
    #
    # # subtracting is the best when finiding if sum can be made with given values
    # # we are specifically looking for leaf to root path
    # if not root.left and not root.right and t - root.val == 0:
    #     return True
    #
    # t -= root.val
    #
    # return hasPathSum(root.left,t) or hasPathSum(root.right,t)


root = Node(1)
root.left = Node(2)
root.left.left = Node(4)
root.left.right = Node(5)
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
t = 10
print(hasPathSum(root,t))

root1 = Node(1)
root1.left = Node(2)
t1 = 1
print(hasPathSum(root1,t1)) # false

# [5,4,8,11,null,13,4,7,2,null,null,null,1]
# 22
root2 = Node(5)
root2.left = Node(4)
root2.left.left = Node(11)
root2.left.left.left = Node(7)
root2.left.left.right = Node(2)

root2.right = Node(8)
root2.right.left = Node(13)
root2.right.right = Node(4)
root2.right.right.right = Node(1)
t2 = 22
print(hasPathSum(root2,t2))


def isSameTree(p,q):
    """ algorithm
           1, traverse both tree
            if neither of them exists, return True because they share the same value (None)
            if one of them doesn't exist or values are different, return False
          2,

    """

    if not p and not q:
        return True

    if not p or not q or p.val != q.val:
        return False

    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

print('same tree')
test = [1,2,1]
test1 = [1,1,2]
test2 = [1,2,3]
test3 = [1,2,3]
test4 = [1,2]
test5 = [1,None,2]
print(isSameTree(converttoBST(test),converttoBST(test1)))
print(isSameTree(converttoBST(test2),converttoBST(test3)))
print(isSameTree(converttoBST(test4),converttoBST(test5)))

# 4.3 List of Depths
""" algorithm
        1, create temporary node that gets current root
        2, manipulate this temporary, 
        3, if both children of current root exists,
            left child gets right child
        4, if only left child exists, connect it to far child from right subtree
        5, if only right child exists, do the same thing above
        
"""

def connectLevel(root):

    if root == None: return

    while root:
        # terminate loop if there are no nodes to explore
       	# keep this root or cannot get root.left as it moves around
        # and temp does all work instead of root

        temp_root = root
        while temp_root: # to traverse all the nodes on that level?
            if temp_root.left:
                if temp_root.right:

                # guaranteed both children exist
                # if temp has left and right child,
                # connect this left child to right child
                    temp_root.left.next_right = temp_root.right

                # if only left child exists,
                # connect it to far right child if it exits
                # if not, it will just get None
                else:
                    temp_root.left.next_right = getRight(temp_root)


            if temp_root.right:
                # this is not else statement
                # for edge case where there are no children of currnet node
                # as it temp_root.right will cause error
                temp_root.right.next_right = getRight(temp_root)

            # will get current level of right child??
            # yes, this temp_root traverse all nodes on the current level
            temp_root = temp_root.next_right  # traverse all nodes on current level

        # after traversing all values on the level
        # there are no nodes to explore on current level
        # so move down to next level
        if root.left:
            root = root.left

        # should be elif cuz if there are 2 children root will end up with right child
        # if this is if, root will end up with right child
        elif root.right:
            root = root.right

        # if there is no child,
        # # root is always connected on the same level
        # if there is a child of 15
        # it will grab its child which becomes root
        # so this will get child that does not belong to left subtree
        else:
	        # traverse all the nodes on current level
            # gets far right child if there is no child in left or current subtree
            root = getRight(root)

def getRight(root):
    # this one gets previous root's next right
    # edge case for when it is not perfect tree
    # left.left.left can get right.right.right as current node (1 level above) are connected
    temp = root.next_right
    while temp:
        if temp.left:
            return temp.left
        if temp.right:
            return temp.right
        temp = temp.next_right

    # if there is no far right children
    return temp

root = Node(1)
root.left = Node(2)
root.left.left = Node(4)
root.left.right = Node(5)
root.left.left.left = Node(9)
root.left.left.right = Node(10)
root.left.right.right = Node(14)
root.right = Node(3)
root.right.left = Node(6)
root.right.right = Node(7)
root.right.right.left = Node(11)
root.right.right.right = Node(12)
root.right.left.left = Node(15)
connectLevel(root)
if root.right.left:
   print(root.right.left.next_right.val)
else:
   print('None')
if root.right.next_right:
   print(root.right.next_right.val)
else:
   print('None')
if root.left.right.next_right:
   print(root.left.right.next_right.val)
else:
   print('None')
if root.left.left.right.next_right:
   print(root.left.left.right.next_right.val)
else:
   print('None')
if root.right.right.left.next_right:
   print(root.right.right.left.next_right.val)
else:
   print('None')

# 110. Balanced Binary Tree
""" algorithm
    1, traverse tree with recursion
    2, two subtrees of every node never differ by more than 1. 
    https://leetcode.com/problems/balanced-binary-tree/discuss/36042/Two-different-definitions-of-balanced-binary-tree-result-in-two-different-judgments
    https://leetcode.com/problems/balanced-binary-tree/discuss/387322/how-1223333444444nullnull55-is-balanced
    3, need to check every subtree not just the whole

"""
print("isBalanced")
def isBalanced(root):
    height = check(root)
    return height != -1

def check(root): # time complexity O(n) just traverse once
    if not root:
        return 0
    left = check(root.left)
    right = check(root.right)
    if abs(left - right) > 1 or left == -1 or right == -1:
        return -1
    return max(left, right) + 1

# def isBalanced(root): Time Complexity O(nlogn) # nested recursion
#     if not root:
#         return True
#
#     left = check(root.left)
#     right = check(root.right)
#
#     # return abs(left - right) < 2 it just checks top root of subtree (just whole left and right)
#     # so just this (above line) will not cover edge case where [1,2,3,3,None,10,None,4,None]
#     # subtree of 2, depth of left subtree is 2 and right is 0 so it should be false
#     return abs(left - right) < 2 and isBalanced(root.left) and isBalanced(root.right)
#
# def check(root):
#     # if not root:
#     #     return 0
#
#     return 1 + max(check(root.left), check(root.right)) if root else 0

root = Node(1)
root.left = Node(2)
root.left.left = Node(3)
root.left.left.left = Node(4)
root.right = Node(2)
root.right.right = Node(3)
root.right.right = Node(3)

root1 = Node(3)
root1.left = Node(9)
root1.right = Node(2)
root1.right.right = Node(3)

print(isBalanced(root))
print(isBalanced(root1))
print()

# 572. Subtree of Another Tree https://leetcode.com/problems/subtree-of-another-tree/discuss/386209/Python-98-speed-with-comments
""" algorithm
    1, just traverse main tree (t)
    2, if t does not exist, it is subtree (The tree s could also be considered as a subtree of itself.)
    3, when checking if they are identical, if both tree's nodes dont exist, return True as they are both None thus identiacl
    4, 

"""
print("isSubtree")
def isSubtree(t,s):

    # edge case
    if not s:
        return True

    # base case
    if not t:
        return

    # call helper that checks if it is identical
    if identical(t,s):
        return True

    # traverse left subtree and right subtree
    return isSubtree(t.left,s) or isSubtree(t.right,s)

def identical(t,s):

    # if they both dont exist
    if not t and not s:
        return True

    # if one of nodes dosent exist, or values are not the same, return False
    if not t or not s or t.val != s.val:
        return False

    return identical(t.left,s.left) or identical(t.right,s.right)

# [1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,null,1,2]
# [1,null,1,null,1,null,1,null,1,null,1,2]


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

# 285. Inorder Successor in BST
"""algorithm
        1, if key's right child exists,
            successor is smallest value in the right subtree (leftmost value) 
        2, if key's right does not exists, it is somewhere in left subtree, 
        3, so keep track of previous node and if previous node equals target
            return current node

"""

def inorderSuccessor(root, p):
    # the successor is somewhere lower in the right subtree
    # successor: one step right and then left till you can
    if p.right:
        p = p.right
        while p.left:
            p = p.left
        return p

    # the successor is somewhere upper in the tree
    stack = []
    prev = 0

    # inorder traversal : left -> node -> right
    while stack or root:
        # 1. go left till you can
        while root:
            stack.append(root)
            root = root.left

        # 2. all logic around the node
        popped = stack.pop()
        if prev == p.val:  # if the previous node was equal to p
            return popped.val  # then the current node is its successor
        prev = popped.val

        # 3. go one step right
        root = popped.right

    # there is no successor
    return None

root = Node(5)
root.left = Node(3)
root.left.left = Node(2)
root.left.right = Node(4)
root.left.left.left = Node(1)
root.right = Node(6)
print(inorderSuccessor(root,Node(6)))

root1 = Node(2)
root1.left = Node(1)
root1.right = Node(3)
print(inorderSuccessor(root1,Node(1)))

# 236. Lowest Common Ancestor of a Binary Tree

# when root is common ancestor, what to do, just return it
# kinda weird question cuz if one of them doesnt exist in tree,
# still returns true

def LCA(root,t1,t2): ### assume both of the inputs exist
    """algorithm
        1, traverse tree with recursion
        2, if t1 or t2 spotted, return
        3, and return the root of these targets

"""
    # time complexity is O(n)
    if root == None: return

    if root.val == t1 or root.val == t2:
        return root.val

    left = LCA(root.left,t1,t2)
    right = LCA(root.right,t1,t2)

    if left and right:
        return root.val

    # store the spotted values
    #return fca(root.left,t1,t2) or fca(root.right,t1,t2) # makes it O(n2)
    return left or right


root = Node(3)
root.left = Node(5)
root.left.left = Node(6)
root.left.right = Node(2)
root.left.left.left = Node(7)
root.left.left.right = Node(4)
root.right = Node(1)
root.right.left = Node(0)
root.right.right = Node(8)
t1 = 4
t2 = 9
print('LCA')
print(LCA(root,t1,t2))


# 102. Binary Tree Level Order Traversal
"""algorithm
    1, traverse tree with BFS to visit every level along with appending all values on each level
    2, manipulate list that holds nodes form previous level and nodes in current level
    3, 

"""
from collections import deque

def levelOrder(root):

    queue = [root]
    res = []
    while queue: # queue is just a list of current level
        # this stores all the values on the current level
        cur_level = []
        next_level = []
        ### this cur_level is now Node instance as it just popped out,
        ### however, if it was in list, it is iterable,
        ### so if just for loop over queue, it is deque instance which is iterable
        #cur_level = queue.popleft()
        # queue.popleft() root does not move
        # root = queue.popleft() # this root is inherited # 'Node' object is not iterable
        # for i in root # dosent work cuz i points to root
        # whihc is not iterable
        for node in queue:
            cur_level.append(node.val)
            # grab currnet node's chidren so it is BFS
            if node.left:
                next_level.append(node.left)

            if node.right:
                next_level.append(node.right)

        res.append(cur_level)
        queue = next_level

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
print(levelOrder(root))

# 103. Binary Tree Zigzag Level Order Traversal

""" algorithm
    1, do level order where you create list for each level
    2, if level is even, reverse the list of current level
    3,

"""

def zigzagLevelOrder(root):

    level = [root]
    res = []
    level_cnt = 0
    while level:
        new_level = []
        cur_level = []
        for node in level:
            cur_level.append(node.val)
            if node.left:
                new_level.append(node.left)

            if node.right:
                new_level.append(node.right)

        if level_cnt % 2 == 0: # current level is even, reverse it
            cur_level  = cur_level[::-1]
            # cur_level[::-1] # just copies so not be updated

        res.append(cur_level)
        level = new_level

    return res

root = Node(3)
root.left = Node(9)
root.right = Node(20)
root.right.left = Node(15)
root.right.right = Node(7)
# root.left.left = Node(4)
# root.left.right = Node(5)
# root.right.left.left = Node(131)
# root.right.right.right = Node(12)
# root.right.right.left = Node(11)
# root.left.right.right = Node(88)
# root.left.left.left = Node(9)
# root.left.left.right = Node(10)
print(zigzagLevelOrder(root))


# 98. Validate Binary Search Tree
"""
    algorithm
    1, traverse tree with inorder iterative traversal
    2, keep track of previous value and if it is greater than current value, return False

"""

def isValidBST(root):
    """
        be careful, for inorder iterative traversal, stack will start off with empty list
        otherwise top_root will be spit out twice
    """
    # inorder is best as it spits out all the values in order
    if not root: return True # if no root, return True
    stack = []
    prev = float('-inf')
    while stack or root:
        # after the original root (5) is spit out, there is no value in stack
        # that is why you need root here
        # go all the way down to the bottom
        while root:
            stack.append(root)
            root = root.left

        root = stack.pop()

        # it is always false as inorder traversal collects values in order if it is based off BST
        if root.val <= prev:
            return False
        prev = root.val
        # if root.right exists, in next while loop
        # it will be collected
        root = root.right

    return True

# [1,1]
# [10,5,15,null,null,6,20]

root = Node(5)
root.left = Node(1)
root.right = Node(4)
root.right.left = Node(3)
root.right.right = Node(6)

root1 = Node(1)
root1.left =  Node(1)

root2 = Node(2)
root2.left = Node(1)
root2.right = Node(4)

print(isValidBST(root))
print(isValidBST(root1))
print(isValidBST(root2))


# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/discuss/34555/The-iterative-solution-is-easier-than-you-think!
# 105. Construct Binary Tree from Preorder and Inorder

def buildTree(preorder, inorder):
    """
        algorithm
        1, keep pushing nodes to stack and if flag is false, top of stack will get cur_val (current index of preorder) as a left child
        2, if top of stack and current index of inorder matches,
            pop the top of stack untile they dont match along with incrementing index of inorder
        3, Repeat 1 and 2 until preorder is empty. The key point is that whenever the flag is set, insert a node to the right of flag
            and reset the flag.
    """

    root = Node(preorder[0])
    stack = [root]
    pre = 1
    idx_in = 0
    while pre < len(preorder):

        cur_node = Node(preorder[pre])
        pre += 1
        prev = None
        while stack and stack[-1].val == inorder[idx_in]:
            prev = stack.pop() # pop until it gets the parent
            idx_in += 1

        if prev:
            prev.right = cur_node

        else:
            stack[-1].left = cur_node

        stack.append(cur_node)

    return root

    ####

    """
        algorithm
        1, the first value in preorder is main root. 
            and in inorder list, all the values before first value of preorder belong to left sbutree
            and all the values it belong to right subtree, e,g 9 belongs to left tree, 15,20,7 belongs to right_subtree
        2, create hash map (dictionary) based off inorder list
        3, build binary tree and middle is always the first value of deque of preorder
        3,  Using a HashTable to store the indices is key here. Otherwise you'd quickly end up with a N^2 solution 
            because for each call you'd need to search for the position of preorder[x] inside of inorder (unsorted) array.

    """
    if not preorder and not inorder:
        return None

    #dic = {val:i for i,val in enumerate(inorder)}
    dic = {}
    for i,val in enumerate(inorder):
        if val not in dic:
            dic[val] = i

    def helper(preorder,l,r):

        if l <= r: # l and r dont really matter as preorder value determines middle

            cur_val = preorder.popleft()
            root = Node(cur_val)
            mid = dic[cur_val] # gets next root

            root.left = helper(preorder,l,mid-1) # this one traverse all the values before main node that is why mid-1
            root.right = helper(preorder,mid+1,r)

            return root

    return helper(deque(preorder), 0, len(preorder)-1)

test = [3, 9, 20, 15, 7]
test1 = [9, 3, 15, 20, 7]
test2 = [1,2,3]
test3 = [3,2,1]

print(levelOrder(buildTree(test, test1)))
print(levelOrder(buildTree(test2, test3)))


# 124. Binary Tree Maximum Path Sum

"""
    algorithm
    1, create gloabal variable that holds max path 
    2, iterate each node and keep track of sub tree sum and max path
        because when subtree is bigger than max path, it will update 
    3,
"""
class MAXPATH:
    def maxPathSum(self,root):

        def find_max_path(root):
            # base case
            if not root:
                return 0 # integer should be returned as None and integer cannot be compaired

            left = find_max_path(root.left)
            right = find_max_path(root.right)

            # potential max path could be 3 choices
            # 1, root itself
            # 2, root and left
            # 3, root and right
            # 4, root and left and right
            self.max_path = max(self.max_path, max((left + root.val + right),(left + root.val), (right + root.val), (root.val)))

            # return biggest path out of 3 choices, root itself or root and left or root and right
            # not return the sum of subtree as a whole cuz  A path is a connection between any two nodes, maybe a single node to.
            return max(left + root.val, right + root.val, root.val)

        self.max_path = float("-inf")
        find_max_path(root)
        return self.max_path

s = MAXPATH()
root = Node(10)
root.left = Node(2)
root.left.left  = Node(-20)
root.left.right = Node(1)
root.right  = Node(10)
root.right.right = Node(-25)
root.right.right.left   = Node(3)
root.right.right.right  = Node(4)

root1 = Node(-20)
root1.left = Node(-40)
root1.right = Node(10)
root1.left.left = Node(90)
root1.left.left.left = Node(110)
root1.left.left.right = Node(50)
root1.left.right = Node(1)
root1.right.right = Node(-25)
root1.right.right.left = Node(3)
root1.right.right.right = Node(4)

root2 = Node(2)
root2.left = Node(-1)

root3 = Node(-10)
root3.left = Node(9)
root3.right  = Node(20)
root3.right.left = Node(15)
root3.right.right  = Node(7)

print(s.maxPathSum(root))
print(s.maxPathSum(root1))
print(s.maxPathSum(root2))
print(s.maxPathSum(root3))
