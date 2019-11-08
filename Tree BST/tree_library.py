class Node:
    def __init__(self,val):
        self.val = val
        self.left = self.right = None
        self.next_right = None # for connect level problem


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
    """
        be careful, for inorder iterative traversal, stack will start off with empty list
        otherwise top_root will be spit out twice
    """
    while stack or root:
        while root:
            stack.append(root)
            root = root.left

        popped = stack.pop()
        res.append(popped.val)
        root = popped.right

    return res

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
        cur_level = []
        next_level = []
        ### this cur_level is now Node instance as it just popped out,
        ### however, if it was in list, it is iterable,
        ### so if just for loop over queue, it is deque instance which is iterable
        #cur_level = queue.popleft()
        # queue.popleft() root does not move
        # root = queue.popleft() # this root is inherited # 'Node' object is not iterable
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