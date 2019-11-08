# Leetcode

# 46. Permutations

def permutation(nums):

    """
    algorithm
    ### Key is that how to slice list and what is base case and constraint
    1, Goal, base case will be based on length of arr
    2, once it reaches 0 it will be termianted as there is not valut to loop over
    3, Choice, slice array

"""
    def dfs(nums, sub, res):
        # base case
        if not nums:
            res.append(sub)
            return # backtracking
            # nums[:0] always returns 0 no matter how many elments in the list
            # when i is 0 it just takes first val until there is no val
            # when i is 1 gest first val and last val, [:1] gets first and
            # [i+1:] gets last val
            # and sub gets nums[i] which is 2 when i is 2
            # when i is 2, get first 2 vals
        """
        1, sub = [1] nums = [2,3] i = 0
        2, sub = [1,2] nums = [3] i = 0
        3, sub = [1,2,3] nums = [] i = 0
        4, sub = [1,3] nums = [2] i = 1 # got 3 out of [2,3] and nums[1] is 3
        5, sub = [1,3,2] nums = [] i = 0 
        6, sub = [2] nums = [1,3] i = 1 got 2 out of [1,2,3] nums[1] = 2
        7, sub = [2,1] nums = [3] i = 0 got 1 out of [1,3] nums[0] = 1
        8, sub = [2,1,3] nums = [] i = 0 
        9, sub = [2,3] nums = [1] i = 1 got 3 out of [1,3] nums[1] = 3
        10, sub = [3] nums = [1,2] i = 2 got 3 out of [1,2,3] nums[2] = 3
        11, sub = [3,1] nums = [2] i = 0 
        12, sub = [3,1,2] nums = [] i = 0
        13, sub = [3,2] nums = [1] i =1 got 2 out of [1,2] nums[1] = 2
        14, sub = [3,2,1] nums = [] i = 0
        """
        for i in range(len(nums)):
            dfs(nums[:i]+nums[i+1:], sub+[nums[i]], res)
    res = []
    dfs(nums, [], res)
    return res

meow = [1,2,3]
print(permutation(meow))

# 78. Subsets

def permutation(nums):
    """
    algorithm
    1, Goal, base case will be based on for loop loop
    2, once there in no value in nums, it will be terminated
    3, Choice, slice array, slice i + 1 but sub gets nums[i]
    """
    def dfs(nums, sub, res):
        # base case is for loop
        # as when there is no nums,it will terminate
        """
        1, sub = [] nums = [1,2,3] i = 0 # 0
        2, sub = [1] nums = [2,3] i = 0 # 1
        3, sub = [1,2] nums = [3] i = 0
        4, sub = [1,2,3] nums = [] i = 0
        5, sub = [1,3] nums = [] i = 1 # 1
        6, sub = [2] nums = [3] i = 1 # 0
        7, sub = [2,3] nums = [3] i = 0
        8, sub = [3] nums = [] i = 0
        """
        res.append(sub)
        for i in range(len(nums)):
            dfs(nums[i+1:], sub+[nums[i]], res)
    res = []
    dfs(nums, [], res)
    return res

    # result = [[]]
    # for num in nums:
    #     result += [i + [num] for i in result]
    # return result

meow = [1,2,3]
print(permutation(meow))

# 39. Combination Sum

def combinationSum(candidates, target):

    # optimal solution
    """
    algorithm
    1, sort list, and as soon as current current value is greater than target, break that loop
    2, subtract target with current value eact time and when target is 0, that is goal
    3,
    """
    def dfs(candidates,t,sub,res):

        # goal
        if t == 0:
            res.append(sub)

        for i,val in enumerate(candidates):
            if t >= val:
                dfs(candidates[i:],t-val,sub+[val],res)
            # base case
            else:
                break

    candidates.sort()
    res = []
    dfs(candidates,target,[],res)
    return res

    # my work
    """
    algorithm
    1, Goal: compute all possible solutions, if we reach target, that is goal
        Base case: if current candidate exceeds target, returnn
    """
    def dfs(candidates, t, sub, res):

        # goal
        if t == 0:
            res.append(sub)
            return
        # base case
        if t < 0:
            return
        for i in range(len(candidates)):
            dfs(candidates[i:],t-candidates[i],sub+[candidates[i]],res)

    res = []
    dfs(candidates,target,[],res)
    return res

candidates = [2,3,6,7]
target = 7
candidates1 = [2,3,5]
target1 = 8
print(combinationSum(candidates,target))
print(combinationSum(candidates1,target1))

# 22. Generate Parentheses

def generateParenthesis(n):

    # how to make left and right
    # initiate them as global variable
    """
    algorithm
    1, choice -  open or close
    2, constraints - parenthesis should be closed
    3, goal - all parenthesis closed
    """
    # choice -  open or close
    # constraints - parenthesis should be closed
    # goal - all parethesis closed
    def dfs(Open,Close,res,ans):
        # constraints - if close is more than open, it cannot be closed
        # (()))
        # ()())
        if Open > Close:
            return

        # goal - if all parentheses are closed
        if not Open and not Close:
            res.append(ans)

        # choice open or close
        if Open:
            dfs(Open - 1, Close, res, ans + "(")

        if Close:
            dfs(Open, Close - 1, res, ans + ")")

    res = []
    left,right = n,n
    dfs(left,right,res,"")
    return res

n = 3
print(generateParenthesis(n))

