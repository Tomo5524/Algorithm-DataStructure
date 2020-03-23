# Dynammic Programming

# Leetcode

# 3/21/2020
# 807. Max Increase to Keep City Skyline
def maxIncreaseKeepingSkyline(grid):

# Time Complexity: O(N2)
    if not grid: return 0

    row_maxes = []
    col_maxes = []

    # get max from each row
    for i in range(len(grid)):
        # col_maxes.append(max(grid[i]))
        row_max = float('-inf')
        col_max = float('-inf')
        for j in range(len(grid[i])):
            # get max from each row and colum
            row_max = max(row_max, grid[i][j])
            col_max = max(col_max, grid[j][i])

        row_maxes.append(row_max)
        col_maxes.append(col_max)

    total = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            total += min(row_maxes[i], col_maxes[j]) - grid[i][j]
            # dont need these lines as we look for the increase.
            # if there is no increase possible, we just subtract current grid and min(rowmax or colmax) e.g. grid[1][1], 4 will remain intact
            # biggest = max(row_maxes[i], col_maxes[j])
            # if grid[i][j] < biggest:
            #     total += min(row_maxes[i], col_maxes[j]) - grid[i][j]

    return total

test = [[3,0,8,4],
        [2,4,5,7],
        [9,2,6,3],
        [0,3,1,0]]

test1 = [[0,0,0],[0,0,0],[0,0,0]]

print(maxIncreaseKeepingSkyline(test))
print(maxIncreaseKeepingSkyline(test1))

# 11/18/2019
# 518. Coin Change 2
def change(amount,coins):

    """
        key is to divide whole problem into sub problems
        and think about base case of brute force
        algorithm
        1, find number of combinations that make up current amount with current coin
        2, create dp and find out how many combinations required to make up current amount
            # to make up 0, there is only one combination which is 0
        3,
        """

    # dp
    # Time complexity: O(N×amount), where N is a length of coins array.

    # create dp
    dp = [1] + [0]*amount

    for coin in coins:
        # now we try to find out number of combinations that make up current amount with current coin
        for i in range(coin,amount+1):
            # dp[i-coin] represents how many coins do we need to make up current amount
            dp[i] = dp[i] + dp[i - coin]

    return dp[-1]

    # brute force
    def get_subset(coins,sub,res):
        if sum(sub) == amount:
            res.append(sub)

        if sum(sub) > amount:
            return

        for i in range(len(coins)):
            get_subset(coins[i:],sub+[coins[i]],res)

    res = []
    get_subset(coins,[],res)
    return len(res)

amount = 5
coins = [1, 2, 5]
print(change(amount,coins)) # 4
amount1 = 3
coins1 = [2]
print(change(amount1,coins1)) # 0
amount2 = 10
coins2 = [10]
print(change(amount2,coins2)) # 1
print()


# 11/17/2019
# 64. Minimum Path Sum
def minPathSum(grid):
    # dp
    """
    algorithm
    1, get first row and colum
    2, and dp from 1,1.
    3, 1,0 and 0,1 are already computed so just get minimum of those
    """
    # time complexity: O(mn)
    # space complexity: O(mn)
    dp = [[0 for _ in range(len(grid))] for _ in range(len(grid[0]))]
    dp[0][0] = grid[0][0]
    # get first row
    for i in range(1,len(grid)):
        dp[0][i] = dp[0][i-1] + grid[0][i]

    for j in range(1,len(grid[0])):
        dp[j][0] = dp[j-1][0] + grid[j][0]

    for x in range(1,len(grid)):
        for y in range(1,len(grid[0])):
            dp[x][y] = min(dp[x-1][y],dp[x][y-1]) + grid[x][y]

    return dp[-1][-1]

test = [[1,3,1],
        [1,5,1],
        [4,2,1]] # 7
print(minPathSum(test))

# 11/4/2019
def maximalSquare(matrix):
    """
    key is how to find square,and how to build dp
    algorithm
    1, create dp and visit each cell
    2, if current cell is 1, that will potentially lead global solution
        get minimum value out of upper left, left, and cell above as they are already visited,
        plus current cell makes square
    3,
    """
    # time complexity O(M*N)
    m,n = len(matrix),len(matrix[0])
    # create 2d array,
    # first row and first colum should be 0 to make compairson
    dp = [[0 for _ in range(n+1)] for i in range(m+1)]
    max_area = 0
    for r in range(m):
        for c in range(n):
            if matrix[r][c] == "1":
                # starts off with r+1 and c+1 cuz dp's size
                # get minimum value from upper left, left and the one above
                dp[r+1][c+1] = min(dp[r+1][c],dp[r][c+1],dp[r][c]) + int(matrix[r][c])

            max_area = max(max_area,dp[r+1][c+1])

    return max_area * max_area

test = [["1","0","1","0","0"],
        ["1","0","1","1","1"],
        ["1","1","1","1","1"],
        ["1","0","0","1","0"]] # 4

test1 = [["0","0","0","1","0","1","1","1"],
        ["0","1","1","0","0","1","0","1"],
        ["1","0","1","1","1","1","0","1"],
        ["0","0","0","1","0","0","0","0"],
        ["0","0","1","0","0","0","1","0"],
        ["1","1","1","0","0","1","1","1"],
        ["1","0","0","1","1","0","0","1"],
        ["0","1","0","0","1","1","0","0"],
        ["1","0","0","1","0","0","0","0"]] # 1

print(maximalSquare(test))
print(maximalSquare(test1))
print()

# Solve it with navie recurive function first
# thinking about sub problems means literally treat each value as its own problem
# e,g, jump game, [3,2,1,0,4], step1, can we go to index 0 with 4 (i+1+cur_val) jumps? yes
                             # step2, can we go to index 1 with 2 jumps? yes
                             # step3, can we go to index 2 with

# 10/26/2019
# 416. Partition Equal Subset Sum

def canPartition(nums):

    """
    key is to find target which can only be even sum
    If the sum of all elements is odd, then you cannot have 2 subsets with equal sum. So test this and return False.
    algorithm
    1, if sum is not even, cannot partition array into 2 sub array so false
    2, compute each sum up to sum(nums) // 2 and see if current val in array can make up current sum
        # dp is all about solving sub problems!
    3, if current sum is grater than current val, it will get above cell or above cell + cur_sum-val

    """
    if sum(nums) % 2 != 0:
        return False
    t = sum(nums) // 2
    # +1 cuz if sum is 0, it is equal so should be True
    dp = [[False for _ in range(t+1)] for i in range(len(nums))]
    for i in range(len(nums)):
        val = nums[i]
        for j in range(t+1):
            cur_sum = j
            if j == 0:
                dp[i][j] = True

            elif val == cur_sum:
                # how to handle first row?
                # index 0,2 should be true but how
                dp[i][j] = True

            elif val <= cur_sum:
                # first low gets from bottom, a = i-1, b = j-i
                # dp[i][j] = dp[i-1][j-i] # j = 6 i = 1 so it is just 5 and cur_val = 6 val = 5 wont be true
                # e,g, [23,13,11,7,6,5,5], val is 11 cur_sum 12,
                # if only #dp[i][j] = dp[i - 1][cur_sum - val]
                # it will get false  but should be true as 11 can make up subset
                # and rest will become false as well
                # so it will get true if one of them true
                dp[i][j] = dp[i - 1][cur_sum - val] or dp[i - 1][j]

            else:
                # inherit previous local answer.
                #e,g, val = 11, cur_sum, w/o this else statement,
                # when it gets the cell (3,11), i = 4 and cur_sum - val = 6
                # it will be false as when val = 11, cur_sum 6 - val 11 wont do anything
                dp[i][j] = dp[i - 1][j]

    return dp[-1][-1]

# [[], [1], [5], [1, 5], [11], [1, 11], [5, 11], [1, 5, 11], [5], [1, 5], [5, 5], [1, 5, 5], [11, 5], [1, 11, 5], [5, 11, 5], [1, 5, 11, 5]]
    # Brute Force / memoization
    """
    key is to find target which can only be even sum
    If the sum of all elements is odd, then you cannot have 2 subsets with equal sum. So test this and return False.
    algorithm
    1, dfs and if target becomes 0, return true
    2, more efficient than dp 
    i think it is cuz in this approach, as soon as it finds answer, it returns true
    whereas dp has to go through all the possbile answers 
    """
    def dfs(nums,t):
        # memo is merely to disqualify unnecessary candidates
        # goal
        if t == 0:
            return True

        # if this target is alredy in memo, it wont lead to global solution so just return its value in memo
        if t in memo:
            return False

        # if t is not answer, store it in memo
        memo[t] = False

        # base case 2 as there will be no nums
        for i in range(len(nums)):
            # if t-nums[i] is less than 0, that means subest will not be equal so don't dfs
            if 0 <= t-nums[i]:
                if dfs(nums[i+1:],t-nums[i]):
                    return True

        return False

    if sum(nums) % 2 != 0:
        return False
    t = sum(nums)
    memo = {}
    return dfs(nums,t)

test = [1,5,11,5]
test2 = [1,2,3,5]
test4 = [28,63,95,30,39,16,36,44,37,100,61,73,32,71,100,2,37,60,23,71,53,70,69,82,97,43,16,33,29,5,97,32,29,78,93,59,37,88,89,79,75,9,74,32,81,12,34,13,16,15,16,40,90,70,17,78,54,81,18,92,75,74,59,18,66,62,55,19,2,67,30,25,64,84,25,76,98,59,74,87,5,93,97,68,20,58,55,73,74,97,49,71,42,26,8,87,99,1,16,79]
test3 = [23,13,11,7,6,5,5]
print(canPartition(test))
print(canPartition(test2))
print(canPartition(test3))
print(canPartition(test4))


# 10/25/2019
# Knapsack Problem

"""
Given weights and values of n items,
put these items in a knapsack of capacity W to get the maximum total value in the knapsack.
"""

def knapSack(val, wt, W):

    """
    key is to treat each problem as sub problem!!!
    algorithm
    1, create 2d array, with len(W) width and len(val) height
    2, solve each subproblem using dp
    3, if current capacity is greater or equal than update dp
    4, W in 2d array denotes the amount of weight,
        so we are trying to figure out how much value can current item (W) and current weight (wt) can hold
        each cell denotes the amount of value
    ### when writng down a graph on piece of paper, keep it in mind
    ### as weight goes down, that means we get more options to explore new possible solutions
    ### and fiding it is so easy using previous information stored in each cell.
    ### [0,6, 6, 6, 6, 6]
    ### [0,6,10,16,16,16]
    ### [0,6,10,16,18,22]
    ### if capacity is just 2, answer is 2 as 10 is the maximum.
    ### so with this graph, we are just adding more possiblities and each cell is the answer for every sub problem
    """

    # DP
    # why i-1 when try to find out max? - cuz that is the local answer
    # n = total items
    # m = max weight (max weight constraint)
    # Time Complexity: O(nm) (we will be solving this many subproblems)
    dp = [[0 for _ in range(W+1)] for i in range(len(val))]

    for r in range(len(wt)):
        cur_w = wt[r]
        for cur_capa in range(W+1):
            # every row at first colum is always 0 as 0 items cannot create snackpack
            if r == 0 and cur_capa == 0:
                dp[r][cur_capa] = 0
            # cur_w means max current weight
            elif cur_w <= cur_capa:
                # store value which current weight and current capacity can hold
                dp[r][cur_capa] = max(dp[r-1][cur_capa], val[r] + dp[r-1][cur_capa-cur_w])

            # cur_w is less than cur_capa so it cannot be a possible solution, so just get the value above
            # as that is the current maximum value for current cell
            # e,g, [6,10,12], [1,2,3], cur_weight is 3 and cur_capacity is 1,
            # that means only value with 1 weight can fit in and the answer is 6
            # as weight 1 has 6 values
            else:
                dp[r][cur_capa] = dp[r-1][cur_capa]

    return dp[-1][-1]

    # brute force
    def subset(val,wt,W,sub,res):
        # goal
        if W == 0:
            res.append(sub)
            return
        # base case
        if W < 0:
            return
        for i in range(len(val)):
            subset(val[i+1:],wt[i+1:],W-wt[i],sub+[val[i]],res)

    res = []
    subset(val,wt,W,[],res)
    return sum(max(res,key=sum))

val = [6, 10, 12]
wt = [1, 2, 3]
W = 5
val1 = [1,4,5,7]
wt1 = [1,3,4,5]
W1 = 7
val2 = [6,5,7,3]
wt2 = [5,3,4,2]
W2 = 5

print(knapSack(val, wt, W))
print(knapSack(val1, wt1, W1))
print(knapSack(val2, wt2, W2))
print()

# 1143. Longest Common Subsequence

def longestCommonSubsequence(text1, text2):

    """
    key is it should be sequence
    "ezupkr", "ubmrapg", answer should be 2,
    how to set up dp is really important!!
    algorithm
    1, when setting up dp table, think about naive recursion approach
        and make table based on base case
    2, if words match, increment current cell
    3, if not, update current cell with previous info, [i-1][j] or [i][j-1]
        all the upper and left cells are sub problems to current cell!!
    """

    # dp
    # how to get rid of p?
    # how not to count p for edge case, 'urap','upkr'
    # 0 needed for dp cuz base case is "" string (0 > s)  which returns 0
    # also, when updating, current cell, if array is not n+1 and m+1
    # it will be out of boundary
    n,m = len(text1),len(text2)
    dp =[[0 for _ in range(m+1)] for i in range(n+1)]

    for r in range(n):
        cur_t1 = text1[r]
        for c in range(m):
            cur_t2 = text2[c]

            # cannot do if r == 0 or c == 0: dp[r][c] = 0 as it will mess up index order
            # if words match
            # just increment counter by updating current cell
            # cuz "jmb","bsb" it is just one LSC
            # thus just update current celll
            if cur_t1 == cur_t2:
                # since first row and first colum should be 0, always think current cell plus 1
                # so r=0 c=0 means current cell we are at is 1,1
                ### this is r and c not r+1 and c+1 as current words are r and c so previous info shoud not be affected
                dp[r+1][c+1] = dp[r][c] + 1

            else:
                # if current words dont match, just update current dp
                # eihter right next cell or cell above is already checked so just grab which ever is greater
                dp[r + 1][c + 1] = max(dp[r + 1][c], dp[r][c + 1])

    return dp[-1][-1]

    # brute force / memoization
    # def dfs(t1, t2, i, j):
    #
    #     if (i,j) in memo:
    #         return memo[(i,j)]
    #
    #     if i < 0 or j < 0:
    #         return 0
    #
    #     if t1[i] == t2[j]:
    #         return 1 + dfs(t1, t2, i - 1, j - 1)
    #     else:
    #         # return max(dfs(t1, t2, i - 1, j), dfs(t1, t2, i, j - 1))
    #         result = max(dfs(t1, t2, i - 1, j), dfs(t1, t2, i, j - 1))
    #         memo[(i,j)] = result
    #         return result
    #
    # if not text2 or not text1: return 0
    # memo = {}
    # return dfs(text1, text2, len(text1) - 1, len(text2) - 1)


t1  = "abcde"
t2 = "ace"
t3 = "ezupkr"
t4 = "ubmrapg"
t5 = "bl"
t6 = "yby"
t7 = "oxcpqrsvwf"
t8 = "shmtulqrypy"
a = "upkr"
b = "urap"
c = "bsbm"
d = "jmb" # 1

print("longestCommonSubsequence")
print(longestCommonSubsequence(t1,t2))
print(longestCommonSubsequence(t3,t4))
print(longestCommonSubsequence(t5,t6))
print(longestCommonSubsequence(t7,t8))
print(longestCommonSubsequence(a,b))
print(longestCommonSubsequence(c,d))
print()

# 45. Jump Game II
### Not Completed!!
def jump(nums):

    # base case
    def dfs(nums,start,end):

        if start >= len(nums)-1:
            return 0

        result = float("inf")
        for cur_step in range(len(nums)):
            # if it is qualified or not
            if  start + nums[cur_step]  + cur_step > cur_step:
                result = min(result,dfs(nums,start+cur_step+nums[cur_step],end)+1)

        return result

    return dfs(nums,0,len(nums)-1)

test = [1, 3, 1, 2, 0, 1]
test1 = [1, 2, 1, 0, 0]
test2 = [2, 2, 0, 0]
test3 = [1, 1, 1, 1, 0, 1]
test4 = [2,0]
test5 =[2,0,0]
test6 = [0]
test7 = [2,5,0,0,0,0]
test8 = [3,2,1,0,4]
test9 = [2,3,1,1,4]

# print(jump(test)) # True
# print(jump(test1)) # False
# print(jump(test2)) # True
# print(jump(test3)) # False
# print(jump(test4)) # True
# print(jump(test5)) # True
# print(jump(test6)) # True
# print(jump(test7)) # True
# print(jump(test8)) # False
print(jump(test9)) # True
print()

# 55. Jump Game
def canJump(nums):

    """
    key is to work backwards from last index
    algorithm
    1, target is length - 1, as that is what we want to reach
    2, traverse backwards and we we can jump from target to current step, update target
    3, and if target is 0 at the end, it is reachable from beginning to end
    """

    # initiate target
    t = len(nums) - 1
    for cur_step in range(len(nums)-2,-1,-1):
        # if current step and the number of steps at current index is greater than target,
        # that means you can jump from current location to destination
        # e,g, [2,5,0,0,0,0]
        # index 4, 0 + 4 cannot make it to the end, so target is still 5
        if cur_step + nums[cur_step] >= t:
            # update cur_step
            t = cur_step

    return t == 0

    """
    key is to keep track of how far can it go from current index and furthest we can go so far.
    algorithm
    1, treat each step as a sub problem,
    2, keep track of furthest step we can reach from current step
        not only the step we can reach from current step,but in between as well
        e,g, [2,2,0,0], index 1 gets 2, as furthest step fo far
    3, as soon as we see current index is greater than furthest, break loop
        as cur_step cannot be reached
    4, if furthest step is greater or equal to destination, it is reachable
    """

    # greedy algorithm, Time complexity O(n)
    # Looking from the start and selecting the locally optimum in the hope of reaching global optimum
    furthest_so_far = 0
    for cur_step in range(len(nums)):
        # impossible to reach current step
        # e,g, [3,2,1,0,4]
        #      [3,3,3,3,3], so it is impossible to reach the end
        # e,g, [2,3,1,1,4]
        #      [2,4,4,4,5], it is possible to reach the end
        if cur_step > furthest_so_far:
            break

        furthest_so_far = max(furthest_so_far, cur_step + nums[cur_step])

    return furthest_so_far >= len(nums)-1

test = [1, 3, 1, 2, 0, 1]
test1 = [1, 2, 1, 0, 0]
test2 = [2, 2, 0, 0]
test3 = [1, 1, 1, 1, 0, 1]
test4 = [2,0]
test5 =[2,0,0]
test6 = [0]
test7 = [2,5,0,0,0,0]
test8 = [3,2,1,0,4]
test9 = [2,3,1,1,4]

print(canJump(test)) # True
print(canJump(test1)) # False
print(canJump(test2)) # True
print(canJump(test3)) # False
print(canJump(test4)) # True
print(canJump(test5)) # True
print(canJump(test6)) # True
print(canJump(test7)) # True
print(canJump(test8)) # False
print(canJump(test9)) # True
print()




# 10/24/2019

# 322. Coin Change

def coinChange(coins,amount):


    # bottom-up
    """
    key is to figure out smallest amount of coins to make up current amount
    """
    # each value dp represents amount
    # so to make 0 amount, we dont need any coin
    # in [1,2,5] t = 11 case, e,g, to make 2 amount, we just need 2 so it will be just 1
    # e,g  to maek amount 7, we need 2 coins, and to make 9, we need 3 coins
    dp = [amount+1 for _ in range(amount+1)]
    dp[0] = 0 # basse case, to make 0, we dont need any coins
    for cur_amount in range(1,amount+1):
        for coin in coins:
            # if cur_amount is greater than cur_coin,
            # by using previous answer, figure out smallest amount of coins does it take to make up current amount
            if cur_amount >= coin:
                dp[cur_amount] = min(dp[cur_amount-coin]+1,dp[cur_amount])

    return dp[-1] if dp[-1] != amount + 1 else -1

    # brute force + memoization
    """
    key is backtracking based off each coin
    writing tree of every step of recursion on a piece of paper helps a lot
    algorithm
    """

    def dfs(coins,amount):

        # if already amount in memo, return that value
        if amount in memo:
            return memo[amount]

        if amount == 0:
            return 0

        # each time, get result and looking for minimum that is why use inf
        result = float("inf")
        for coin in coins:
            if coin <= amount:
                result = min(result,dfs(coins,amount-coin)+1)

        memo[amount] = result
        return result

    memo = {}
    ans = dfs(coins,amount)
    return ans if ans != float('inf') else -1

    # brute force
    # my work
    # def dfs(coins, amount, sub, res):
    #
    #     if amount == 0:
    #         res.append(sub)
    #         return
    #     if amount < 0:
    #         return
    #     for i in range(len(coins)):
    #         dfs(coins[i:], amount - coins[i], sub + [coins[i]], res)
    #
    # if not coins: return 0
    # res = []
    # dfs(coins, amount, [], res)
    #
    # return len(min(res, key=len)) if res else -1

test = [1,2,5]
amount = 6
test1 = [2,5,10,1]
amount1 = 27
test2 = [2]
amount2 = 3
test3 = [186,419,83,408]
amount3 = 6249
test4 = [1,2,5]
amount4 = 11

print(coinChange(test,amount))
print(coinChange(test1,amount1))
print(coinChange(test2,amount2))
# print(coinChange(test3,amount3))
print(coinChange(test4,amount4))

# 10/23/2019

# 139. Word Break

def wordBreak(s, wordDict):

    """
    key - how many false do we need? and where to start
    algorithm
    1, initiate a list full of False and mark first one as True
        otherwise, nothing will be checked
    2, each letter will be checked if that leads to a global solution
        unless they are marked as True
        otherwise it will be skipped. and that is why DP is super efficient
    3, if there is a word (that in dictionary), from start letter to current letter,
        mark last letter's index as True
    """

    dp = [False for _ in range(len(s) + 1)]
    # first one has to be true as we need to check from index first
    dp[0] = True
    for start in range(len(s)):
        # if there is a potential answer,
        # find if the word leads to global solution
        if dp[start]:
            for end in range(start, len(s)):
                # be careful which word to check
                # if s[:end+1] in dic: this doest work as it checks the whole string
                # e,g, s[:9] is "leetcode"
                # so make it s[start:end + 1] and it will work
                if s[start:end + 1] in wordDict:
                    dp[end + 1] = True

    return dp[-1]

    # BFS
    """
    algorithm
    Key is to think of this problem as Graph
    find out there is a path from start (0) to destination(len(s))
    and think carefully about index
    1, think of the start index as vertex 
    2, edge is a word that is in dictionary 
    3, if there is a path from start to destination, return True
    """

    # Time Complexity O(n2)
    # q = deque([0])
    # visited = set()
    # des = len(s)
    # while q:
    #     start = q.popleft()
    #     if start == des:
    #         return True
    #     for edge in range(start, len(s) + 1):
    #         # visited needed here
    #         # e,g, cat and cats lead to and which has end 6
    #         # so this vivisted prevents from visiting twice
    #         if edge not in visited:
    #             # if edge exists in dictionary, append it to queue
    #             # and mark it as visited
    #             if s[start:edge] in wordDict:
    #                 q.append(edge)
    #                 visited.add(edge)
    # return False

    # Brute Force
    """
    algorithm
    key is where to slice, and what letter to cut off
    1, dfs on each word and if we get to a point where there is no string,
        return True
    """

    # def dfs(s):
    #
    #     # goal
    #     if not s:
    #         return True
    #
    #     for i in range(len(s)):
    #         # get from beginning to i + 1
    #         # leetcode -> "leet" [0:4], i = 3
    #         cur_word = s[:i+1]
    #         # cut the word off if current word(
    #         if cur_word in wordDict:
    #             # get i to the end
    #             # leetcode -> 'code' [4:] i = 3
    #             if dfs(s[i+1:]):
    #                 return True
    #
    #     return False
    #
    # return dfs(s)


s = "leetcode"
test  = ["leet", "code"] # True
s1 = "cars"
test1 = ["car","ca","rs"] # True
s2 = "cbca"
test2 = ["bc","ca"] # False
s3 = "bb"
test3 = ["a","b","bbb","bbbb"] # a = bbb, a[0:2] = bb so it is True
s4 = "catsandog"
test4 = ["cats", "dog", "sand", "and", "cat"] # False
print()
print(wordBreak(s,test))
print(wordBreak(s1,test1))
print(wordBreak(s2,test2))
print(wordBreak(s3,test3))
print(wordBreak(s4,test4))
print()

# 213. House Robber II
# https://leetcode.com/problems/house-robber-ii/discuss/227366/Thinking-process-from-easy-question-to-harder-question-within-the-same-question-set

"""From HRI, we already have the solution to a non circular houses,
meaning we don't need to worry about the first and the last.
For this problem, the only major thing that is different is that we have to worry about the first and the last.
You want the first, leave the last. You want the last, leave the first.
"""
# 198. House Robber

def HouseRobber(nums):

    # bottom-Up

    """
    key is to think about edge case plus, where for loop starts
    algorithm
    1, initiate 2 houses, first one is the first house and the other one is the richer house
    2, and rob houses iteratively

    """
    if len(nums) == 1: return nums[0]
    if len(nums) == 2: return max(nums)
    h1 = nums[0]
    h2 = max(nums[0], nums[1])
    max_money = 0
    for i in range(2,len(nums)):
        # decides which house to rob
        max_money = max(h1 + nums[i], h2)
        h1 = h2
        h2 = max_money

    return max_money


    # memoization
    """
     algorithm
     1, thanks to naive solution, everything is already laid out
        so just store visited houses
    """
    def GetMoney(nums,house):
        # base case,
        if house >= len(nums):
            return 0

        if house in memo:
            return memo[house]

        # goal - get max_money
        # constraint - decide which house to rob
        result = max(nums[house] + GetMoney(nums,house+2), GetMoney(nums,house+1))
        memo[house] = result
        return result

    memo = {}
    return GetMoney(nums,0)


    # brute force
    """
    when coming up with naive recursive solution, think about recursion literally
    e,g, in this problem, the choice is to rob current house and any house after neighbour (the house right next to it)
        or just rob the next house.
        so literally it is like, max(cur_house + house+2, or house+1)
        dont think about recursion logically as it is too complicated, just read it literally
    """
    # Time complexity O(2^n)
    # def GetMoney(nums,house):
    #     # base case,
    #     if house >= len(nums):
    #         return 0
    #
    #     # goal - get max_money
    #     # constraint - decide which house to rob
    #     return max(nums[house] + GetMoney(nums,house+2), GetMoney(nums,house+1))
    #
    # return GetMoney(nums,0)


test = [1, 2, 3, 1]  # 4
test1 = [2, 7, 9, 3, 1]  # 12
test2 = [1, 2]  # 2
test3 = [2, 1, 1, 2]  # 4
test4 = [0, 0]  # 0
test5 = [1, 1]  # 1
test6 = [3, 1, 2, 5, 4, 2]  # 10
test7 = [3, -1,-1, -1, -1, 2]

print(HouseRobber(test))
print(HouseRobber(test1))
print(HouseRobber(test2))
print(HouseRobber(test3))
print(HouseRobber(test4))
print(HouseRobber(test5))
print(HouseRobber(test6))
print(HouseRobber(test7))
print()



# 300. Longest Increasing Subsequence

def lengthOfLIS(nums):

    # optimal solution
    """
    key is to binary search
    algorithm,
    1, binary search will be based on value at middle index in cache
    2, if cur_val is greater than value at middle index in cache,
        cur_val will be placed in the right bound as it is greater
        so increase left bound
    3, if cur_val is smaller than value at middle index in cache,
        cur_val will be placed in the left bound as it is smaller
        but, cur_val could be placed at middle index in cache
        so not to disqualify cur_val, right bound just gets middle
    """
    size = 0
    cache = [0 for _ in range(len(nums))]
    for cur_val in nums:
        # size denotes right
        left,right = 0,size
        # each element goes into cache at right the index
        while left < right:
            mid = (left+right) // 2
            # if cur_val is greater, cur_val should be placed further left than middle
            if cache[mid] < cur_val:
                left = mid+1
            # if middle value is greater, it should go to either middle or further right
            # could be placed right at middle and that is why right gets mid not mid -1
            # mid is qualified for the local answer
            else:
                right = mid

        cache[left] = cur_val
        size = max(left+1,size)

    return size

    """
    key is to treat each value as a subproblem, meaning is it possible to be ILS from start to end,
    e,g, 10,2,5, 10 is 1, 2 is 1, 5 is 2. treat each value as a sub problem and just add new value to sub problem
    keep track of current LIS using cache
    and where 2 pointers start in each iteration
    find out why 2 for loops going over whole length dont work, for i in range(len(arr)), for j in range(len(arr))
    algorithm - because if there is a smaller value at the end, front value will accumulate these values which is wrong
    e,g, [4,10,4,3,8,9]
    1, create cache with all elements being 1, as one element it self can be ILS
    2, whenever we see greater value than original value (the value current for loop started off with)
    3, and return max value out of cache
    ### return cache[-1] will not work in this case, [1,3,6,7,9,4,10,5,6]
    ###
    # 
    """
    # Time Complexity O(n2)
    # initiate chache
    # it is all 1 cuz one value itself can become ILS
    cache = [1 for _ in range(len(nums))]
    # end denotes the original value on which comparison is based off
    for end in range(len(nums)):

        # for start in range(len(nums)): will not work for [4,10,4,3,8,9]
        # as it will become 4 cuz first 4 will visit each value and gets 2 which is wrong

        # end denotes the values that will be compaired to
        for start in range(end):
            if nums[end] > nums[start]:
                cache[end] = max(cache[end],cache[start]+1)

    return max(cache)

test = [10,9,2,5,3,4]
test1 = [4,10,4,3,8,9]
test2 = [1,3,6,7,9,4,10,5,6]
test3 = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
test4 = [0,2,6,9,11,15]
test5 = [10,9,2,5,3,7,101,18]
print(lengthOfLIS(test))
print(lengthOfLIS(test1))
print(lengthOfLIS(test2))
print(lengthOfLIS(test3))
print(lengthOfLIS(test4))
print(lengthOfLIS(test5))
print()


# 10/22/2019

# 53. Maximum Subarray

def maxSubArray(nums):
    """
    Key is not to disqualify visited value by comparing current value and accumulative sum
    algorithm
    1, iterate each value and if current value is greater than accumulative sum,
        update accumulative sum
    2, and keep tracking of max_sum by always comparing it to accumulative sum
    """
    # edge case
    if not nums: return 0
    if len(nums) == 1: return nums[0]
    cur_max, max_sum = 0, float("-inf")
    for i in range(len(nums)):
        # update current max so far
        # if current value is greater than accumulative sum, update current max
        cur_max = max(cur_max + nums[i], nums[i])
        # update max_sum each iteration
        max_sum = max(max_sum, cur_max)
    return max_sum

    #brute force
    # iterate each value and from current value, find out total just adding each value
    # and compare to biggest so far
    # biggest_sofar = 0
    # for i in range(len(nums)):
    #     total = nums[i]
    #     for j in range(i+1,len(nums)):
    #         total += nums[j]
    #         biggest_sofar = max(biggest_sofar,total)
    #
    # return biggest_sofar


test = [-2,1,-3,4,-1,2,1,-5,4]
print(maxSubArray(test))
print()

# 62. Unique Paths

def uniquePaths(m,n):

    # another memoization
    # https://leetcode.com/problems/unique-paths/discuss/23122/Sharing-my-0ms-java-solution-with-detailed-exploration
    dp = [[1 for i in range(n)] for _ in range(m)]
    # first row should be all 1 as from start there is only path
    # same as left colum
    # that is why it starts off with 1
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[-1][-1]

    # memoization
    """
    algorithm
    key is to return 1 when reaching the destination,
    how to get path from current cell, path = dfs(m+1,n) + dfs(m,n+1)
    1, create memoization
    2, start off with left top
    3, dfs until destination along with storing paths
    # we return paths of first cell, so dont worry about updating destination cell [1]
    """

    def dfs(r,c):

        # constraint
        if r < 0 or r >= m or c < 0 or c >= n:
            return 0

        # if already visited, return paths of current cell
        if (r,c) in memo:
            return memo[(r,c)]

        # if current cell is destination, return 1, [1]
        if (r,c) == dts:
            return 1

        paths = dfs(r+1,c) + dfs(r,c+1)
        memo[(r,c)] = paths
        return paths

    dts = (m-1,n-1)
    memo = {}
    dfs(0,0)
    return memo[(0,0)]
    #return dfs(0,0)

    # def path(r, c):
    #     # base case
    #     if r < 0 or r >= m or c < 0 or c >= n:
    #         return 0
    #
    #     # goal
    #     if (r, c) == (m - 1, n - 1):
    #         return 1
    #
    #     # this gets how many paths are there that lead to destination from current cell
    #     # as if it reaches destination, it will return 1
    #     # choice
    #     return path(r + 1, c) + path(r, c + 1)
    #
    #     # it just gets how many cells from start to destination
    #     # return 1 + max(path(r + 1, c), path(r, c + 1))
    #
    # return path(0, 0)

print(uniquePaths(3,2))
print(uniquePaths(7,3))
print ()

# 70. Climbing Stairs

def climbStairs(n):

    """
    algorithm
    key is how to compute each step with recursion,
        naive recursive solution is the key to optimal solution
        and find out what should the be base case,
        in memoization, when to put value into dictionary
        in bottom-up, find out where to start, (val1=1,val2=2)
        ## dont forget base case!!!
    """

    # bottom-up
    # how to go to step 0 to step 0, it is already on top of stiars so it took 1 step
    # representation -> [1,1,2,3,5]
    """
    algorithm
    1, starts off with 1, initiate two variables. 
        one keeps current sum, the other one kees previous sum
    2, and loop from 2 (we are already at step 2) to end of stairs
    3, return current sum
    """

    # edge case
    if n == 1: return 1
    # clibmed 1 stair
    prev_sum = 1
    # clibmed 2 stairs
    cur_sum = 2
    # for loop starts at 2 because of index
    # [1,2, ,] # the empty spot is where loop starts out which is index 2
    for i in range(2,n):
        sum_of_current_stairs = prev_sum + cur_sum
        prev_sum = cur_sum
        cur_sum = sum_of_current_stairs

    return cur_sum

    # memoization
    """
    algorithm, 
    1, create dictionary
    2, and after computation each step, store result to dictionary
    3, and return dictionary value each step
    """

    # def memo_fib(n):
    #     # base case
    #     if n < 0:
    #         return 0
    #     if n == 0:
    #         return 1
    #     # here is the magic of memoization
    #     if n in memo:
    #         return memo[n]
    #     # get result of current step
    #     result = memo_fib(n-2) + memo_fib(n-1)
    #     # store result
    #     memo[n] = result
    #     # and return result
    #     return result
    #
    # memo = {}
    # return memo_fib(n)


    # brute force
    # base case
    # if n < 0:
    #     return 0
    # # goal
    # if n == 0:
    #     return 1
    # # constraint
    # return climbStairs(n-2) + climbStairs(n-1)

print(climbStairs(1)) # 1
print(climbStairs(2)) # 2
print(climbStairs(3)) # 3
print(climbStairs(4)) # 5