# Dynamic Programming

# first come up with naive recursion solution
# if there is subarray problem, apply dp
# memo -> bottomup O(n) space -> bottomup O(1) space


# 322. Coin Change


# time complexity (O*coin)
def coinChange(coins,amount):
    dp = [float('inf') for i in range(amount+1)]
    dp[0] = 0
    for i in range(1, amount+1):
        for coin in coins:
            # figuring out smallest amount of coins 
            if i - coin >= 0:
                # dp[i-coin] keeps track of smallest coin so far
                dp[i] = min(dp[i], dp[i-coin] + 1)
    
    return dp[-1] if dp[-1] != float('inf') else -1
    
meow = [1,2,5]
s = 11
meow1 = [1,2,5]
s1 = 10
meow2 = [1,2,3]
s2 = 5
print(coinChange(meow,s))
print(coinChange(meow1,s1))


# Brute Force
def cc(coin,s):
    res = []
    sub = []
    helper(coin,s,sub,res)
    return min(res,key=len)

def helper(coin,s,sub,res):
    if sum(sub) > s: return
    if sum(sub) == s:
        res.append(sub)
        return

    for i in range(len(coin)):
        helper(coin,s,sub+[coin[i]],res)

coin = [1,2,5]
s = 11
print(cc(coin,s))
        

def wordBreak(s,wordDict):

    # dp

    dp = [False for _ in range(len(s)+1)]
    # dp[i - word] should be True when found 
    dp[0] = True
    for i in range(1,len(s)+1):
        for word in wordDict:
            # dp[i-len(word)] guarantees that dp is true right before the word you're looking for,
            # and s[i-len(w):i]==w just means that you've found the word in s.
            # this works for edge case s = "cbca",  wdic = ["bc","ca"]
            # ca is true but since dp[i-len(word)] is not true
            # ca becomes false as answer should be false
            if dp[i - len(word)] == True and \
            s[i - len(word):i] == word:
            # the beginning of current word is i - len(word)
            # the end of current word is i
            # so s[i - len(word):i] should work
            # [i - len(word):i] gets the right word
            # e.g. cars, and word = ca, s[i-2:i]
                dp[i] = True

    return dp[-1]
    
#Brute Force(backtracking)

##def wordBreak(s,wordDict):

##    def backtrack(rem):
##
##        # base case, goal
##        if not rem:
##            return True
##
##        for i in range(len(rem)):
##            # check from the beginning,
##            # if no match, move on to next char
##            if rem[:i+1] in wordDict:
##                # if rem in worddict, split the remaining and matched one
##                if backtrack(rem[i+1:]):
##                    return True
##
##        return False
##
##    # no need to declare wordDict as it is already declared as argument
##    return backtrack(s)

# my work 
##    if not s: return False
##    for i in wordDict:
##        if i in s:
##            # string is immutable, so need to create a new string
##            # and assign it
##            s = s.replace(i,'')
##                    
##    return True if not s else False


s = "leetcode"
meow  = ["leet", "code"]
s1 = "cars"
meow1 = ["car","ca","rs"]
s2 = "cbca"
meow2 = ["bc","ca"]
s3 = "bb"
meow3 = ["a","b","bbb","bbbb"] # a = bbb, a[0:2] = bb so it is True
print(wordBreak(s,meow))
print(wordBreak(s1,meow1))
print(wordBreak(s2,meow2))
print(wordBreak(s3,meow3))


# Knapsack Problem

"""
Given weights and values of n items,
put these items in a knapsack of capacity W to get the maximum total value in the knapsack.
"""

def knapSack(W, wt, val, n):
    # DP
    dp = [[0 for i in range(W+1)] for i in range(len(val)+1)]
    for row in range(len(val)+1):

        for col in range(W+1):

            # if there is no val or weight, should be 0
            # also it corresponds to val+1 and W+1
            # other wise it will be out of boundry 
            if row == 0 or col == 0: 
                dp[row][col] = 0

            # wt[row-1] (current weight) has not reached col (corresponding weight)
            # yet so there is no posibility
            elif wt[row-1] <= col:
                dp[row][col] = max(val[row-1] + dp[row-1][col-wt[row-1]], dp[row-1][col])

            # if current weight is not greater than 
            else:
                dp[row][col] = dp[row-1][col]

    return dp[row][col]
   
val = [6, 10, 12] 
wt = [1, 2, 3] 
W = 5
n = len(val)             
print(knapSack(W, wt, val, n))



# brute force
def napsack(val,weight,capa):

    res = []
    sub = []
    
    pack(val,weight,0,capa,sub,res)
    return max(res)

def pack(val,w,wcnt,capa,sub,res):

    # base case
    if wcnt > capa: return

    # goal
    if wcnt == capa:
        res.append(sub)
        return

    for i in range(len(val)):
        pack(val[i+1:],w[i+1:],wcnt+w[i],capa,sub+[val[i]],res)


val = [60, 100, 120] 
wt = [10, 20, 30] 
W = 50
print(napsack(val, wt, W)) 







# my work
##def dp(val,weight,capa):
##    max_weight = 0
##    cur_w = weight[0]
##    cur_m = val[0]
##    for i in range(1,len(val)):
##        if (weight[i] + cur_w) <= capa:
##            cur_w = min(weight[i], cur_w + weight[i-1])
##            cur_m += val[i]
##
##        max_weight = max(max_weight, cur_m)
##
##    return max_weight
        



val = [60, 100, 120]
wt = [10, 20, 30] 
#wt = [5, 10, 30] 
W = 50
#print(dp(val, wt, W))

# 122. Best Time to Buy and Sell Stock II

def maxProfit(prices):

    # Design an algorithm to find the maximum profit.
    # You may complete as many transactions as you like
    # You may not engage in multiple transactions at the same time
    # you must sell the stock before you buy again.
    
    
    if len(prices) == 0 or len(prices) == 1: return 0

    profit = 0
    for i in range(1,len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]

    return profit
        

meow = [7,1,5,3,6,4]
meow1 = [7, 2, 3, 6, 7, 6, 7]
print(maxProfit(meow))
print(maxProfit(meow1))


# Brute Force 

def maxprofit(prices):
    return calculate(prices,0)

def calculate(prices,s):
    if s >= 4:
        return 0
    max_profit = 0
    for start in range(s,len(prices)):
        cur_max = 0
        for j in range(start+1,len(prices)):
            if prices[start] < prices[j]:
                profit = calculate(prices,j+1) + prices[j] - prices[start]
                if profit > cur_max:
                    cur_max = profit

        if cur_max > max_profit:
            max_profit = cur_max

    return max_profit


meow = [7,1,5,3,6,4]
print(maxprofit(meow))

# Unique Path

def UniquePaths(m,n):

    
    dp = [1 for _ in range(m)]
    # should double for loop as in m = 4 n = 3 case, dp[3] will remain intact
    #for i in range(1,n):
    for i in range(1,n):
        for j in range(1,m):
            dp[j] += dp[j-1]
    return dp[-1]

print(UniquePaths(3,7))
print(UniquePaths(4,3))

           

# 276. Paint Fence

# Return the total number of ways you can paint the fence.

def paintFence(n,k):

    if n == 0:
         # no color
        return 0
    if n == 1:
        return k
    # if we choose to paint the same color 
    # if k = 3, you have 3 ways to paint as only one post is painted so far
    # rr, bb, yy
    same = k 
    # if we choose to paint diffrent cololr than previous
    # k = k*(k-1)
    # rb,by,yr,br,yb,ry
    dif = k*(k-1)
    for i in range(3,n+1):
        same = same*(k-1)
        dif = (same+dif) * (k-1)

    return same+dif

print(paintFence(2,3))

    

# leetcode 53. Maximum Subarray

def maxSubArray(arr):

    # brute force
##    max_sum = 0
##    for i in range(len(arr)):
##        total = arr[i]
##        for j in range(i):
##            total +=arr[j]
##            max_sum = max(max_sum,total)
##    return max_sum
    
    if not arr: return 0
    if len(arr) == 1: return arr[0]
    cur_max = max_sum = arr[0]
    for i in range(1,len(arr)):
        cur_max = max(cur_max + arr[i], arr[i])
        max_sum = max(max_sum,cur_max)
    return max_sum

meow = [-2,1,-3,4,-1,2,1,-5,4]
meow1 = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
##print(maxSubArray(meow))
##print(maxSubArray(meow1))


def ILS(arr):

# [10,9,2,5,3,7,101,18]
# 4, [2,3,7,101]
# [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
# 6, [0,2,6,9,11,15]

    
    # dp
    cache = [1 for i in range(len(arr))]
    for start in range(len(arr)):
        
        for end in range(i):
            # if key (arr[i]) is grater than arr[j], include arr[j] + 1
            # keep track of each ILS
            if arr[start] > arr[end]:
                cache[i] = max(cache[start],cache[end]+1)
            

    return max(cache)


##def reutrnsILS(arr):
##
##    # my brute force
##    return ILS(arr,float('-inf'),0)
##
##def ILS(arr,prev,curPos):
##
##    # base case
##    if curPos == len(arr): return 0
##
##    taken = 0
##    if arr[curPos] > prev:
##        
##        # if current val is greater than prev, update prev
##        taken = 1 + ILS(arr,arr[curPos],curPos+1)
##        
##    # if prev is bigger, let it be
##    # find out the length of LIS possible by not including the current element in the LIS.
##    # find out LIS from curPos+1 so not including curPos
##    # if curpos is 3, compare pre(arr[2]) and cp+1(arr[4])
##    # e.g. [3,2,4,1,5]
##    notTaken = ILS(arr,prev,curPos+1)
##
##    return max(taken,notTaken)





    # ok brute force my work
##    max_len = 0
##    for i in range(1,len(arr)):
##        key = arr[i]
##        # starts off with 1 cuz tryna figure out length which starts off with 1
##        cur_max = 1
##        for j in range(i,len(arr)):
##
##            if arr[j] > key:
##                cur_max +=1
##                key = arr[j]
##
##            max_len = max(max_len,cur_max)
##
##    return max_len

    # legit brute force


meow = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
# 6, [0,2,6,9,11,15]
meow1 = [10,9,2,5,3,7,101,18]
# 4, [2,3,7,101]
print(ILS(meow))
print(ILS(meow1))


def HouseRobber(arr):

     # brute force
    def subset(arr,cur_house):

        #for each house in the neighborhood, you are deciding between starting from it
        # and skipping the current house and making the decision again at the next houes

        if cur_house >= len(arr):
            return 0
         
        return max(arr[cur_house] + subset(arr,cur_house+2), subset(arr,cur_house+1))
       
    return subset(arr,0)

# the choice is whether or not you should steal from the current house,
    # given that you could instead have stolen from the *previous* house.

    # Bottom Up
    
    if not arr: return 0
    if len(arr) == 1: return arr[0]

    val1 = arr[0]
    # at this point, we know lengh is more than 2
    val2 = max(arr[0],arr[1])
    if len(arr) == 2: return val2
    max_sum = 0

    # first 2 values are already checked so starts with 2
    for i in range(2,len(arr)):
        
        # robbery of current house + loot from houses before the previous(i-2) 
        # loot from the previous house robbery and any loot captured before that

        # if decdiding to rob current house
        # she can't rob previous i-1 house
        # but can safely proceed to the one before previous i-2 and
        # gets all cumulative loot that follows.

        # if decdiding not to rob current house
         # the robber gets all the possible loot
         # from robbery of i-1 and all the following buildings.
        max_sum = max(arr[i] + val1, val2)
        # update current sum without current house
        val1 = val2
        val2 = max_sum

    return max_sum

    # DP, space O(n), Memoizatiion
##    dp = [0 for i in range(len(arr))]
##    dp[0] = arr[0]
##    dp[1] = max(arr[0],arr[1])
##    for i in range(2,len(arr)):
##        dp[i] = max(arr[i] + dp[i-2], dp[i-1])
##
##    return dp[-1]

meow = [1,2,3,1] # 4
meow1 = [2,7,9,3,1] # 12
meow2 = [1,2] # 2
meow3 = [2,1,1,2] # 4
meow4 = [0,0] # 0
meow5 = [1,1] # 1
meow6 = [3,1,2,5,4,2] # 10

##print(HouseRobber(meow))
##print(HouseRobber(meow1))
##print(HouseRobber(meow2))
##print(HouseRobber(meow3))
##print(HouseRobber(meow4))
##print(HouseRobber(meow5))
##print(HouseRobber(meow6))

def fib(n):
    
    if n == 1 or n == 2:
        return 1

    else:
        return fib(n-1) + fib(n-2)

#print(fib(5))

def fib_2(n,memo):
    if n in memo:
        return memo[n]

    if n == 1 or n == 2:
        result = 1

    else:
        result = fib_2(n-1,memo) + fib_2(n-2,memo)

    memo[n] = result

    return result

def memoFib(n):
    
    memo = {} # range is plus 1 cuz starts with 1
    return fib_2(n,memo)

#print(memoFib(5))

def bottomUp(n):

    # bottom up
    a,b = 1,1
    for _ in range(n-1):
        a,b = b,a+b

    return a

#print(bottomUp(5))

def fib_bottom_up(n):

    if n == 1 or n == 2:
        return 1

    bottom_up = [0] * (n+1)
    bottom_up[1] = 1
    bottom_up[2] = 1 # 2 will be 1 too 
    for i in range(3, n+1):
        bottom_up[i] = bottom_up[i-1] + bottom_up[i-2]

    return bottom_up[n]
    

#print(fib_bottom_up(5))



##def longest_increasing_subsequence(arr):
##    if not arr:
##        return 0
##    if len(arr) == 1:
##        return 1
##
##    max_ending_here = 0
##    for i in range(len(arr)):
##        ending_at_i = longest_increasing_subsequence(arr[:i])
##        if arr[-1] > arr[i - 1] and ending_at_i + 1 > max_ending_here:
##            max_ending_here = ending_at_i + 1
##    return max_ending_here



##
##def a(arr):
##    
##    if not arr:
##        return 0
##    cache = [1] * len(arr)
##    for i in range(1, len(arr)):
##        for j in range(i):
##            if arr[j] < arr[i]:
##                cache[i] = max(cache[i], cache[j] + 1)
##    return max(cache)
##
##meow = [10,9,2,5,3,7,101,18]
##print(a(meow))
    
