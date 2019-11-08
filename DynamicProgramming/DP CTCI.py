# DP CTCI

# Round 2

# 10/22/2019


def stairs(n):
    # think of this problem as climbing stairs down

    #bottom-up
    val1 = 1
    val2 = 2
    val3 = 4

    for i in range(3,n):
        cur_sum = val1 + val2 + val3
        val1 = val2
        val2 = val3
        val3 = cur_sum

    return val3

    # memoization

    def memo_stairs(n):
        if n in memo:
            return memo[n]

        if n < 0:
            return 0
        if n == 0:
            return 1

        result = memo_stairs(n-3) + memo_stairs(n-2) + memo_stairs(n-1)

        memo[n] = result
        return memo[n]

    memo = {}
    return memo_stairs(n)

    # brute force
    # time complexity, O(N^2)
    # when writing tree on a piece of paper, when it reaches 0, finised climbing staris
    # so counting how many 0 will give the answer
#     if n < 0:
#         return 0
#
#     if n == 0:
#         return 1
#
#     return stairs(n-3) + stairs(n-2) + stairs(n-1)
#
print(stairs(4))





# 8.11 Coins

# how many ways are there to make 

def Coins(s,n):

    def dfs(s,n,sub,res):

        # base case
        if sum(sub) > n:
            return
        # goal 
        if sum(sub) == n:
            res.append(sub)
            return

        for i in range(len(s)):
            # choice
            dfs(s[i:],n,sub+[s[i]],res)
        
    
    s.sort()
    res = []
    sub =[]
    dfs(s,n,sub,res)
    return res
    #return res

s = [25,10,5,1]
#s1 = [1,2,5]


print(Coins(s,10))
#print(Coins(s,11))


# 322. Coin Change

def Coins(coins,n):
    
    def dfs(coins,amount,sub,res):
    
    # base case
        if sum(sub) > amount:
            return
        # goal 
        if sum(sub) == amount:
            res.append(sub)
            return

        for i in range(len(coins)):
            # choice
            dfs(coins[i:],amount,sub+[coins[i]],res)

    if not coins: return -1
    coins.sort()
    res = []
    sub =[]
    dfs(coins,n,sub,res)
    return -1 if len(res) == 0 else len(min(res,key=len))
    #return res

s1 = [1,2,5]

print(Coins(s1,11))


# 8.10 P    Paint Fill

def PaintFill(grid,nc,r,c):
    # grid[Row][Col] is tartget
    # everytime you run into target, paint it with new color

    def dfs(grid,nc,r,c,t):

        # base case
        # find false
        # if one of these constraitns gets violated
        if r < 0 or len(grid) <= r or c < 0 or len(grid[0]) <= c:
            return
        
        # this is for when nc and t are the same
        # and as it will look for all the surroundings for each cell,
        # if t and nc are the same, never ends
        # grid[r][c] == nc this will make sure if t and nc are the same
        # it will bail out
        
        # if it is not target, dont need to paint
        if grid[r][c] != t:
            return False
        # edge case when new color is same as target
        if grid[r][c] == nc:
            return False
        
        grid[r][c] = nc

        dfs(grid,nc,r+1,c,t)     
        dfs(grid,nc,r-1,c,t)    
        dfs (grid,nc,r,c+1,t)   
        dfs(grid,nc,r,c-1,t)
       
    t = grid[r][c]
    dfs(grid,nc,r,c,t)
    return grid

photo = [[1, 1, 1],
          [1, 0, 1],
          [1, 0, 0]]

Row = 1
Col = 1
newColor = 5

##photo = [[0, 0, 0],
##          [0, 1, 1]] #1
##Row = 1
##Col = 1
##newColor = 1
print(PaintFill(photo,newColor,Row,Col))


screen = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 1],
            [1, 2, 2, 2, 2, 0, 1, 0],
            [1, 1, 1, 2, 2, 0, 1, 0],
            [1, 1, 1, 2, 2, 2, 2, 0],
            [1, 1, 1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1, 2, 2, 1],
        ]

Row = 4
Col = 4
newColor = 3
print(PaintFill(screen,newColor,Row,Col))

# 8,9 Parens

def Parenthes(n):

    def dfs(Open,Close,ans,res):

        if Open > Close:
            return

        if not Open and not Close: res.append(ans)

        if Open: dfs(Open-1,Close,ans+'(',res)

        if Close: dfs(Open,Close-1,ans+')',res)

    l = r = n
    res = []
    ans = ''
    dfs(l,r,ans,res)
    return res

print(Parenthes(3))

# 8.8 Permutations with Dups

def PermutationswithDups(s):

    def build(s,ans,res):

        # brute force without Force
        # s will change 
        if not s:
            res.add(tuple(ans))
            #res.append(ans)
        for i in range(len(s)):
            
            build(s[:i]+s[i+1:],ans+[s[i]],res)
    
    res = set()
    #res = []
    ans = []
    build(s,ans,res)
    return [list(i) for i in res]


s = [1,1,2]
print(PermutationswithDups(s))

# 8.7 Permutations without Dups

def PermutationswithoutDups(s):

    def build(s,ans,res):

        # s will change 
        if not s:
            #res.add(ans)
            res.append(ans)
        for i in range(len(s)):
            
            build(s[:i]+s[i+1:],ans+s[i],res)
    
    #res = set()
    res = []
    ans = ''
    build(s,ans,res)
    return res


s = 'abc'
print(PermutationswithoutDups(s))

# 8.6 Towers of Hanoi

def hanoi(towers, n, start, temp, end):
    #The number of moves necessary to move a tower with n disks can be calculated as: 2n - 1
    
    """ Move the first disk from A to C
        Move the first disk from A to B
        Move the first disk from C to B
        Move the first disk from A to C
        Move the first disk from B to A
        Move the first disk from B to C
        Move the first disk from A to C

    """
    # base case
    if n > 0:
        # move top n - 1 disks from origin to end, using destination as a end. */
        hanoi(towers, n-1, start, end, temp)
        if towers[start]:
            disk = towers[start].pop()
            towers[end].append(disk)
        # move top n - 1 disks from buffer to destination, using origin as a buffer.
        hanoi(towers, n-1, temp, start, end)
        
towers = [[3,2,1], [], []]
hanoi(towers, len(towers[0]), 0, 1, 2)
#print(towers)

##def hanoi(towers, n, start, temp, end):
##
##    
##    
##    towers = [[3,2,1], [], []]
##print(hanoi(towers, len(towers[0]), 0, 1, 2))

# 8.5 Recursive Multiply

def RecursiveMultiply(one,two):

    def build(one,two,i):

        # base case, goal
        if two == 1:
            return one

        #build(one, one + two,i-1)
        # does not return anything so cannnot get two
        return build(one+i, two-1,i)

    i = one
    return build(one,two,i)

#print(RecursiveMultiply(17,1)) 

# 8.4 Power Set

def Powerset(s):

    def buildSubset(s,sub,res):

        res.append(sub)
        for i in range(len(s)):
            # after sub is 123, s is 3 and i will be 1, however,
            # s is just len 1 and in order for i to be 1, length has to be 2
            # so loop will move on to the next one
            buildSubset(s[i+1:],sub+[s[i]],res)

    res= []
    sub = []
    buildSubset(s,sub,res)
    return res


meow = [1,2,3]
#print(Powerset(meow))


# 8.3 Magic Index

#def MagicIndex(n):
def find_magic_index(n,t):

    # Given a sorted array of distinct integers
    # how to make it O(n) to O(1)
    # brute force
##    for i in n:
##        if arr[i] == i:
##            return i
##
##    return 'King Kong aint got shit on me"


    # When we look at the middle element A [ 5] = 3,
    # we know that the magic index must be on the right side,
    # sinceA[mid] < mid.    
    def binary(n,l,r):
        # base case
        if l <= r:
            mid = (l+r) // 2
            if n[mid] == mid:
                return mid
            # since it is sorted, log(nOn)
            if n[mid] < mid:
                return binary(n,mid+1,r)
            else:
                return binary(n,l,mid-1)
                
    l,r = 0,len(n)-1
    return binary(n,l,r)

##print(find_magic_index([0,2,3,4], 0))
##print(find_magic_index([-1,0,1,3], 3))
##print(find_magic_index([-2,1,4,5,6,7], 1))
##print(find_magic_index([-2,2,4,5,6,7], False))
#print(find_magic_index([0,0,3,3], 3))

def find_magic_index_not_distinct(n):

    
    def binary(n,l,r):
        # base case
        if l <= r:
            mid = (l+r) // 2
            if n[mid] == mid:
                return mid
            else:
                return binary(n,mid+1,r) or binary(n,l,mid-1)
                
    l,r = 0,len(n)-1
    return binary(n,l,r)

meow = [-10,-5,2,2,2,3,4,7,9,12,13]
##print(find_magic_index_not_distinct(meow))
##print(find_magic_index_not_distinct([0,0,0,0]))
##print(find_magic_index_not_distinct([0,0,3,3]))

# dont even bother
# 8.2
# Robot in a Grid

# 8.1 
# Triple Step

def BUTripleStep(n,x):

    # bottom up , space expensive
    
    if n == 0: return 1 # # alredy on top of stairs
    dic = {}
    dic[0] = 1
    # starts from 1 as 0 is already done
    for cur_stair in range(1,n+1): # if not n+1, last index is always 0
        total = 0
        for steps in x :
            ###
            # f(n) = f(n-1) + f(n-2) + f(n-3)
            # according to input
            # f(n) = f(n-1) + f(n-3) + f(n-5)
            # you can easily look up n-k by dictionary
            # so if you have dic ready, it is just easy math
            # put in the dic[cur_stair] the amount of steps of getting on the floor (0)
            if cur_stair - steps >= 0:
                total += dic[cur_stair - steps]
        dic[cur_stair] = total
    
    return dic[n]

x = [1,3,5]
# f(n) = f(n-1) + f(n-3) + f(n-5)
# {0: 1, 1: 1, 2: 1, 3: 2, 4: 3}
#ans = 3
n = 4
print(BUTripleStep(n,x))


x = [1,2,3]
# f(n) = f(n-1) + f(n-2) + f(n-3)
# {0: 1, 1: 1, 2: 2, 3: 4, 4: 7}
# ans = 7
n = 4
print(BUTripleStep(n,x))

def TripleStep(n):
    
    # brute force
    
    if  0 > n:
        return 0

    elif 0 == n:
        return 1

    else:
        return TripleStep(n-1) + TripleStep(n-3) + TripleStep(n-5)
        return TripleStep(n-1) + TripleStep(n-2) + TripleStep(n-3)
    
#print(TripleStep(4))



