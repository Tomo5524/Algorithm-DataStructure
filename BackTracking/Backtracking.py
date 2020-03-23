# Backtracking

# River SIzes # https://www.algoexpert.io/questions/River%20Sizes
from Algorithm_DataStructure.Soultions_Riverproblem import TestProgram
# how to import another file 

"""given a two-dimensional array,
    each 0 represents land and each 1 represents part of a river
    A river consists of any number of 1s that are eihter horizontally or vertically adjacent
    but not diagonally adjacent)
    return each total number of connected rivers
    
"""

#class Soultions_Riverproblem:
def riverSizes(matrix):
    # Write your code here.
        
    def dfs(matrix,r,c):
            if r < 0 or r >= len(matrix) or c < 0 or c >= len(matrix[0]) or matrix[r][c] != 1:
                    return 0
            
            matrix[r][c] = -1  # mark it as visited
            
            up = dfs(matrix,r-1,c)
            down = dfs(matrix,r+1,c)
            left = dfs(matrix,r,c-1)
            right = dfs(matrix,r,c+1)
            
            return up + down + left + right + 1
            
    res = []
    for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if matrix[row][col] == 1:
                    num_rivers = dfs(matrix,row,col)
                    if num_rivers: # if it is greater than 0
                        res.append(num_rivers)
      
    return res

testInput = [[0]] # []
testInput1 = [[1]] # [1]
testInput2 = [[1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0]] # expected = [1, 2, 3]
testInput3 = [
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
                        ] # [1, 1, 2, 3]
testInput4 = [
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
        ] # expected = [1, 2, 2, 2, 5]

testInput5 = [
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        ] # expected = [1, 1, 2, 2, 5, 21]

print(riverSizes(testInput))
print(riverSizes(testInput1))
print(riverSizes(testInput2))
print(riverSizes(testInput3))
print(riverSizes(testInput4))
print(sorted(riverSizes(testInput5))) # sort dont matter tho hahaha

def MaxAreaofIsland(grid):
    cnt = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                cnt = max(cnt,dfs(grid,i,j))

    return cnt

def dfs(grid,r,c):

    # base case
    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] != 1:
        return 0
          
    grid[r][c] = "*" # without this it will cause loop not to termintate    
    
    down = dfs(grid,r+1,c)
    up = dfs(grid,r-1,c)
    right = dfs(grid,r,c+1)
    left = dfs(grid,r,c-1)

    return up + down + right + left + 1 # # when 1 forund it will return accumulative + 1

[[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 0, 0, 0],
 [0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 2, 0, 0, 2, 5, 0, 0, 1, 0, 5, 0, 0],
 [0, 3, 0, 0, 3, 4, 0, 0, 2, 3, 4, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 4, 5, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0]]

test = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,1,1,0,1,0,0,0,0,0,0,0,0],
        [0,1,0,0,1,1,0,0,1,0,1,0,0],
        [0,1,0,0,1,1,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,1,1,0,0,0,0]]

print("lkjfsjdf")
print(MaxAreaofIsland(test))

# Ball BFS

from collections import deque

# how to keep track of it

def ball(maze,start,des):

    #visited = [[None for i in range(len(maze[0]))] for i in range(len(maze))]
    visited = set()

    dirs = [(0,1),(0,-1),(-1,0),(1,0)]
    #dirs = [(-1,0),(1,0),(0,1),(0,-1)]
    queue = deque([])
    queue.append(start)
    
    #visited.add(start) # TypeError: unhashable type: 'list'
    start = tuple(start)
    visited.add(start) 
    while queue:
        cur_cell = queue.popleft()
        # goal
        if cur_cell[0] == des[0] and cur_cell[1] == des[1]:
            #print(visited)
            return True

        for neighbor in dirs:
            row = cur_cell[0] + neighbor[0]
            col = cur_cell[1] + neighbor[1]
            # this will go on until it gets out of boundry or encouter 1
            # constraitns
            while row >= 0 and col >= 0 and row < len(maze) and col < len(maze[0]) and maze[row][col] == 0 and (row,col) not in visited:
                # if it satisfies the condition, move on according to neighbor
                # choice
                row += neighbor[0]
                col += neighbor[1]

            # be careful, 2 - -1 = 3
            # x - neighbor[0] cuz when encoutering obstacles,
            # cannot mark current cell (obstacle) as it is already visited or obstacle  so subtract
            # e,g,when (1,3) and moves on to (2,3) it is an obstacle
            # cannot mark it as True
            # e,g,when (0,3) and moves on to (0,2) it is an obstacle
            # so mark (0,3) = 2 - -1
            # not current cell but the very pervious cell that satisfied condition
            # that way subtract nei and current cell to get previous cell
            
            if (row - neighbor[0],col - neighbor[1]) not in visited:

                if [row-neighbor[0],col-neighbor[1]] == des:
                        return True
                # append visited safe places
                queue.append([row - neighbor[0], col - neighbor[1]])
                # queue.append([r - nei[0], c - nei[1]]) is wrong
                # [2,1] == (2,1) is False

                #visited[row - neighbor[0]][col - neighbor[1]] = True
                visited.add((row - neighbor[0],col - neighbor[1]))
                

    return False

start = [0, 4]
des = [2, 1]

maze = [
        [1,1,1,0,0],
        [1,1,1,0,0],
        [0,0,0,0,1],
        [0,0,1,1,1]
        ]
 
##start = (0, 4)
##des = (4, 4)
##
##maze = [
##        [0,0,1,0,0],
##        [0,0,0,0,0],
##        [0,0,0,1,0],
##        [1,1,0,1,1],
##        [0,0,0,0,0]
##        ]

print(ball(maze,start,des))

# 733. Flood Fill

def PaintFill(grid,nc,r,c):

    def dfs(grid,nc,r,c,t):

        # base case, constraint
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
            return False
        
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
        dfs(grid,nc,r,c+1,t)
        dfs(grid,nc,r,c-1,t)
    

    target = grid[r][c]
    dfs(grid,nc,r,c,target)
    return grid
    
##photo = [[1, 1, 1],
##          [1, 0, 1],
##          [1, 0, 0]]
##
##Row = 1
##Col = 1
##newColor = 5

photo = [[0, 0, 0],
          [0, 1, 1]] #1
Row = 1
Col = 1
newColor = 1
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

# 47. Permutations II

# 39. Combination Sum

def combinationSum(arr,target):

    
    def dfs(arr,t,res,sub):

        # goal, base case
        if t == 0:
            res.append(sub)

        for i in range(len(arr)):
            # they are all sorted so there will not be duplicates
            # as arr[i] will be bigger than t so no duplicates
            # constraint
            if arr[i] <= t:
                # dfs
                # choic
                dfs(arr[i:],t-arr[i],res,sub+[arr[i]])

            else: # currnet value is greater than target so no point keepig looping
                break
    
    res = []
    sub = []
    arr.sort()
    dfs(arr,target,res,sub)
    return res



## My work ###
##def combinationSum(candidates,target):
##       
##        
##    def builComsum(candidates,ans,sub,t):
##        
##        # how to filter out duplicates
##        if sum(sub) == target and sorted(sub) not in ans:
##            ans.append(sorted(sub))
##            return
##        
##            #print(sorted(sub))
##            # when appending sub they are not sorted so new sorted ones are not in ans
##            # as they are not sorted when the first ones were appended
##            #ans.append(sub)
##            
##        if sum(sub) > target:
##            return
##        
##        for i in candidates:
##            builComsum(candidates,ans,sub+[i],target)
##            
##    sub = []
##    ans = []
##    builComsum(candidates,ans,sub,target)
##    
##    return ans

test= [2,3,6,7] # t,7
test1 = [2,3,5] # t,8
test2 = [8,7,4,3] # t,11

print(combinationSum(test,7))
print(combinationSum(test1,8))
print(combinationSum(test2,11))


############       return islands

def returnIslands(graph):

    # time complexity O(n2)
    cnt = 0
    for row in range(len(graph)):
        for col in range(len(graph[row])):
            if graph[row][col] == 1:
                dfs(graph,row,col)
                cnt +=1

    return cnt

def dfs(graph,r,c):

    # base case, constraint
    if r < 0: return
    if r >= len(graph): return
    if c < 0: return
    if c >= len(graph[0]): return
    if graph[r][c] != 1: return

    # mark visited cell "*"
    # checked so it will be false when revisieted
    graph[r][c] = "*" # without loop will go on forever
    dfs(graph,r+1,c)
    dfs(graph,r-1,c)
    dfs(graph,r,c+1)
    dfs(graph,r,c-1)

# just ahving these dont cover edge case
# dfs(graph,r+1,c)
# dfs(graph,r,c+1)
# in the graph below, lower left (1) will be left so islands will be 2

graph = [[1,1,1],
         [0,1,0],
         [1,1,1]]



##graph = [[1, 1, 0, 0, 0], 
##        [1, 1, 0, 0, 0], 
##        [0, 1, 0, 0, 0],
##        [1, 0, 0, 1, 1]]

print(returnIslands(graph))
print('test')

# NQueen problem

"""
solution requires that no two queens share the same row, column, or diagonal.

"""

"""algorithm
    1, choice - where to place queen
    2, constraints - row and col are not out of boundry,
        queen cannot be placed horizontally, vertically, diagnallly
    3,  check over the colum, not row (vertically,not horizontally)
    4, Goal, if col is equal to length of board[0]
    5, choice which col and which row to place
    6, validation

"""
def placeQueen(board,col):

    # base case
    if col >= len(board):
        return True

    # no need to check each colum in the current row
    # e.g. [0][0],[1][0],[2][0],[3][0]
    # as loop is based on colum

    
    # go through all the rows
    # Check this row on left side
    # after the current row cannot be the solution, it will move on with 0 col
    for row in range(len(board)):
        # check current cell
        # if there is already queen in the current row, go back to for loop and go to next row
        if isValid(board,row,col):
            board[row][col] = 1

            # move to the next cell
            # why do you need to return??
            # why indented?
            # when it is false, and indented, forloop will go to the next row
            # other wise when it is false, it will be stuck in the first row
            # just col keeps incrementing until it hits base case 
            if placeQueen(board,col+1):
                return True

            # if queen cannot be placed, mark current cell as 0
            # If placing queen in board[i][col]
            # doesn't lead to a solution, then 
            # queen from board[i][col] 
            board[row][col] = 0

    # if the queen can not be placed in any row in 
    # this colum col then return false 
    return False
        

def isValid(board,r,c):

    # check the row horizontally
    #for i in range(len(board[0])):
    for i in range(c):
        if board[r][i] == 1:
            return False

    # check diagnally upper left
    for i,j in zip(range(r,-1,-1),range(c,-1,-1)):
        if board[i][j] == 1:
            return False

    # check diagnally lower left 
    for i,j in zip(range(r,len(board)),range(c,-1,-1)):
        if board[i][j] == 1:
            return False

    return True

def QueenBoard(board):
    col = 0
    if placeQueen(board,col):
        return board
    return False

board = [ [0, 0, 0, 0], 
         [0, 0, 0, 0], 
         [0, 0, 0, 0], 
         [0, 0, 0, 0]
        ]  

print(QueenBoard(board))

# A rat in a maze

# the only can move down and right 

def wayOutForRat(maze):

    path = [[0 for i in range(len(maze))] for i in range(len(maze[0]))]

    # start off with the first cell
    if dfsRat(maze,0,0,path):
        print(path)
        return True

    #print("ayudame")
    return False

def isValid(maze,row,col):

    if row >= 0 and len(maze) > row and col >= 0 and len(maze[0]) > col and maze[row][col] == 1:
        return True

    return False

def dfsRat(maze,r,c,path):


    # Goal. check if last cell is 1 and if the rat has reached its destination
    #print(maze)
    if r == len(maze)-1 and c == len(maze[0])-1 and maze[r][c] == 1:
        path[r][c] = 'Goal'
        return True
    # check if current cell is valid cell
    # base case, constraints
    if isValid(maze,r,c):
        
        path[r][c] = "*"

        # after reaching the destination, since it does not return anything
        # it will go to false stament and becomes false
##        WaytoHeaven(maze,r+1,c)
##        WaytoHeaven(maze,r,c+1)

        # recurse and r increments so the cell right below will be checked
        # choice 
        if dfsRat(maze,r+1,c,path):
            return True
        if dfsRat(maze,r,c+1,path):
            return True

        return False
 
##maze = [ [1, 0, 1, 1, 1], 
##         [1, 1, 1, 0, 1], 
##         [0, 0, 0, 1, 1], 
##         [1, 0, 0, 1, 1]
##        ]

maze = [ [1, 0, 0, 0], 
         [1, 1, 0, 1], 
         [0, 1, 0, 0], 
         [1, 1, 1, 1]
        ] 


print(wayOutForRat(maze))
print('rat')

# 289. Game of Life
 
def gameOfLife(board):
   
    # # Neighbors array to find 8 neighboring cells for a given cell 
    # check all suuroundings
    surroundings = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]

    # used to avoid index out of range
    rows = len(board)
    cols = len(board[0])

    # Iterate through board cell by cell.
    for row in range(rows):
        for col in range(cols):

            # For each cell count the number of live neighbors.
            live_neighbors = 0
            for neighbor in surroundings:

                # row and column of the neighboring cell
                r = (row + neighbor[0])
                c = (col + neighbor[1])

                # if false this terminates
                # r should be grater than 0 so when it is -1 it will be false
                # c should be grater than 0 so when it is -1 it will be false
                # Check the validity of the neighboring cell and if it was originally a live cell.
                # make sure they are not out of bounds
                # looking for True
                # cuz for loop is base case
                if r < rows and \
                r >= 0 and \
                c < cols and \
                c >= 0 and \
                abs(board[r][c]) == 1:
                    
                    live_neighbors += 1

            # Rule 1 or Rule 3
            # if there is no parenthesis, it will be true even thotugh current cell is 0
            if board[row][col] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                # -1 signifies the cell is now dead but originally was live.
                board[row][col] = -1
            # Rule 4
            if board[row][col] == 0 and \
            live_neighbors == 3:
                # 2 signifies the cell is now live but was originally dead.
                board[row][col] = 2

    print(board)
    # Get the final representation for the newly updated board.
    for row in range(rows):
        for col in range(cols):
            if board[row][col] > 0:
                board[row][col] = 1
            else:
                board[row][col] = 0


    return board


board =[
         [0,1,0],
         [0,0,1],
         [1,1,1],
##          [0,0,0]
        ]

print(gameOfLife(board))

def returnPaths(maze,m,n):
    path = [0 for i in range(m+n-1)]
    allpath = []
    r,c = 0,0
    indx = 0
    # indx represents index of path
    findPaths(m,n,r,c,path,allpath,indx)
    #return allpath

def findPaths(m,n,r,c,path,allpath,indx):

    # when it hits the bottom
    if r == m-1:
        for i in range(c,n):
##            if arr[r][i] == "*":
##                return 
            # wanna move to right so from r
            path[indx + (i-c)] = maze[r][i]
            #path[indx + i - c] = maze[r][i]

        print(path)
        allpath.append(path)
        return

    # when it hits the right most
    if c == n-1:
        for i in range(r,m):
##            if arr[i][c] == "*":
##                return
            path[indx + (i-r)] = maze[i][c]
            #path[indx + i - r] = maze[i][c]

        print(path)
        allpath.append(path)
        return


##    if arr[r][c] == "*":
##        return 
    path[indx] = maze[r][c]
    
    findPaths(m,n,r+1,c,path,allpath,indx+1)
    findPaths(m,n,r,c+1,path,allpath,indx+1)
    

maze = [[1,2,3], 
            [4,5,6], 
            [7,8,9]]
##maze = [[1,2,3,4,5], 
##            [6,"*",8,9,10], 
##            [11,12,13,"*",15]] 
print(returnPaths(maze,3,3))


def permutation(nums):

    """algorithm
        1, Goal, base case will be based on length of arr
        2, once it reaches 0 it will be termianted as there is not valut to loop over
        3, Choice, slice array 

"""
    def dfs(nums, sub, res):
        # dfs, decrement,
        # base case 
        if not nums:
            res.append(sub)
            # return # backtracking
            # nums[:0] always returns 0 no matter how many elments in the list
            # when i is 0 it just takes first val until there is no val
            # when i is 1 gest first val and last val, [:1] gets first and
            # [i+1:] gets last val
            # and sub gets nums[i] which is 2 when i is 2
            # when i is 2, get first 2 vals
        for i in range(len(nums)):
            dfs(nums[:i]+nums[i+1:], sub+[nums[i]], res)
    res = []
    dfs(nums, [], res)
    return res


test = [1,2,3]
print(permutation(test))


# subset

##########################################

def subset(arr):
    #### MY WORK ######

    def build(arr,res,sub):


        res.append(sub)
        # base case
        # arr will be none as i increments 
        for i in range(len(arr)):
            build(arr[i+1:],res,sub+[arr[i]])

    res = []
    sub = []
    build(arr,res,sub)
    return res
    

test = [1,2,3]
# [ [3],[1],[2],[1,2,3],[1,3],[2,3],[1,2],[]]
print(subset(test))

def returnsubset(arr):

    res = []
    sub =[]
    ind = 1
    createSubset(arr,res,sub,ind)
    return res

def createSubset(arr,res,sub,ind):
    # if ind == len(arr): return
    res.append(sub)
    for i in range(ind,len(arr)+1):
        createSubset(arr,res,sub+[i],i+1)
        

test = [1,2,3]
# [ [],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
# [ [3],[1],[2],[1,2,3],[1,3],[2,3],[1,2],[]]
#print(returnsubset(test))

##########################################################

# generate parenthesis
# constraint is pair of open and close



def returnParenthesis(n):

    l = r = n
    res = []
    ans = ''

    dfsPare(l,r,res,ans)
    return res

def dfsPare(Open,Close,res,ans):
    # choice -  open or close
    # constraints - parenthesis should be closed
    # goal - all parethesis closed

    # constraints - if close is more than open, it cannot be closed
    if Open > Close:
        return

    # goal - if all paretheisies are closed
    if not Open and not Close:
        res.append(ans)

    # choice open or close
    if Open:
        dfsPare(Open-1,Close,res,ans+"(")

    if Close:
        dfsPare(Open,Close-1,res,ans+")")

print(returnParenthesis(3))
#[ "((()))", "(()())", "(())()", "()(())", "()()()"]

    



