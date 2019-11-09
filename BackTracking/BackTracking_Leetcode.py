# Leetcode
# Backtracking

# still not completely understand recursion so more practice!!

# new problem will be on top

# 10/27/2019

# 79. Word Search
def exist(temp_board, word):
    """
    key is to place back the original value at current cell
        when current cell does not lead to global solution.
        return true as soon as all letters are found.
        evaluate word by each letter not if current value is in given string
    The or statement in posted code returns immediately once either up, down, left, right returns true,
    algorithm
    1, visit each cell to see if current cell leads to global solution
    2, mark all visited cells but if the original cell is not the answer
        put back all the values at current cell
    """
    # this is a little bit more efficient because it returns true as soon as all the letters are found
    def dfs(board, r, c, ans, cnt):

        # constraint, base case
        if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]) or board[r][c] != word[cnt]:
            return False

        ans += word[cnt]
        # goal
        # as soon as all letters are found
        if ans == word:
            return True
        # when current cell does not lead to global solution, place original value at current cell
        # so stroe original value
        temp = board[r][c]
        # w/o this, it will cause infinite loop
        board[r][c] = "*"
        # current letter is found so next letter
        cnt+=1

        if dfs(board, r + 1, c, ans,cnt) or \
            dfs(board, r - 1, c,ans,cnt) or \
            dfs(board, r, c + 1,ans,cnt) or \
            dfs(board, r, c - 1,ans,cnt):
            return True

        # place original string at the current cell
        # grid = [["a","b"]], s = "ba", otherwise this case won't work
        # in test case 6, if, when it gets 2 A, board[1][1] gets B since cnt is 2 so this wont work
        board[r][c] = temp
        return False

    board = temp_board[:]
    for i in range(len(board)):
        for j in range(len(board[0])):
            # takes place only when current cell equals the first letter of target
            if board[i][j] == word[0]:
                if dfs(board,i,j,"",0):
                    return True

    return False

    # brute force
    # this code evaluates all of them no matter what and then return the or statement.
    def dfs(board, r, c, ans):

        # brute force
        # base case
        if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]) or board[r][c] not in word:
            return False

        ans += board[r][c]
        # goal
        if ans == word:
            return True

        # mark it as visited
        tmp = board[r][c]
        board[r][c] = "*"

        # choice
        up = dfs(board, r + 1, c, ans)
        down = dfs(board, r - 1, c, ans, )
        left = dfs(board, r, c + 1, ans, )
        right = dfs(board, r, c - 1, ans, )

        board[r][c] = tmp

        return up or down or left or right

    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(board, i, j, ''):
                return True

    return False

board = [['A','B','C','E'],
         ['S','F','C','S'],
         ['A','D','E','E']]

# word = "AB"
word = "ABCCED" # return true.
word1 = "SEE" # return true.
word2 = "ABCB" # return false.
board1 = [["a","b"]] # true
word3 = "ba"
board2 = [["b"],["a"],["b"],["b"],["a"]] # False
word4 = "baa"
meow =[["C","A","A"],["A","A","A"],["B","C","D"]]
s = "AAB"
# [["a","b"],["c","d"]]
# "cdba"
print(exist(board,word))
print(exist(board,word1))
print(exist(board,word2))
print(exist(board1,word3))
print(exist(board2,word4))
print(exist(meow,s)) # test case 6
print()

# 10/21/2019


# 329. Longest Increasing Path in a Matrix

def longestIncreasingPath(matrix):
    """
    key is # how to keep going after moving to another cell than current one, dfs in dfs function just like graph recursion bfs
    and how to keep track of longest path - get max path out of current cell
    # path = 1 + max(down,up,left,right), will fetch the max path
    and how to make it more efficient - using dictionary, update max path after exploring all possible solutions

    algorithm
    1, visit each cell, keep dfs going until previous is greater or current cell out of boundary
    2, for the sake of efficiency, use dictionary to keep track of each cell's max path
    3, it is efficient because dictionary will get updated during iteration
        so all cells are not visited.
    """

    # Time Complexity O(mn). Each vertex/cell will be calculated once and only once, and each edge will be visited once and only once.
    # The total time complexity is then O(V+E)O(V+E).
    # V is the total number of vertices and E is the total number of edges.
    def dfs(row, col, prev):

        # constraint
        # make sure it is out of boundary or previous value was larger
        if row < 0 or row >= len(matrix) or col < 0 or col >= len(matrix[0]) or matrix[row][col] <= prev:
            return 0

        # if current cell already visited, return its max path
        if (row,col) in dic:
            return dic[(row,col)]

        # get max path out of current cell
        path = 1 + max(dfs(row+1, col, matrix[row][col]),dfs(row-1, col, matrix[row][col]),dfs(row, col+1, matrix[row][col]),dfs(row, col-1, matrix[row][col]))

        dic[(row,col)] = path
        return path

    if not matrix: return 0
    # m = len(matrix)
    # n = len(matrix[0])
    LIP = 0
    dic = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (i,j) not in dic:
                LIP = max(LIP, dfs(i, j, float("-inf")))

    return LIP

    # brute force
    # time complexity
    # Time complexity : O(2m+n)
    # The search is repeated for each valid increasing path. In the worst case we can have O(2^{m+n})O(2m+n) calls

    # def dfs(row, col, prev):
    #     if row < 0 or row >= len(matrix) or col < 0 or col >= len(matrix[0]) or matrix[row][col] <= prev:
    #         return 0
    #
    #     return 1 + max(dfs(row + 1, col, matrix[row][col]), dfs(row - 1, col, matrix[row][col]),
    #                    dfs(row, col + 1, matrix[row][col]), dfs(row, col - 1, matrix[row][col]))
    # if not matrix: return 0
    # LIP = 0
    # for i in range(len(matrix)):
    #     for j in range(len(matrix[i])):
    #        LIP = max(LIP, dfs(i, j, float("-inf")))
    #
    # return LIP

meow = [[9,9,4],
        [6,6,8],
        [2,1,1]]

meow1 = [[7,6,1,1],
         [2,7,6,0],
         [1,3,5,1],
         [6,6,3,2]]

print(longestIncreasingPath(meow))
print(longestIncreasingPath(meow1))
print()


# 74. Search a 2D Matrix

def searchMatrix(matrix, target):
    """
    key is to treat this matrix as 1d array
    and how to find out row and col
    algorithm
    1, operate normal binary search, but how to find out row and col
        and move on to next sub array?
        row is, mid // length of current row, 5 // 4 = 1, 4 // 5 = 0, 2 // 4 = 0
        col is mid % lenght of current row, 5 % 4 = 1, 4 % 5 = 4, 2 % 4 = 2
    2, if current val, matrix[row][col] is greater than target, increase left bound
        other wise increase right bound
    3, if current mid is target, return True
    """

    # time Complexity O(log(mn))
    m,n = len(matrix),len(matrix[0])
    # left starts off at 0,0 and r starts off at 2,3
    l,r = 0,(m*n) - 1
    res = []
    while l <= r:
        mid = (l+r) // 2
        row = mid // n
        col = mid % n
        #res.append(matrix[row][col])
        if matrix[row][col] == target:
            #print(res)
            return True

        # if current cell is greater, target is in left bound, so go to left
        if matrix[row][col] > target:
            r = mid - 1

        else:
            l = mid + 1

matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3

matrix1 = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target1 = 13


print(searchMatrix(matrix,target))
print(searchMatrix(matrix1,target1))
print()
# 289. Game of Life

def gameOfLife(board):
    """
    qlgorithm
    key is in O(1) space solution, if we update board, when 0 will turn into 1, that will mess up the board
    so turn cell that will die into -1 and turn cell that will be alive into 2
    1, create 8 directions, vertically, horizontally, diagonally
    2, visit each cell, so we dont need any hashset to keep track of visited cells
    3, bfs from each cell so we know how many will die and become alive
    """

    # Time Complexity, O(M×N), where MM is the number of rows and NN is the number of columns of the Board.
    # space complexity O(1)
    m, n = len(board), len(board[0])
    directions = [(0, 1), (1, 0), (-1, 1), (1, -1), (0, -1), (-1, 0), (1, 1), (-1, -1)]
    # visit each cell
    for row in range(len(board)):
        for col in range(len(board[row])):
            # initiate temporary variables
            # so checking original cell is easier
            r, c = row, col
            lives = 0

            # visit all neighbours
            for x, y in directions:
                r = row + x
                c = col + y
                # abs should be over current cell, not the 1 the is compared to
                if 0 <= r < m and 0 <= c < n and abs(board[r][c]) == 1:
                    lives += 1

            # Rule 1 or Rule 3
            if board[row][col] == 1:
                if lives < 2 or lives > 3:
                    # -1 denotes the cell is now dead but originally was live.
                    board[row][col] = -1

            if board[row][col] == 0 and lives == 3:
                # 2 signifies the cell is now live but was originally dead.
                board[row][col] = 2

    # Get the final representation for the newly updated board.
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == -1:
                board[i][j] = 0

            elif board[i][j] == 2:
                board[i][j] = 1

    return board

board =[
         [0,1,0],
         [0,0,1],
         [1,1,1],
         [0,0,0]]

print(gameOfLife(board))


def riverSizes(matrix):
    # Write your code here.

    def dfs(matrix, r, c):
        if r < 0 or r >= len(matrix) or c < 0 or c >= len(matrix[0]) or matrix[r][c] != 1:
            return 0

        matrix[r][c] = -1  # mark it as visited

        up = dfs(matrix, r - 1, c)
        down = dfs(matrix, r + 1, c)
        left = dfs(matrix, r, c - 1)
        right = dfs(matrix, r, c + 1)

        return up + down + left + right + 1

    res = []
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] == 1:
                num_rivers = dfs(matrix, row, col)
                if num_rivers:  # if it is greater than 0
                    res.append(num_rivers)

    return res


testInput = [[0]]  # []
testInput1 = [[1]]  # [1]
testInput2 = [[1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0]]  # expected = [1, 2, 3]
testInput3 = [
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 0],
    [1, 0, 1, 0],
]  # [1, 1, 2, 3]
testInput4 = [
    [1, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
]  # expected = [1, 2, 2, 2, 5]

testInput5 = [
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
]  # expected = [1, 1, 2, 2, 5, 21]

print(riverSizes(testInput))
print(riverSizes(testInput1))
print(riverSizes(testInput2))
print(riverSizes(testInput3))
print(riverSizes(testInput4))
print(sorted(riverSizes(testInput5)))  # sort dont matter tho hahaha

from collections import deque
# 490. The Maze
def hasPath(maze, start, destination):
    """
    key is as long as current cell is not wall, it will keep going until it hits wall
    and how to store safe cells in visited and queue
    and how you set up directions
    algorithm
    1, iterative BFS, as long as current cell is not wall, keep going by adding direction to current cell
    2, when it hits the wall, store the cell right before the wall
    3, when queue spits front value, make sure it is destination or not

    """
    m, n = len(maze), len(maze[0])
    # crate direction so it can become bfs as vitis adjacent nodes
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    q = deque([start])
    visited = set()
    while q:
        row, col = q.popleft()
        # goal
        if [row, col] == destination: return True
        for x, y in directions:
            r = row + x
            c = col + y
            # this will go on until it is out of boundary or encounter 1
            # constraint
            # template to make sure if current cell is not out of boundary
            while 0 <= r < m and 0 <= c < n and maze[r][c] == 0:
                # if it satisfies the condition, move on according to neighbor
                # choice
                r += x
                c += y

            # be careful, 2 - -1 = 3
            # x - neighbor[0] cuz when encoutering obstacles,
            # cannot mark current cell (obstacle) as it is already visited or obstacle  so subtract
            # e,g,when (1,3) and moves on to (2,3) it is an obstacle
            # cannot mark it as True
            # e,g,when (0,3) and moves on to (0,2) it is an obstacle
            # so mark (0,3) = 2 - -1
            # not current cell but the very pervious cell that satisfied condition
            # that way subtract nei and current cell to get previous cell
            if (r - x, c - y) not in visited:
                visited.add((r - x, c - y))
                q.append((r - x, c - y))

    return False


start = [0, 4]
des = [2, 1]

maze = [
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1]
]

start1 = [0, 4]
des1 = [4, 4]
maze1 = [
       [0,0,1,0,0],
       [0,0,0,0,0],
       [0,0,0,1,0],
       [1,1,0,1,1],
       [0,0,0,0,0]
       ]

print(hasPath(maze, start, des))
print(hasPath(maze1, start1, des1))

# 695. Max Area of Island

def maxAreaOfIsland(grid):
    """
    algorithm
    key is mark the visited places by placing "-1"
    and how to count all the isldnds from current cell
    1, visit each unvisited cell
    2, and count how many times dfs takes place
    """

    def dfs(grid,r,c):
        # constraint
        # when checking colum, if it is not grid[0] it will return wrong answer
        # e,g, [[0.1]], should return 1 but if just grid, c will be out of boundary and end up with wrong answer
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] != 1:
            return 0

        # mark it as visited
        grid[r][c] = -1
        # these 4 variables keeps track of counter
        # just like tree. like count height of tree
        up = dfs(grid,r+1,c)
        down = dfs(grid,r-1,c)
        left = dfs(grid,r,c+1)
        right = dfs(grid,r,c-1)

        # # when 1 forund it will return accumulative + 1
        return up + down + left + right + 1

    cnt = 0
    #res = []
    # visit each unvisited cell
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                #res.append(dfs(grid,i,j))
                cnt = max(cnt,dfs(grid,i,j))

    #print(res)
    return cnt

meow = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]

meow1 = [[0,1]]
meow2 = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
print()
print(maxAreaOfIsland(meow))
print(maxAreaOfIsland(meow1))
print(maxAreaOfIsland(meow2))


"""
Given a 2d grid map of '1's (land) and '0's (water),
count the number of islands.
An island is surrounded by water and is formed by connecting adjacent
lands horizontally or vertically.
"""
def numIslands(grid):

    """
    algorithm
    key is mark the visited places by placing "-1"
    1, visit each unvisited cell
    2, and count how many times dfs takes place
    """

    def dfs(grid,r,c):
        # constraint
        # check if current cell is not out of boundary
        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] != 1:
            return

        # mark current cell as visited
        grid[r][c] = "-1"
        # go 4 directions from current cell
        dfs(grid, r+1, c)
        dfs(grid, r-1, c)
        dfs(grid, r, c+1)
        dfs(grid, r, c-1)

    cnt = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                dfs(grid,i,j)
                cnt+=1

    return cnt

graph = [[1,1,1],
         [0,1,0],
         [1,1,1]]

graph1 = [[1, 1, 0, 0, 0],
       [1, 1, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [1, 0, 0, 1, 1]]

print(numIslands(graph))
print(numIslands(graph1))




# 733. Flood Fill

def floodFill(image, sr, sc, newColor):

    """
    algorithm
    1, get target and dfs
        goal: is to fill current cell
        Constraint: current cell out of boundary or current cell not target
        Choice, go 4 directions from current cell
    """

    def dfs(image, r, c, nc, t):
        # base case, constraint
        if r < 0 or r >= len(image) or c < 0 or c >= len(image[0]):
            return False

        # this is for when nc and t are the same
        # and as it will look for all the surroundings for each cell,
        # if t and nc are the same, never ends
        # grid[r][c] == nc this will make sure if t and nc are the same
        # it will bail out
        # if it is not target, dont need to paint
        if image[r][c] != t:
            return False
        # edge case when new color is same as target
        if image[r][c] == nc:
            return False

        # goal
        image[r][c] = nc

        dfs(image, r + 1, c, nc, t)
        dfs(image, r - 1, c, nc, t)
        dfs(image, r, c + 1, nc, t)
        dfs(image, r, c - 1, nc, t)

    target = image[sr][sc]
    dfs(image, sr, sc, newColor, target)
    return image

photo = [[1, 1, 1],
         [1, 1, 0],
         [1, 0, 1]]

Row = 1
Col = 1
newColor = 2

photo1 = [[0, 0, 0],
          [0, 1, 1]] #1
Row1 = 1
Col1 = 1
newColor1 = 1

screen = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 1],
            [1, 2, 2, 2, 2, 0, 1, 0],
            [1, 1, 1, 2, 2, 0, 1, 0],
            [1, 1, 1, 2, 2, 2, 2, 0],
            [1, 1, 1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1, 2, 2, 1],
        ]

R = 4
C = 4
nC = 3

print(floodFill(photo,Row,Col,newColor))
print(floodFill(photo1,Row1,Col1,newColor1))
print(floodFill(screen,nC,R,C))



