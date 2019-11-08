# LeetCode


# 11/5/2019


# 11/4/2019
# 269. Alien Dictionary

from collections import defaultdict,deque
def alienOrder(words):
    """
    key is how to map out indegree and graph
    other than that, just a simple bfs topological sort
    algorithm
    1, create indegree map for each letter being 0
    2, create graph where we compare current word and next word
        and if letters don't match, update indegree map and current map
    3, be careful with the way cycle is detected
    """
    # create dictionary for indegree
    indegree = {}
    for word in words:
        for c in word:
            # no duplicates as nothing is appended, just updated
            indegree[c] = 0

    # now all letter has 0 indgree
    # create graph
    g = defaultdict(list)

    # starts off with 1 as we would like to compare two neighbours at the same time
    for i in range(1,len(words)):
        cur_w = words[i-1]
        next_w = words[i]
        # do we need to check its lenghts?
        # no because zip terminates when shorter word is done
        #  # Note that by using zip here, will save us the trouble of having to
        # find out which of the two words is the shortest. We are only guaranteed
        # lexicographical order up to the first character that is different or to the
        for c1,c2 in zip(cur_w,next_w):
            if c1 != c2:
                indegree[c2] += 1

                g[c1].append(c2)
                # if there is no break, there is gonna be a cycle in test 5, ["za","zb","ca","cb"]
                # b gets a after 2nd iteration which is a cycle
                # You need to break out of the loop because once there are two characters that don't match, the rest of the strings don't give you any more useful lexicographical information.
                break # why break?

    q = deque([letter for letter in indegree if indegree[letter] == 0])
    res = ''
    while q:
        cur_w = q.popleft()
        res += cur_w
        for edge in g[cur_w]:
            # visiting
            indegree[edge] -=1
            if indegree[edge] == 0:
                q.append(edge)

    # how to detect cycle is to compare the length of result and indgree array
    return res if len(res) == len(indegree) else None

meow = ["wrt", "wrf", "er", "ett", "rftt"] # w -> e -> r -> t -> f
meow1 = ["z","x","z"]
meow2 = ["a","zy","zx"] # order doesnt matter
meow3 = ["wrtkj","wrt"]
meow4 = ["wrt","wrtkj"]
meow5 = ["za","zb","ca","cb"] # "abzc"
meow6 = ['sbde','sbax']
print(alienOrder(meow))
print(alienOrder(meow1))
print(alienOrder(meow2))
print(alienOrder(meow3))
print(alienOrder(meow4))
print(alienOrder(meow5))
print(alienOrder(meow6))
print()


# 210. Course Schedule II
from collections import defaultdict,deque
def findOrder(numCourses,prerequisites):

    # BFS
    """
    algorihtm
    1, create undirected graph,
    2, and create array that keeps track of indegree.
    3, create queue and append vertices that have no incoming endges
    4, right after popping vertex from queue, append it to result array
    5, Decrease in-degree by 1 for all its neighboring nodes or current vertex (edges)
    6, If in-degree of a neighboring nodes is reduced to zero, then add it to the queue.
    """
    g = defaultdict(list)
    # directed graph
    for u, v in prerequisites:
        g[u].append(v)

    # get number of incoming edges for each vertex
    indegree = [0] * numCourses
    for i in range(numCourses):
        for edge in g[i]:
            indegree[edge] += 1

    # Pick all the vertices that have 0 in-degree and add them into a queue
    q = deque([i for i in range(len(indegree)) if indegree[i] == 0])
    # for i in range(len(indegree)):
    #     if indegree[i] == 0:
    #         q.append(i)
    res = deque([])
    while q:
        # start off with the vertex that has no in-degree edges
        cur_vtx = q.popleft()
        res.appendleft(cur_vtx)
        for edge in g[cur_vtx]:
            # now we are visiting
            indegree[edge] -= 1

            # if neighboring edge has no incoming edge, append it to result
            if indegree[edge] == 0:
                q.append(edge)

    # how to check if there is a cycle is to compare length of result and length of indgree
    return res if len(res) == numCourses else []

    # recursive dfs
    g = defaultdict(list)
    # directed graph
    for u, v in prerequisites:
        g[u].append(v)

    def dfs(g, v, visited, each_walk, res):
        visited.add(v)
        each_walk.add(v)
        for edge in g[v]:

            # this one returns False
            # if edge in each_walk: return True

            if edge not in visited:
                dfs(g, edge, visited, each_walk, res)

            # be careful of where to put this if statement,
            # if this comes before above if statement, dfs will still take place and
            # return false even cycle is found
            if edge in each_walk:
                return True

        each_walk.discard(v)
        res.append(v)
        return False

    visited = set()
    each_walk = set()
    res = deque([])
    for vertex in range(numCourses):
        if vertex not in visited:
            if dfs(g, vertex, visited, each_walk, res):
                return []

    return res

courses = 4
meow = [[1,0],[2,0],[3,1],[3,2]] # #  [0,1,2,3] or [0,2,1,3]
c1 = 2
meow1 = [[1,0]]
c2 = 3
meow2 = [[0,2],[1,2],[2,0]]
c3 = 2
meow3 = [[0,1],[1,0]]

print(findOrder(courses,meow))
print(findOrder(c1,meow1))
print(findOrder(c2,meow2))
print(findOrder(c3,meow3))
print()


from collections import defaultdict,deque
# 684. Redundant Connection
# An edge will connect two nodes into one connected component.
# When we count an edge in, if two nodes have already been in the same connected component,
# the edge will result in a cycle. That is, the edge is redundant.
# https://leetcode.com/problems/redundant-connection/discuss/123819/Union-Find-with-Explanations-(Java-Python)
# https://leetcode.com/problems/redundant-connection/discuss/331028/Best-Python-Solution-(Union-Find-Explained) # great explanation


def findRedundantConnection(edges):  # detect loop and remove it
    """
    algorithm,
    1, detect loop and remove it. [1,2] and [1,3] are already connected as they are undirected
    1, create union and find function and if loop found, delete it
    :param edges:
    :return:
    """

    def find_parent(x):
        # If x is the parent of itself
        # return x
        if parent[x] == 0:
            return x
        # if x is not parent of itself,
        # in test case2, after 3 is retured in line 387 (as x), parent[2] will become 3
        # and parent[2] = 3 will be returned in ln 392 and parent[1] gets 3 as it is what is returned
        # same as parent[0]
        # keep calling parent function until its parent found
        parent[x] = find_parent(parent[x])
        # return parent grabbed from find_parent function
        return parent[x]

    def union(x, y):
        """
        1, find parent(root) of x and y
        2, if they are same, x and y are coming from the same parent
        3, if the are not just update it

        """
        x = find_parent(x)  # x repersents parent(original vertex)
        y = find_parent(y)  # y represents edge
        if x == y:
            return True
        # For every edge, we unify u and v. #[1]
        # Which means u and v and all of their parents all lead to one root. [1,2] [1,3] 1 to 3 are all connected
        parent[x] = y

    parent = [0 for i in range(len(edges))]
    for u, v in edges:  # u is vertex v is edge
        if union(u - 1, v - 1):  # -1 cuz index and list number don't match
            # first number starts off with 1 so -1 to adjust to index
            return [u, v]

    return None


meow = [[1, 2], [1, 3], [2, 3]]
meow1 = [[1, 2], [2, 3], [3, 4], [1, 4], [1, 5]]
print(findRedundantConnection(meow))
print(findRedundantConnection(meow1))

# 547. Friend Circles

def findCircleNum(M):
    """
    algorithm
    1, this is undirected adjacencyy matrix so make graph with two way edges
    2, find connected components so use bfs
    3, every time it reaches edge that has no incoming
    """
    #### my work, nailed it the first time ###
    # time complexity, O(N2) due to adjacencyy matrix

    # create undirected graph
    graph = defaultdict(list)
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i][j] == 1:
                graph[i].append(j)
                graph[j].append(i)

    # this one traverse one component
    # meaning, it will go until the edge that is not connected to anything
    def bfs(vertex,visited):
        visited.add(vertex)
        queue = deque([vertex])
        while queue:
            cur_vtx = queue.popleft()
            for edge in graph[cur_vtx]:
                if edge not in visited:
                    visited.add(edge)
                    queue.append(edge)

        return 1

    visited = set()
    cnt = 0
    for vertex in range(len(graph)):
        if vertex not in visited:
            cnt += bfs(vertex, visited)

    return cnt

meow = [[1,1,0],
        [1,1,0],
        [0,0,1]]

meow1 = [[1,1,0],
        [1,1,1],
        [0,1,1]]

print('meow')
print(findCircleNum(meow))
print(findCircleNum(meow1))

# 743. Network Delay Time
# https://leetcode.com/problems/network-delay-time/discuss/187713/Python-concise-queue-and-heap-solutions
# https://leetcode.com/problems/network-delay-time/discuss/329376/Efficient-O(E-log-V)-Python-Dijkstra-min-heap-with-explanation

from collections import defaultdict
import heapq
def networkDelayTime(times, N, K):
    """
       algorithm
       1, How long does network have to wait to get signal to all nodes
       meaning what is the furtherst node and shortest path to it
       1, return the biggest path from node K
       # tricky part is that
        biggest path is actually shortest path out of all biggest paths from node K
        e,g, in meow4, from K(1) to 3, path from 1 to 3 is 7
        but path from 1 to 2 to 3 is 4 which is shortest out of biggest path
       2, crete dict that keeps track of each node path and holds smallest path (3:7 and 3:4) holds 3:4
       3, if length of dict equals N return max path
       """
    g = defaultdict(dict)
    for u, v, w in times:
        g[u][v] = w

    heap = [(0, K)]
    dst = {}
    while heap:
        cost, s = heapq.heappop(heap)
        # this dic keeps track of path along with duplicates
        # if already visited current node, skip this vertex
        if s not in dst:
            # if current vertex not visited, visit all the neighbours and keep track of path
            dst[s] = cost

            for d in g[s]:
                heapq.heappush(heap, (cost + g[s][d], d))

    # if lenght of dst equals N that means all nodes are visited
    return max(dst.values()) if len(dst) == N else -1


meow  =[[2,1,1],[2,3,1],[3,4,1]]
n = 4
k = 2
meow1 = [[1,2,1],[2,1,3]]
n1 = 2
k1 = 2
meow2 = [[1,2,1],[2,3,7],[1,3,4],[2,1,2]]
n2 = 3
k2 = 1
meow3 = [[1,2,1]]
n3 = 2
k3 = 2
meow4 = [[1,2,1],[2,3,2],[1,3,7],[2,1,2]]
n4 = 3
k4 = 1
print(networkDelayTime(meow,n,k))
print(networkDelayTime(meow1,n1,k1))
print(networkDelayTime(meow2,n2,k2))
print(networkDelayTime(meow3,n3,k3))

# 787. Cheapest Flights Within K Stops

def findCheapestPrice(n,flights,src,dst,k):

    """
    algorithm
    1, dijikistra algorithm
    2, find the cheapest path from current city to destination
    3, key is k should start off 1 as the current city you are in should already be visited
    4, otherwise, when k is 0, you will not travel any city

    """
    # create graph
    g = defaultdict(dict)
    for u,v,c  in flights:
        g[u][v] = c

    # heap = [cost,cur_vtx,stop]
    heap = [(0,src,k+1)] # K should be +1 as current city you are in should count
    while heap:
        cost,cur_city,k = heapq.heappop(heap)
        if dst == cur_city:
            return cost

        # as long as k is greater than 0, visit neighbor cities
        # if k is 0 just get neighbours of the beginning city
        if k > 0:
            for nei in g[cur_city]:

                heapq.heappush(heap,(cost+g[cur_city][nei],nei,k-1)) # nei is key of dict

    return -1

n = 3
meow = [[0,1,100],[1,2,100],[0,2,500]]
src = 0
dst = 2
k = 0

n1 = 4
meow1 = [[0,1,1],[0,2,5],[1,2,1],[2,3,1]]
src1 = 0
dst1 = 3
k1 = 1

n2 = 2
meow2 =[[0,1,2]]
src2 = 1
dst2 = 0
k2 = 0

n3 = 3
meow3 = [[0,1,100],[1,2,100],[0,2,500]]
src3 = 0
dst3 = 2
k3 = 1

# when you are already in destination
n4 = 3
meow4 = [[0,1,100],[1,2,100],[0,2,500]]
src4 = 0
dst4 = 0
k4 = 1

print(findCheapestPrice(n, meow,src, dst,k))
print(findCheapestPrice(n1, meow1,src1, dst1,k1))
print(findCheapestPrice(n2, meow2,src2, dst2,k2))
print(findCheapestPrice(n3, meow3,src3, dst3,k3))
print(findCheapestPrice(n4, meow4,src4, dst4,k4))

from collections import defaultdict, deque
import heapq


def findMinHeightTrees(n, edges):
    # edge case
    if n == 1: return [0]
    # why this is set?
    # indirected so create 2 way graph
    adj = defaultdict(list)
    for u,v in edges:
        adj[u].append(v)
        adj[v].append(u)
    #adj = [set() for _ in range(n)]
    # for i, j in edges:
    #     adj[i].add(j)
    #     adj[j].add(i)

    # cut leaves here
    # leaves = [i for i in range(n) if len(adj[i]) == 1]
    # get leaves meaning vertices that have no edges
    leaves = []
    for i in range(n):
        if len(adj[i]) == 1:
            leaves.append(i)

    # cut leaves
    while n > 2:
        n -= len(leaves)
        newLeaves = []
        for i in leaves:
            j = adj[i].pop()
            adj[j].remove(i)
            if len(adj[j]) == 1:
                newLeaves.append(j)
        leaves = newLeaves
    return leaves

# n = 6
# edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]
# print(findMinHeightTrees(n, edges))

n1 = 4
edges1 = [[1, 0], [1, 2], [1, 3]]
print(findMinHeightTrees(n1, edges1))

# Word Ladder

def ladderLength(beginWord, endWord, wordList):
    """
    algorithm
    1, first create combination of all the words in wordlist
        memorize how to create combination word
        manipulate index and slice notation
    2, and create combination of current word,
    3, if combination of current word in dic, push words(values) under the combination(key) to heap
        along with keeping track of level
    3, if we reach end word return the how long it took to get there

    """
    ##

    # edge case
    if endWord not in wordList or not endWord or not beginWord or not wordList:
        return 0

    # Since all words are of same length.
    L = len(beginWord)

    # Dictionary to hold combination of words that can be formed,
    # from any given word. By changing one letter at a time.
    # create all combination dictionary
    all_combo_dict = defaultdict(list)
    visited = set()  # To prevent cycles, use a visited dictionary.
    for word in wordList:
        for i in range(L):
            # Key is the generic word
            # Value is a list of words which have the same intermediate generic word.

            #### remember how to create combination word #######
            all_combo_dict[word[:i] + "*" + word[i + 1:]].append(word)

    # Queue for BFS
    #queue = deque([(beginWord, 1)])
    # heap
    heap = [(1, beginWord)]
    # Visited to make sure we don't repeat processing same word.
    # cuz we already visited current word(by being here)
    visited.add(beginWord)
    while heap:
        level,current_word = heapq.heappop(heap)
        # current_word, level = queue.popleft()
        for i in range(L):
            # Intermediate words for current word
            intermediate_word = current_word[:i] + "*" + current_word[i + 1:]
            # Next states are all the words which share the same intermediate state.
            # loop over all the words that can be transformed to next combination
            # hit - hot, h*t = h*t
            for word in all_combo_dict[intermediate_word]:
                # If at any point if we find what we are looking for
                # i.e. the end word - we can return with the answer.
                if word == endWord:
                    return level + 1
                # Otherwise, add it to the heap. Also mark it visited
                if word not in visited:
                    visited.add(word)
                    heapq.heappush(heap,(level + 1, word))
                    # queue.append((word, level + 1))
            # print(all_combo_dict[intermediate_word])
            # all_combo_dict[intermediate_word] = []
    return 0


beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
print(ladderLength(beginWord, endWord, wordList))

beginWord1 = "hit"
endWord1 = "cog"
wordList1 = ["hot","dot","dog","lot","log"]
print(ladderLength(beginWord1, endWord1, wordList1))


# 323. Number of Connected Components in an Undirected Graph
# https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/discuss/77675/Python-union-find-code
def countComponents(n, edges):
    """ algorithm,
        1, this is indirected graph so need to create graph
            that shows each vertex with a two-way realationship
        2, do either bfs or dfs and if I reach the vertex
            where I can't explore any edges, increment counter
        3,
    """

    ## bfs is faster

    visited = set()
    graph = defaultdict(list)
    # create representation
    for v, e in edges:
        # if there is just one graph, (graph[v].append(e)
        # in the case of 2, [[1,0]]
        # output will be 2 as 0 and 1 are both components
        # The edges indicate a two-way relationship, in that each edge can be traversed in both directions.
        graph[v].append(e)
        graph[e].append(v)

    def bfs(vertex):
        queue = deque([vertex])
        while queue:
            cur_vtx = queue.popleft()
            visited.add(cur_vtx)
            for edge in graph[cur_vtx]:
                if edge not in visited:
                    queue.append(edge)

        return 1

    return sum(bfs(i) for i in range(n) if i not in visited)
    # for vertex in range(n):
    #     if vertex not in visited:
    #         bfs(vertex,visited)
    #         cnt +=1
    # return cnt

##    def dfs(vertex):
##
##        if not vertex in visited:
##            visited.add(vertex)
##            for edge in graph[vertex]:
##                if edge not in visited:
##                    dfs(edge)
##
##        return 1

# return sum(dfs(i) for i in range(n) if i not in visited)

meow = [[0, 1], [1, 2], [3, 4]]
meow1 = [[0, 1], [1, 2], [2, 3], [3, 4]]
meow2 = [[1,0]] # 1
meow3 = [[0,1],[1,2],[0,2],[3,4]]
print("countComponents")
print(countComponents(5,meow))
print(countComponents(5,meow1))
print(countComponents(2,meow2))
print(countComponents(5,meow3))

# 797. All Paths From Source to Target

def allPathsSourceTarget(graph):
    # Time Complexity: O(2^N)
    """
    algorithm
    1, create result as global variable
    2, dfs over graph and when current vertex is destination(len -1) append it to res
    3, every time it moves to adjacent edges, append it to path as we are trying to keep track of all vertex to the path
    3, this is pretty similar to backtracking, pretty similar to brute force for dp
        dfs untile target is found and append all the path to result

    """

    def dfs(cur,path):

        if cur == des:
            res.append(path)

        for edge in graph[cur]:
            dfs(edge,path+[edge])

    res = [] # this is global
    des = len(graph) -1
    # we are looking for the path from 0 that is why it is 0
    dfs(0,[0]) # starts off with 0, and and we already visited 0
    return res

    # vistied = set() # since i wanna find all paths, we dont need to check every visit
    # allpath = []
    # paths = [[0]]  # path starts off at 0
    # des = len(graph) # empty list alos counts
    # while paths:
    #     # pop() is O(1)
    #     # pop(k) is O(k)
    #     path = paths.pop()  # pops all the vertxies that have been visited
    #     for edge in graph[path[-1]]:  # loop over the last visited vertex
    #         # [0,2] = graph[2] which is 3
    #
    #         # if edge is destination, we found a path
    #         if edge == des:
    #             allpath.append(path + [edge])
    #
    #         else:  # should be else as we try to find out all paths to des
    #             # thus no need to have visit
    #             paths.append(path + [edge])
    #
    # return allpath

meow = [[1, 2], [3], [3], []]
print('allPathsSourceTarget')
print(allPathsSourceTarget(meow))

# detect cycle in directed graph
# 207. Course Schedule

def canFinish(numCourses,prerequisites):
    """
    algorithm, detect cycle in directed graph
    1, if ther is a cycle, return false
    2, cuz if there is a cycle you cannot take prerequisite
    3, by having each_loop, it keeps track of visited vertices in each vertex
    4, if we are already visited, there is a loop

    """
    graph = defaultdict(list)
    # create representation
    for u, v in prerequisites:
        graph[u].append(v)

    def dfs(vertex, visited, each_loop):

        visited.add(vertex)
        each_walk.add(vertex)

        for edge in graph[vertex]:
            if edge not in visited:
                dfs(edge, visited, each_loop)

            if edge in each_walk:  # detect cycle
                return True

        each_walk.discard(vertex)
        return False

    visited = set()
    each_walk = set()
    for vertex in range(numCourses):  # visit all vertices(classes)
        if dfs(vertex, visited, each_walk):  # if cycle found, return false
            return False

    return True

print('canFinish')
print(canFinish(4, [[1,0],[2,0],[3,1],[3,2]])) # true
print(canFinish(3, [[1,0],[1,2],[0,1]])) # false
print(canFinish(3, [[0,1],[0,2],[1,2]])) # true
print(canFinish(2, [[1,0]])) # true

# 802 Find Eventual Safe States
def eventualSafeNodes(graph):

    # 0 represents unvisited
    # 1 represents safe where that vertex will no lead to circle
    # 2 represents unsafe as it has circle

    def dfs(start):
        if color[start] == 2:
            return False
        if color[start] == 1:
            return True

        color[start] = 2 # visited

        for edge in graph[start]:
            if not dfs(edge):
                return False


        color[start] = 1
        return True

    color = [0] * len(graph) # 0 represents unvisited
    res = []
    for vertex in range(len(graph)):
        # if dfs does not find cycle, append cur vertex to result
        if dfs(vertex):
            res.append(vertex)

    return res

    # """
    # algorithm, be familiar with how to detect graph in directed cycle
    # 1, create directed graph, the may of making graph is different than usual
    # 2, visit all the edges of current vertex with dfs
    # 2, if vertex has no outgoing directed edges, return that vertex
    # 3, if there is no cycle from current vertex, return that vertex
    # """
    #
    # g = defaultdict(list)
    # for vertex in range(len(graph)):
    #     for edge in graph[vertex]:
    #         g[vertex].append(edge)
    #
    # def dfs(v):
    #     # visiting currnet vertex at this very moment
    #     visited.add(v)
    #     each_v[v] = True
    #     for edge in g[v]:
    #         # 0:[1,2], doesnt get to 2 cuz 1 will return true as 1 is alerady visited
    #         if edge not in visited:
    #             dfs(edge)
    #
    #         # detect cycle
    #         # in first test case, 0:[1,2] 2 wont be visited cuz 1 will return true
    #         if each_v[edge]:
    #             return True
    #
    #     # when current vertex has no edges, reset
    #     # e,g, index 5 has no edge, yes is visited from index 2
    #     # so need to reset
    #     # vertex that has no outgoing edges will stay false
    #     # if it does, it will stay true as loop is detected
    #     each_v[v] = False
    #     return False
    #
    # visited = set()
    # each_v = [False for i in range(len(graph))]
    # res = []
    # for v in range(len(graph)):
    #     # if loop found, dont wannt add it to result
    #     # as the goal is to return non cycle vertex
    #     if not dfs(v):
    #         res.append(v)
    #
    # return res

meow = [[1,2],[2,3],[5],[0],[5],[],[]] # [2,4,5,6]
meow1 = [[],[0,2,3,4],[3],[4],[]] # [0,1,2,3,4]
meow2 = [[4,9],[3,5,7],[0,3,4,5,6,8],[7,8,9],[5,6,7,8],[6,7,8,9],[7,9],[8,9],[9],[]]
# [0,1,2,3,4,5,6,7,8,9]
print(eventualSafeNodes(meow))
print(eventualSafeNodes(meow1))
print(eventualSafeNodes(meow2))







