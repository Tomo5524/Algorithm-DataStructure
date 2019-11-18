# Adjacency matrix.
# Adjacency list.

# BFS and DFS
from collections import defaultdict,deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        # # using deque, will be index out of range

    def addEdge(self, start, des):
        self.graph[start].append(des)

    def BFS(self, vertex):
        """
            Time Complexity O(V+E)
            algorithm
            1, create queue
            2, every edge out of current vertex should go into visited
                as they are already visited
            3,
        """
        visited = set()
        ### set is hashtable and looks better
        # visited = [False for i in range(len(self.graph))]
        res = []
        q = deque([vertex])
        # add first vertex, otherwise when this vertex shows up next, it is not visited
        # thus it will go into queue and there will be no duplicates
        visited.add(vertex)
        while q:
            cur_vtx = q.popleft()
            res.append(cur_vtx)
            for edge in self.graph[cur_vtx]:
                # every edge goes into visited
                if edge not in visited:
                    q.append(edge)
                    visited.add(edge)

        return res

    def DFS(self, vertex):  ## iterative DFS
        """
            Time Coplexity O(V+E)
            algorithm
            1, create stack
            2, node stems out of current vertex doest go into visited
                as they are not visited yet
                stack pops out the last element which is current vertex
                so that goes into viisted
        """

        # Use Set instead of list to keep track of visited vertices
        visited = set()
        stack = [vertex]
        res = []
        while stack:

            cur_vtx = stack.pop()
            # Stack may contain same vertex twice. So
            # we need to append the popped item only
            # if it is not visited.
            if cur_vtx not in visited:
                visited.add(cur_vtx)
                res.append(cur_vtx)
            for edge in self.graph[cur_vtx]:
                # current edge is not visisted yet so does not go into stack
                if edge not in visited:
                    stack.append(edge)

        return res

    # # recursive
    # def recursive_DFS(self,vertex):
    #
    #     res = []
    #     if not vertex in visited


### find if Find if there is a path between two vertices (start and des) in a directed graph
    """
        if there is a path between start and des, there is a path
        keep in mind that start does not have to come back to itself
        but cycle does
    
    """
    def path_bfs(self, start, des):
        """
            Time Coplexity O(V+E)
            algorithm,
            1, same as iterative bfs
            2, check is node is destination before it goes to visited

        """
        # use queue
        path = deque([start])
        visited = set()
        while path:
            cur_vertex = path.popleft()
            for edge in self.graph[cur_vertex]:

                if edge == des:
                    return True

                if edge not in visited:
                    visited.add(edge)
                    path.append(edge)

        return False

    def path_dfs(self, start, des):
        """
            Time Coplexity O(V+E)
            algorithm,
            1, same as iterative dfs
            2, check is node is destination before it move on to next node

        """
        # use queue
        path = [start]
        visited = set()
        #visited.add(start)
        while path:
            cur_vertex = path.pop()
            if cur_vertex not in visited:
                visited.add(cur_vertex)
            for edge in self.graph[cur_vertex]:

                if edge == des:
                    return True

                if edge not in visited:
                    path.append(edge)

        return False

    # detect cycle in directed cycle

    def find_cycle_iterative_dfs(self):
        """ recursion BFS
            Time Coplexity O(V+E)
            algorithm
            1, definition of cycle is that original should come back to itself,
            2, visit every vertex by for loop and if current vertex is already visited return True
            3, in bfs function, visite every node that stems out of current vertex
            4, if there is no cycle, update all the visited vertices as False
            3, if new node is already visited, return true as there is cycle

        :return:
        """

        def dfs(vertex,visited,each_vertex):
            # mark current vertex as visited
            visited.add(vertex)
            # mark current vertex as visited
            each_vertex[vertex] = True
            # visit all edges that stem out of current vertex
            for edge in self.graph[vertex]:

                # go all the way to the node that has no indirected edge
                if not edge in visited:
                    dfs(edge, visited, each_vertex)
                # if current vertex is already visited, cycle found
                if each_vertex[edge]:
                    return True

            # does it make the original vertex false too?
            # yes original vertex will be false too
            each_vertex[vertex] = False
            return False

        visited = set()
        each_vertex = [False for _ in range(len(self.graph))]
        for vertex in range(len(self.graph)):
            if dfs(vertex,visited,each_vertex):
                return True

        return False

g = Graph()

# False test case for detect cyle, but set length as global variable
# g.addEdge(1, 0)
# g.addEdge(0, 2)
# g.addEdge(1, 2)
# g.addEdge(2, 1)

# test case
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
print(g.BFS(2))
print(g.DFS(2))
print(g.path_bfs(2,0))
print(g.path_dfs(3,1))
print(g.find_cycle_iterative_dfs())


class OG:
    def __init__(self,n):
        self.graph = defaultdict(list)
        self.n = n

    def addEdge(self, start, des):
        self.graph[start].append(des)

    def toplogicalSort(self):
        # The first vertex in topological sorting is always a vertex with in-degree as 0 (a vertex with no incoming edges).
        """
        algorithm
        1, start off at 0, visit in order of numbers
        2, visit every vertex using dfs recursion
        3, when it reaches a vertex that has no incoming edge, appendleft it to stack
        4, do step 1 and step 2 along with keeping the track of visited
        5, vertcies that have no incoming edges come first

        """

        def dfs(vertex, visited, stack):
            visited.add(vertex)
            # visit all the way to the vertex that has no edge
            for edge in self.graph[vertex]:
                if edge not in visited:
                    dfs(edge, visited, stack)

            # current vertex is no incoming one here
            # and goes into the front of stack
            stack.appendleft(vertex) # O(1)

        visited = set()
        stack = deque([])
        for vertex in range(self.n):
            if vertex not in visited:
                dfs(vertex, visited, stack)

        return stack

g = OG(6)
g.addEdge(5, 2)
g.addEdge(5, 0)
g.addEdge(4, 0)
g.addEdge(4, 1)
g.addEdge(2, 3)
g.addEdge(3, 1)
print(g.graph)
print('meow')
print(g.toplogicalSort())


class UndirecetedGraph:
    def __init__(self,n):
        self.graph = defaultdict(list)
        self.n = n
        self.adjacencyMatrix = [[0 for _ in range(n)] for i in range(n)]

    def addEdge(self, start, des):
        self.graph[start].append(des)
        self.graph[des].append(start)

    def adjacency_matrix(self, u, v):
        self.adjacencyMatrix[u][v] = 1
        self.adjacencyMatrix[v][u] = 1

    def detect_cycle(self): # https://www.geeksforgeeks.org/detect-cycle-undirected-graph/
        """
        algorithm, Time complexity O(V+E)
        1, pretty much same as directed cycle detection
        2, keep track of parent and edge
        3, it they are not match, that means edge comes from the original parent
        """

        def dfs(vertex,visited,parent):

            visited.add(vertex)
            for edge in self.graph[vertex]:
                # if edge hasn't been visited yet, visit it
                if edge not in visited:
                    dfs(edge,visited,vertex) # this vertex represents the origianl vertex (parent)

                # if parent is different than edge, that means edge came from parent
                # often times, since it is undirected (two way nodes)
                # edge always goes back to its parent (where it came from)
                # e,g 0,1, 0 goes to 1 and 1 goes back 0
                # and 1 goes to 2, 2 goes to 0 and parent(1) and edge(0) are different
                elif parent != edge:
                    return True

            return False

        visited = set()
        for vertex in range(len(self.graph)):
            if vertex not in visited:
                if dfs(vertex,visited,-1): # this -1 represents where original index
                    return True

        return False

no_dire = UndirecetedGraph(5)
#no_dire = UndirecetedGraph()

no_dire.addEdge(1, 0)
no_dire.addEdge(0, 2)
no_dire.addEdge(2, 1)
no_dire.addEdge(0, 3)
no_dire.addEdge(3, 4)
print(no_dire.graph)
print(no_dire.detect_cycle())
print(no_dire.adjacencyMatrix)
no_dire.adjacency_matrix(1, 0)
no_dire.adjacency_matrix(0, 2)
no_dire.adjacency_matrix(2, 0)
no_dire.adjacency_matrix(0, 3)
no_dire.adjacency_matrix(3, 4)
print(no_dire.adjacencyMatrix)

# LeetCode

# 684. Redundant Connection

# An edge will connect two nodes into one connected component.
#
# When we count an edge in, if two nodes have already been in the same connected component,
# the edge will result in a cycle. That is, the edge is redundant.
# https://leetcode.com/problems/redundant-connection/discuss/123819/Union-Find-with-Explanations-(Java-Python)
# https://leetcode.com/problems/redundant-connection/discuss/331028/Best-Python-Solution-(Union-Find-Explained) # great explanation


def findRedundantConnection(edges): # detect loop and remove it
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

    def union(x,y):
        """
        1, find parent(root) of x and y
        2, if they are same, x and y are coming from the same parent
        3, if the are not just update it

        """
        x = find_parent(x) # x repersents parent(original vertex)
        y = find_parent(y) # y represents edge
        if x == y:
            return True
        # For every edge, we unify u and v. #[1]
        # Which means u and v and all of their parents all lead to one root. [1,2] [1,3] 1 to 3 are all connected
        parent[x] = y

    parent = [0 for i in range(len(edges))]
    for u,v in edges: # u is vertex v is edge
        if union(u-1,v-1): # -1 cuz index and list number don't match
                            # first number starts off with 1 so -1 to adjust to index
            return [u,v]

    return None

import heapq
def dijkstra(src, dst, edges):
    graph = defaultdict(list)

    # create graph
    for cur_ver, edge, c in edges:
        graph[cur_ver].append([edge, c])

    heap = [(0, src)]  # first verex has no cost
    #print(graph)
    while heap:

        cost, s = heapq.heappop(heap)  # spit the vertex with smallest cost(path)

        if s == dst:
            return cost

        # visit all the nodes of current vertex
        # c denotes cost
        for edge,c in graph[s]:  # {0: [(1, 4), (7, 8)]}
            heapq.heappush(heap, (cost + c, edge))

    return None

edges = [[0, 1, 4], [0, 7, 8], [1, 2, 8], [1, 7, 11], [7, 8, 7], [7, 6, 1],
         [2, 3, 7], [2, 8, 2], [2, 5, 4], [6, 8, 6], [6, 5, 2], [3, 4, 9],
         [3, 5, 14], [5, 4, 10]]

print()
print("dijkstra with list")
print(dijkstra(0, 4, edges)) # 21
print(dijkstra(0, 3, edges)) # 19
print(dijkstra(0, 5, edges)) # 11
print(dijkstra(0, 8, edges)) # 13
print()

from collections import defaultdict
import heapq


def prim(graph, n,start):
    g = defaultdict(list)
    #heap = []
    for u, v, c in graph:
        # get all heap
        #heap.append((c, v))
        g[u].append((v, c))

    # how to srart off from the smallest cost
    #heapq.heapify(heap)
    visited = set()
    heap = [(0,start)]
    res = 0
    while len(visited) < n:
        cost, src = heapq.heappop(heap)
        if src not in visited:
            res += cost
            visited.add(src)
            for edge, c in g[src]:
                if edge not in visited:
                    heapq.heappush(heap, (c, edge))

    return res

edges = [['a', 'b', 2],['a', 'c', 3],['b', 'd', 3],
        ['b', 'c', 5], ['b', 'e', 4], ['c', 'e', 4],
        ['d', 'e', 2], ['d', 'f', 3], ['e', 'f', 5]]

n = 6
print(prim(edges, n,"a")) # 13

# does not give the right answer as 1:[2,8] and 0:[7,8] same cost and choose 2 which leads to wrong answer
edges1 = [[0, 1, 4], [0, 7, 8], [1, 2, 8], [1, 7, 11], [7, 8, 7], [7, 6, 1],
         [2, 3, 7], [2, 8, 2], [2, 5, 4], [6, 8, 6], [6, 5, 2], [3, 4, 9],
         [3, 5, 14], [5, 4, 10]]
n1 = 9
print(prim(edges1,n1,0)) # 37