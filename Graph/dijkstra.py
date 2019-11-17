from collections import defaultdict
import heapq
def dijkstra(src, dst, edges):
    # return the shortest path of given src and dst
    # greedy algoritm, store all posibble solutions(edges out of current vertex)
    # and explore edges that has the smallest cost by heap
    # traverse graph bfs(not technically so kinda)
    # and keep track of minimum path by heap

    # Key is src(vertex) and in value’s dictionary, key is edge value is cost (weight)
    # {'A': {'C': 3, 'B': 4, 'E': 7}
    graph = defaultdict(dict)

    # create graph
    for cur_ver, edge, c in edges:
        graph[cur_ver][edge] = c

    # first_cost = float("inf") # TypeError: 'float' object is not iterable
    # cannot use inf, float('inf') + 2 = inf

    #### HEAP #####
    # when using heap, be careful of whether it is nested list or not
    # if you wanna pop it, should be nested list
    heap = [(0, src)]  # first verex has no cost
    # think of src as current vertex
    while heap:
        # think of s as current vertex
        # first index is cost and second one is vertex
        #print(heap)
        cost, s = heapq.heappop(heap)  # spit the vertex with smallest cost(path)

        # if current vertex is destination, return cost
        # currnet vertex has smallest path for sure
        # as heap only spits smallest path
        if s == dst:
            return cost

        # visit all the nodes of current vertex
        for edge in graph[s]: # {'A': {'C': 3, 'B': 4, 'E': 7}
            # print(graph[s]) # {C:3}
            # print(edge) # "C"
            heapq.heappush(heap, (cost + graph[s][edge],edge))

    return None

edges = [["A","C",3],["A","B",4],["A","E",7],["B","D",5],["B","C",6],
        ["C","D",11],["C","E",8],["E","D",2],["E","G",5],["D","G",10],
        ["D","F",2],["G","F",3]]

print()
print("dijkstra with dic")
print(dijkstra("A","F",edges))
print(dijkstra("A","D",edges))


# defaultdict with list

def dijkstra(src, dst, edges):
    graph = defaultdict(list)

    # create graph
    for cur_ver, edge, c in edges:
        graph[cur_ver].append([edge, c])

    heap = [(0, src)]  # first verex has no cost
    print(graph)
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
print(dijkstra(0, 4, edges))
print(dijkstra(0, 3, edges))
print(dijkstra(0, 5, edges))
print(dijkstra(0, 8, edges))
print()

# Vertex   Distance from Source (0)
# 0                0
# 1                4
# 2                12
# 3                19
# 4                21
# 5                11
# 6                9
# 7                8
# 8                14


from collections import defaultdict
import heapq

class Dijkstra:
    def __init__(self):
        self.g = defaultdict(dict)

    def add_Edge(self, src, edge, cost):
        self.g[src][edge] = cost

    def dijkstra(self, src, dst):

        heap = [(0, src)]
        # for s,d,c in edges: # create graph
        #     g[s][d] = c
        while heap:
            # if heap not used, we need to use for loop and find minimum cost every time which is expensive
            # this heap will pop out edge that has with smallest cost
            cost, s = heapq.heappop(heap)
            if s == dst: return cost
            for d in self.g[s]:  # d is edge of current vertex
                # this for loop spit out all edges out of cur_vtx
                # heap gets sum of current vertex weight
                # and edge's weight. and edge
                heapq.heappush(heap, (cost + self.g[s][d], d))

        return -1

d = Dijkstra()
d.add_Edge("A", "C", 3)
d.add_Edge("A", "B", 4)
d.add_Edge("A", "E", 7)
d.add_Edge("B", "D", 5)
d.add_Edge("B", "C", 6)
d.add_Edge("C", "D", 11)
d.add_Edge("C", "E", 8)
d.add_Edge("E", "D", 2)
d.add_Edge("E", "G", 5)
d.add_Edge("D", "G", 10)
d.add_Edge("D", "F", 2)
d.add_Edge("G", "F", 3)
print(d.g)
print(d.dijkstra("A", "F"))