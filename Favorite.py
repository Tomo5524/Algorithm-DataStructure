
# 11/18/2019
# 1165. Single-Row Keyboard
def calculateTime(keyboard,word):
    """
    algorithm
    1, create hash map for keyboard
    2, get distance from cur_word to previous word using absolute
        e,g, 'cba', c moves from 0 to 2, so curmax = 2,
                    b moves from 2 to 1 so cur_max = 3
                    a moves from 1 to 0 so cur_max = 4
    """

    dic = {}
    for i in range(len(keyboard)):
        dic[keyboard[i]] = i

    res = 0
    pre = 0
    for j in range(len(word)):
        cur = pre - dic[word[j]]
        res += abs(cur)
        pre = dic[word[j]]

    return res

keyboard = "abcdefghijklmnopqrstuvwxyz"
word = "cba"
print(calculateTime(keyboard,word))
keyboard1 = "pqrstuvwxyzabcdefghijklmno"
word1 = "leetcode"
print(calculateTime(keyboard1,word1))
print()

# collection of questions
from collections import defaultdict
import heapq
def minCostToSupplyWater(n, wells, pipes):
    """
    key is to create right graph and use prim algorithm
    be aware of the diffreence between prim and dikstra
    algorithm
    1, create undirected(bidirectional) graph,
    2, wells themselves can be pipe so create heap with wells as they can be potential answers
        in test cast 3, all the houses will get wells
    3, and do prim
    """
    g = defaultdict(list)
    for u,v,c in pipes:
        g[u].append((v,c))
        g[v].append((u,c))

    heap = [(wells[i],i+1) for i in range(len(wells))]
    # heap = []
    # for i in range(len(wells)):
    #     heap.append((wells[i],i+1))

    # hepify O(n)
    # without heapify, we cannnot start off with the smallest well
    # and fail test 2
    heapq.heapify(heap)
    visited = set()
    res = 0
    # keep visiting until we visit all the nodes
    while len(visited) < n:

        cost,src = heapq.heappop(heap)
        # if current node is not visited yet, add it to visited
        # along with adding up current cost and accumulative cost
        if src not in visited:
            res += cost
            visited.add(src)

            # visit edges of current node
            for edge,c in g[src]:
                # dont add up cost as we are tyring to figure out the smallest path from current node to neighbours
                heapq.heappush(heap,(c,edge))

    return res

n = 3
wells = [1, 2, 2]
pipes = [[1, 2, 1], [2, 3, 1]]
# print(minCostToSupplyWater(n, wells, pipes)) # 3
n1 = 5
wells1 = [46012,72474,64965,751,33304]
pipes1 = [[2,1,6719],[3,2,75312],[5,3,44918]]
# print(minCostToSupplyWater(n1, wells1, pipes1)) # 131704
n2 = 9
wells2 = [58732,77988,55446,79246,8265,30789,39905,79968,61679]
pipes2 = [[2,1,45475],[3,2,41579],[4,1,79418],[5,2,17589],[7,5,4371],[8,5,82103],[9,7,55500]]
# print(minCostToSupplyWater(n2, wells2, pipes2)) # 362782
n3 = 3
wells3 = [1, 2, 2]
pipes3 = [[1, 2, 1000], [2, 3, 2221]] # 5
print(minCostToSupplyWater(n, wells, pipes))
print(minCostToSupplyWater(n1, wells1, pipes1))
print(minCostToSupplyWater(n2, wells2, pipes2))
print(minCostToSupplyWater(n3, wells3, pipes3))
print()

# 11/10/2019
# 939. Minimum Area Rectangle

def minAreaRect(points):
    min_area = float("inf")
    points_table = set()

    for x, y in points:
        points_table.add((x, y))

    for x1, y1 in points:
        for x2, y2 in points:

            # Skip looking at same point
            # make sure if current coordinate is valid or not
            if x1 > x2 and y1 > y2:
                # passing this if means x1,y1 x2,y2 are valid
                # e,g, test2 1,3 and 3,4

                # now we check all 4 coordinates
                # x1,y2 checks upper right, x2,y1 checks upper left
                # if x1,y2 and x2,y1 are in points table, they are valid
                if (x1, y2) in points_table and (x2, y1) in points_table:
                    # now we will find out area of rectangle
                    area = abs(x1 - x2) * abs(y1 - y2)
                    if area:
                        min_area = min(area, min_area)

    return 0 if min_area == float('inf') else min_area


test = [[0, 1], [1, 3], [3, 3], [4, 4], [1, 4], [2, 3], [1, 0], [3, 4]] # 2
test1 = [[1,1],[1,3],[3,1],[3,3],[2,2]] # 4
test2 = [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]] # 2

print(minAreaRect(test))
print(minAreaRect(test1))
print(minAreaRect(test2))

# [[0,1],[1,3],[3,3],[4,4],[1,4],[2,3],[1,0],[3,4]]
#

# 11/9/2019
# 947. Most Stones Removed with Same Row or Column

from collections import defaultdict
def removeStones(stones):

    # time complexity is O(V+E)
    """
    key is The key point is here, we define an island as number of points that are connected by row or column.
    how to create map is the key
    algorithm
    1, create dictionary for row and col
    2, what we want to return is the islands that are connected by row and column
    3, length of stones - islands is the answer
    4, how to move from current cell to another is the key
    5, row only moves horizontally and col moves vertically
    e,g, [[0,0],[0,2],[1,1],[2,0],[2,2]], 0,0 -> 0,2 by row -> 2,2 by col -> 2,0 by row
    """
    def dfs(i, j):
        visited.add((i, j))
        # move horizontally
        for r in row[i]:
            if (i, r) not in visited:
                dfs(i, r)

        # move vertically
        for c in col[j]:
            if (c, j) not in visited:
                dfs(c, j)

    row = defaultdict(list)
    col = defaultdict(list)
    for x, y in stones:
        row[x].append(y)
        col[y].append(x)

    visited = set()
    cnt = 0
    for i, j in stones:
        if (i, j) not in visited:
            dfs(i, j)
            cnt += 1

    return len(stones) - cnt

test = [[0,1],[1,0],[1,1]]
test1 = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
test2 = [[0,0],[0,2],[1,1],[2,0],[2,2]]
test3 = [[0, 0]]
test4 = [[0,1],[1,0]]

# print(removeStones(test))
print(removeStones(test1))
print(removeStones(test2))
print(removeStones(test3))
print(removeStones(test4))
print()


# 1188. Design Bounded Blocking Queue

from collections import deque
import threading
class BoundedBlockingQueue():
    def __init__(self, capacity):
        self.queue = deque()
        self.cur = 0
        self.ept, self.ful = threading.Semaphore(0), threading.Semaphore(capacity)

    def enqueue(self, element):
        self.ful.acquire()
        self.queue.appendleft(element)
        self.cur += 1
        self.ept.release()

    def dequeue(self):
        self.ept.acquire()
        temp = self.queue.pop()
        self.cur -= 1
        self.ful.release()
        return temp

    def size(self):
        return self.cur

# from collections import deque
# class BoundedBlockingQueue:
#
#     def __init__(self, capacity: int):
#         self.capa = capacity
#         self.q = deque([])
#         self.waiting = deque([])
#         self.waiting_F = False
#         self.consumer_F = False
#
#     def enqueue(self, element: int) -> None:
#         if self.consumer_F:
#             self.consumer_F = False
#             return element
#
#         if len(self.q) >= self.capa:
#             self.waiting.append(element)
#             self.waiting_F = True
#
#         else:
#             self.q.append(element)
#
#         return ''
#
#     def dequeue(self) -> int:
#
#         if len(self.q):
#             popped = self.q.popleft()
#             if self.waiting_F:
#                 self.q.append(self.waiting.popleft())
#                 self.waiting_F = False
#
#             return popped
#
#         self.consumer_F = True
#         return ''
#
#     def size(self) -> int:
#         return len(self.q)

queue = BoundedBlockingQueue(2) #;   // initialize the queue with capacity = 2.
queue.enqueue(1)# ;   // The producer thread enqueues 1 to the queue.
# print(queue.dequeue())# ;    // The consumer thread calls dequeue and returns 1 from the queue.
# print(queue.dequeue())# ;    // Since the queue is empty, the consumer thread is blocked.
# print(queue.enqueue(0))# ;   // The producer thread enqueues 0 to the queue. The consumer thread is unblocked and returns 0 from the queue.
# print(queue.enqueue(2))# ;   // The producer thread enqueues 2 to the queue.
# print(queue.enqueue(3))# ;   // The producer thread enqueues 3 to the queue.
# print(queue.enqueue(4)  )# ;   // The producer thread is blocked because the queue's capacity (2) is reached.
# print(queue.dequeue())# ;    // The consumer thread returns 2 from the queue. The producer thread is unblocked and enqueues 4 to the queue.
# print(queue.size())# ;
print()

# 11/3/2019
# 1055. Shortest Way to Form String
from collections import defaultdict
import bisect
def shortestWay(source, target):
    """
    key is to use hashmap and how and what to store
    and increment position every iteration as we look for next letter in next iteration
    algorihtm
    1, first create dictionary (inverted index) meaning, key is letter and values are the indices that appear
        e,g, test5, {a:[0,3]}
    2, if current letter exists in dictionary, perform binary search,
        look for current letter's index in source
        so perform bisect left and returned index is the index we are looking for
    3, when source counter is greater than length of source or the last value in current letter's list
        get the smallest index of current letter
    """
    # time complexity O(M + N*logM)
    dic = defaultdict(list)
    for i,val in enumerate(source):
        dic[val].append(i)

    s_pos = 0
    minimum = 1
    # iterate through each letter in target
    for letter in target:
        #if letter does not exist, return -1
        # if this statement comes after positions,
        # when current letter doesnt exist,
        # dictionary gets current letter with empty list
        # so later causes error
        if letter not in dic:
            return -1

        # get current letter's list (indices)
        positions = dic[letter]
        # if we cannot make subsequence from current position,
        # start over and get the smallest index of current letter
        # with out second if statement, in test 5 when we reach c, it will go out of boundary
        # as s_pos is 4 which is out of range in c {c:[0]}
        if s_pos >= len(source) or s_pos > positions[-1]:
            s_pos = positions[0] + 1 # O(1)
            minimum +=1

        else:
            # get the closest letter from current index in source
            loc = bisect.bisect_left(positions,s_pos)
            s_pos = positions[loc] + 1


    return minimum

    # brute force, time complexity O(source*target)
    source_p = 0
    # starts off with 1 cuz if source can make up target, minimum is at least 1 letter
    minimum = 1
    for letter in target:

        # find leftmost letter, starting at source_p
        # search starts from source pointer
        source_p = source.find(letter,source_p)

        # if letter not found from current pointer to end of source
        if source_p == -1:
            # search letter from the beginning
            source_p = source.find(letter)

            # if letter not found, return False
            if source_p == -1:
                return -1

            # one subsequence ends so increment counter
            minimum +=1

        # if letter found, we want to compare next word in next iteration so increment counter
        source_p +=1

    return minimum

source = "abc"
target = "abcbc"
source1 = "abc"
target1 = "acdbc"
source2 = "xyz"
target2 = "xzyxz"
s3 = "adbsc"
t3 = "addddddddddddsbc"
s4 = "aaaaa"
t4 = "aaaaaaaaaaaaa"
s5 = "abcab"
t5 = "aabbaac"

# print(shortestWay(source,target))
# print(shortestWay(source1,target1))
# print(shortestWay(source2,target2))
print(shortestWay(s3,t3))
print(shortestWay(s4,t4))
print(shortestWay(s5,t5))
print()


from collections import defaultdict
# 792. Number of Matching Subsequences
def numMatchingSubseq(S, words):

    """
    algorithm,
    1, create dictionary whose key is first word in each word
        and append all the words that share the first element together
        e,g, ["a", "bb", "acd", "ace"], {a:[a,acd,ace]}
    2, iterate through each character in the given string, and if current character is in dictionary
        for each character, access the dictionary to retrieve the list as we iterate over the list
        if cur word is 1 length, we finished subsequence, so increment counter
        otherwise, slice off the first character and add the sliced word back to the dictionary.
        e,g, after first element a, {c:[cd,ce]}
    3, do not forget to empty current letter (from given string) list,
        otherwise when we go through b, dictionary[b] gets single b and list will go over this b
        and count as subsequence which is not true
    """

    # time complexity: O(S+words)

    # create dictionary whose key is first word in each word
    dic = defaultdict(list)
    for word in words:
        dic[word[0]].append(word)

    res = 0
    for letter in S:
        # get the list whose key is current letter in given string
        waiting_words = dic[letter]
        # reset curent letter's list
        dic[letter] = []
        for cur_word in waiting_words:
            if len(cur_word) == 1:
                res+=1

            else:
                # slice off the first character and add the sliced word back to the dictionary
                # with key being the second letter in cur_word
                dic[cur_word[1]].append(cur_word[1:])

    return res


S = "abcde"
words = ["a", "bb", "acd", "ace"]
print(numMatchingSubseq(S,words))
print()

# 11/2/2019
# 1087. Brace Expansion
# key is to loop over target sub array
# e,g, "{a,b}c{d,e}f"
# 1st loop would be {a,b} and second c{d,e}f
# should return lexicographical order

def expand(S):
    """
    # key is to loop over target sub array, e,g, "{a,b}c{d,e}f"
    # 1st loop would be c{d,e}f and second is just f
    key is to get rid of comma(,) and how to backtrack
    and what should be the base case and how to end current dfs
    algorithm
    1, find open bracket, it current list starts off with open bracket, get the first element in bracket
    2, if current list's first element is not bracket, just append first element pass in list after the first element
    3, return result in lexicographical order.
    """

    def dfs(S,ans):

        # base case
        if not S:
            res.append(ans)
            return

        # if first element is open bracket, chop off this sub list
        if S[0] == "{":
            # get end bracket
            end = S.find("}")

            # loop over current sub list from open to close
            # this starts off with 1 cuz at this point we know first element is {
            for letter in S[1:end]:

                # chop off sublist and next sub list starts out right after close bracket
                dfs(S[end+1:],ans+letter)

        # if first element in current list is not open {
        # just get the first element and pass in sub list that starts out with after first value
        else:
            dfs(S[1:],ans+S[0])

    S = S.replace(',','')
    res = []
    dfs(S,"")
    # return lexicographical order
    res.sort()
    return res

meow = "{a,b}c{d,e}f" # ["acdf","acef","bcdf","bcef"]
meow1 = "abcd"
meow2 = "{a,b}{z,x,y}"
print(expand(meow))
print(expand(meow1))
print(expand(meow2))
print()

# 11/1/2019
# 1146. Snapshot Array

from collections import defaultdict
import copy
import bisect
class SnapshotArray:
    ## key is how to store values and where to insert key
    """
    we want to get the latest snapshot in current snapID
    # how to store values, and how to access them
    eveyry time snap called, snapID increments so
    if values share the same snapID, it will be updated and previous one will not be called
    e,g, obj.set(0,4), obj.set(0,16), obj.set(0,13), gets 13
    3 nested list, and how to store each value
    algorithm
    1, we create sub arrays for every index
    2, append values according to given index and snapid
    3, in each sub array in each index, first value is snapID and second value would be snapID
    4,
    """

    def __init__(self, n):
        # starts off with dummy sub list for edge case when there is only one value
        # we wanna store snapshot(1st list) in current index(2nd list) and append current SnapID and val in current index
        self.arr = [[[0,-1]] for _ in range(n)]
        self.cur_snapID = 0

    def set(self, index, val):
        # append curID and val in current index
        # current index array is a nested list.
        self.arr[index].append([self.cur_snapID, val])

    def snap(self):
        # just increment snap
        self.cur_snapID += 1
        return self.cur_snapID - 1

    def get(self, index, snap_id):

        # loc = bisect.bisect_right(self.arr[index], [snap_id+1])
        # return self.arr[index][loc-1][1]

        l,r = 0,len(self.arr[index])-1
        # search insert point
        while l <= r:
            mid = (l+r) // 2
            # snap ID +1 cuz that is where we want to insert value
            if self.arr[index][mid][0] < snap_id+1:
                l = mid + 1

            else:
                r = mid - 1

        # loc = l
        # a[1][0][0] = 2
        # a[0][1][1] IndexError: list index out of range
        # first index indicates index 0 out of whole list
        # second index denotes first index's list
        # third index denotes current list
        # a[3][1][1] = 22

        # -1 cuz we want to get the value right before insert location
        # else 0 needed for edge case for 3,
        # a = [[[1, -1]], [[2, -1]], [[3, -1]], [[4, -1],[11,22]]]

        return self.arr[index][l-1][1] if l else 0

        # brute force
        # search insert point by for loop
        # for i,shot in enumerate(self.A[index]):
        #     if shot[0] > snap_id:
        #         return self.A[index][i-1][1]
        #
        # return self.A[index][i][1]

# obj = SnapshotArray(3)
# obj.set(0,5)
# print(obj.snap())
# obj.set(0,6)
# print(obj.get(0,0))
## print(obj.snap())
## print(obj.get(0,0))

# obj = SnapshotArray(1)
# obj.set(0,4)
# obj.set(0,16)
# obj.set(0,13)
# print(obj.snap())
# print(obj.get(0,0))
# print(obj.snap())

# test case 3
obj = SnapshotArray(4)
print(obj.snap())
print(obj.snap())
print(obj.get(3,1))
obj.set(2,4)
print(obj.snap())
obj.set(1,4)
# ["SnapshotArray","snap","snap","get","set","snap","set"]
# [[4],[],[],[3,1],[2,4],[],[1,4]]
print()

# ["SnapshotArray","set","set","set","snap","get","snap"]
# [[1],[0,4],[0,16],[0,13],[],[0,0],[]]
# [null,null,null,null,0,13,1]


# 10/31/2019
from collections import defaultdict
import bisect
class TimeMap:
    """
    key is how to store times and values
    and how to find out time we want to return
    it is just merely to store times and value in a correspoding way
    algorithm
    1, store time and values separately so they are stored at the same index
        e,g, {love:[high,low]
             {love:[10,20]
    2, we want to return timestamp that is at -1 index of where timestamp is inserted
        e,g, timestamp = 15, should be inserted at 1 but since it donesnt exist,
        return 10 whihc is [i-1]
    3, we can use binary search since the given times are increasing meaning they are sorted
    """

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.times = defaultdict(list)
        self.values = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:

        self.times[key].append(timestamp)
        self.values[key].append(value)


    def get(self, key: str, timestamp: int) -> str:
        # # bisect will get the location of insert timestamp
        # so we want to return value that is located insert position -1
        loc = bisect.bisect_right(self.times[key],timestamp)
        return self.values[key][loc-1] if loc else ""
        """
        key is to figure out low and high.
        and where exactly insert point ends up
        """
        # performe normal binary serach insert
        # we are trying to find out current key's timestamp
        l,r = 0,len(self.times[key])-1
        while l <= r:
            mid = (l+r) // 2
            if self.times[key][mid] == timestamp:
                return self.values[key][mid]

            if self.times[key][mid] < timestamp:
                l = mid + 1

            else:
                r = mid - 1

        return self.values[key][l-1] if l > 0 else ''

        # brute force
        # if self.times[key][0] > timestamp:
        #     return ''
        #
        # for i, time in enumerate(self.times[key]):
        #     if time > timestamp:
        #         return self.values[key][i - 1]

        # # when timestamp is largerer than any values
        # return self.values[key][i]


kv = TimeMap()

# kv.set("foo", "bar", 1) # ; // store the key "foo" and value "bar" along with timestamp = 1
# print(kv.get("foo", 1)) # ;  // output "bar"
# print(kv.get("foo", 3)) # ; // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"
# kv.set("foo", "bar2", 4)
# print(kv.get("foo", 4)) # ; // output "bar2"
# print(kv.get("foo", 5)) # ; //output "bar2"

# kv.set("love", "high", 10)
# kv.set("love", "low", 20)
# print(kv.get("love", 5))
# print(kv.get("love", 10))
# print(kv.get("love", 15))
# print(kv.get("love", 20))
# print(kv.get("love", 25))

kv.set("ljfvbut","tatthnvvid",3)
print(kv.get("ljfvbut",4))
print(kv.get("ljfvbut",5))
print(kv.get("ljfvbut",6))
print(kv.get("ljfvbut",7))
kv.set("eimdon","pdjbnnvje",8)
print(kv.get("eimdon",9))
print(kv.get("eimdon",10))
print()

# 10/30/2019

# 1197. Minimum Knight Moves
from collections import deque
def minKnightMoves(n,k,x,y):

    dp = [[0 for _ in range(n)] for i in range(n)]
    directions = [(-1,2),(1,-2),(2,-1),(-2,1),(-1,-2),(1,2),(-2,-2),(2,2)]
    q = deque([(0,x,y)])
    moves = 0
    while q and moves < k:
        land,r,c = q.popleft()
        for i,j in directions:
            row = r + i
            col = c + j

            if 0 <= row < n and 0 <= col < n:
                q.append((land+1,r,c))
        moves += 1

    ans = land / (8 ** k)
    return ans

n = 3
k = 2
print(minKnightMoves(n,k,0,0))

# 1219. Path with Maximum Gold

def getMaximumGold(grid):
    """
    algorithm
    key is to return current max
    1, visit each cell except cell's value is 0
    2, to make it space O(1), mark visit cell as 0 so we will not visit again
    3, get temporary value before mark current cell, and after getting max_path out of current cell
        place it back to original value
    """
    # time complexity  O(m * n) where m = no_rows, n = no_cols
    # space O(1)
    def dfs(grid, r, c, cur_max):

        if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]) or grid[r][c] == 0:
            # return current max cuz if it returns 0, gold will get 0 so cur_max will not be passed on
            return cur_max

        # get current max
        cur_max += grid[r][c]
        # get current cell value
        temp = grid[r][c]
        # mark it as visited
        grid[r][c] = 0
        # get max value from current cell
        gold = max(dfs(grid, r + 1, c, cur_max),
                   dfs(grid, r - 1, c, cur_max),
                   dfs(grid, r, c + 1, cur_max),
                   dfs(grid, r, c - 1, cur_max))

        # place original value back to current cell
        grid[r][c] = temp
        return gold

    max_gold = 0
    # visit each cell
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            # will not visit current cell, unless it is greater than 0
            if grid[r][c] > 0:
                max_gold = max(max_gold, dfs(grid, r, c, 0))

    return max_gold

meow = [[0,6,0],
        [5,8,7],
        [0,9,0]]

print(getMaximumGold(meow))
print()

# 1110. Delete Nodes And Return Forest
class Node:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

def delNodes(root, to_delete):

    # question basically boils down to return root that has no parent
    """
    algorithm,
    1, traverse tree in inorder fashion
    1, when root has no parent and is not in delted, append it to result
    2, We only need to add the root node of every tree.

    """
    # def traverse(root,parent):
    #
    #     # base case
    #     if not root:
    #         return None
    #
    #     # when we see node that is to be deleted, children gets no parent so make parent false
    #     if root.val in to_delete:
    #         parent = False
    #         root.left = traverse(root.left,parent)
    #         root.right = traverse(root.right,parent)
    #         # delete curret node
    #         return None
    #
    #     # if current node is not in deleted, dfs
    #     else:
    #         # if current node has no parent, append it to result
    #         if not parent:
    #             res.append(root.val)
    #
    #         # make parent true at this point, we know current node's parent exists
    #         parent = True
    #         root.left = traverse(root.left, parent)
    #         root.right = traverse(root.right, parent)
    #         return root
    #
    # res = []
    # parent = False
    # # make lookup O(1)
    # to_delete = set(to_delete)
    # traverse(root,parent)
    # return res


    """
    key is to create new list when encountering target and don't append target

    algorihtm
    1, traverse tree in inorder fashion
    2, start off with nested list, so appending value becomes easier
    2, if we see target, create new list and go on and find its children
        since we created new list, its children will become rooot
    3,
    """

    def traverse(root):

        if not root:
            return None

        if root.val in to_delete:
            #res[-1].append(None)
            if root.left:
                res.append([])
                traverse(root.left)

            if root.right:
                res.append([])
                traverse(root.right)

            #root = None

        else:

            if not res:
                res.append([root.val])
            else:
                res[-1].append(root.val)

            if root.left:
                traverse(root.left)
            if root.right:
                traverse(root.right)

    res = []
    traverse(root)
    return res

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)
to_delete = [1,5]
print(delNodes(root,to_delete))
print()



# 10/29/2019
# Campus Bikes
# 1057. Campus Bikes

import heapq

def assignBikes(workers, bikes):

    """
    key is to keep smallest distance at the end in each worker sub array
    1, get every distance from every worker to every bike
    2, and keep smallest distance at the end by reversing current worker subarray
    3, create heap which has only smallest distance for each worker
    4, pop heap and if bike is alredy taken, get next small for the current worker
    """
    distances = []
    for w,(w_row,w_col) in enumerate(workers):
        distance = []
        for b,(b_row,b_col) in enumerate(bikes):
            # get distance
            dst = abs(w_row - b_row) + abs(w_col - b_col)
            distance.append((dst,w,b))

        distance.sort(reverse=True) # sort current worker and reverse it
        distances.append(distance) # smallest distance is at the end in each worker list

    # magic begins!!!
    # get the smallest distance from each worker
    # and length of heap equals to length of workers
    heap = [distances[i].pop() for i in range(len(workers))]
    # has to sort otherwise in test case 1,worker 0 gets bike 0
    # [3,0,0], [2,1,0]
    heap.sort()
    # dont use res = [], cuz appending is not suitable for this situation
    # as goal is to worker[i] gets bike in smallest distance
    # return ans where ans[i] is the index (0-indexed) of the bike that the i-th worker is assigned to.
    res = [-1] * len(workers)
    used_bikes = set()
    # keep going untile all workers get bikes
    while len(used_bikes) < len(workers):
        dst,w,b = heapq.heappop(heap)
        if b not in used_bikes:
            res[w] = b
            used_bikes.add(b)

        # if bike is already taken, get current workers next closest bike
        else:
            # magic begins!!!
            # get the next smallest distance from current worker's choices
            #
            heapq.heappush(heap,(distances[w].pop()))

    return res

    # brute force
    # takeaway is that res.append doesn't work but res * [0] = len(workers) works
    # get distance
    # distances = []
    # # we need parenthesis since it just has 2 values, index and worker's coordinate in list
    # for w, (r, c) in enumerate(workers): # ValueError: not enough values to unpack (expected 3, got 2)
    #     for b, (x, y) in enumerate(bikes):
    #         dst = (abs(r - x) + abs(c - y))
    #         # build a heap with the distance from every bike to every worker. So each push/pop is O(log wb) = O(log w) + O(log b).
    #         # too expensive
    #         distances.append((dst, b, w))
    #
    # distances.sort()
    #
    # # workers_seen = set()
    # # res = []
    # # these don't work for # w = [[0,0],[2,1]], b = [[1,2],[3,3]]
    # # because first value in res will be 0 as smallest ditance 2 is worker 1 and bike 0
    # # if b not in used_bike and w not in workers_seen:
    #
    # used_bike = set()
    # res = [-1] * len(workers)
    #
    # for dst,b,w in distances:
    #     if b not in used_bike and res[w] == -1:
    #         used_bike.add(b)
    #         res[w] = b
    #
    #
    # return res

workers = [[0,0],[2,1]]
bikes = [[1,2],[3,3]]
print(assignBikes(workers,bikes))
workers1 = [[0,0],[1,1],[2,0]]
bikes1 = [[1,0],[2,2],[2,1]]
print(assignBikes(workers1,bikes1))

# 10/28/2019

# 1007. Minimum Domino Rotations For Equal Row
def minDominoRotations(A, B):
    # edge case
    # all values are same, one element,
    """
    key is to rotate when cur_val in the row is not target
    e,g, A1 = [1,2,1,1,1,2,2,2]
        B1 = [2,1,2,2,2,2,2,2] t = 2, A1 has to rotate,
        first index, rotate
    if rotation is possible and answer is not rotate a, answer is rotate be for sure
    return minimum number that can mae all the same elements in a row
    algorithm
    1, to be a valid answer, both or one of the dice has to be target, otherwise return false
    2, 3 possible solutions, Dice A can be all the same value or Dice B can, or neither of them can
    3, pick the first number, and see if it can lead to the answer
    ## we don't actually rotate anything, just find out how many time it needs to virtually flip it to get same numbers in either row
    """

    def helper(A,B,t):
        # traverse list A
        rotation_a = 0
        rotation_b = 0

        for i in range(len(A)):
            # neither of them are target so possible to make all the element in one row
            if A[i] != t and B[i] != t:
                return -1

            ### at this point, we know that either of values has to be target,
            ### otherwise the first if statement takes place

            # if cur_val in A doesn't equal to target, virtually rotate it by incrementing counter_a
            # if both values are same, dont rotate (increment counter)
            elif A[i] != t:
                rotation_a += 1

            # if cur_val in B equals to target, virtually rotate it by incrementing counter_B
            # if both values are same, dont rotate (increment counter)
            elif B[i] != t:
                rotation_b += 1

        return min(rotation_a,rotation_b)

    rotate_check = helper(A,B,A[0])
    # if returned value is -1, impossible to make the row filled with same values
    # If first element B could make all elements in A or B return rotate_check
    # if first element in B could make all elements in A or B call helper function with B
    return rotate_check if rotate_check != -1 else helper(A,B,B[0])
    # helper(A,B,B[0]) needed here for when first value in A cannot be the answer
    # e,g A1 = [1,2,1,1,1,2,2,2]
    #     B1 = [2,1,2,2,2,2,2,2]

A = [2,1,2,4,2,2]
B = [5,2,6,2,3,2]

# return minimum
A1 = [1,2,1,1,1,2,2,2]
B1 = [2,1,2,2,2,2,2,2]

A2 = [1,1,1,1,1,1,1,1]
B2 = [1,1,1,1,1,1,1,1]

# edge case
A3 = [2]
B3 = [2]

A4 = [3,5,1,2,3]
B4 = [3,6,3,3,4]
print(minDominoRotations(A,B))
print(minDominoRotations(A1,B1))
print(minDominoRotations(A2,B2))
print(minDominoRotations(A3,B3))
print(minDominoRotations(A4,B4))

# 1170. Compare Strings by Frequency of the Smallest Character

def numSmallerByFrequency(queries, words):

    """
    key is to get number of alphabetically smallest letters from each sub array
    from both query and words
    algorithm
    1, count number of smallest letter for each word
    2, get answer by subtract each frequency in query with each frequency in word
    """

    def helper_cnt(s):
        # smallest letter is at front
        smallest = sorted(s)[0]
        # get number of frequency of smallest letter
        return s.count(smallest)

    # this will get smallest letter frequency
    query = [helper_cnt(q) for q in queries]
    word = [helper_cnt(w) for w in words]
    res = []
    for q in query:
        cnt = 0
        for w in word:
            # if word frequency is greater than q, increment counter
            if q < w:
                cnt +=1

        res.append(cnt)

    return res

q = ["cbd"]
words = ["zaaaz"]
qu = ["bbb","cc"]
wo = ["a","aa","aaa","aaaa"]
print(numSmallerByFrequency(q,words))
print(numSmallerByFrequency(qu,wo))

from collections import defaultdict,deque
# https://leetcode.com/discuss/interview-question/340230/
import heapq
class Logger:
    # time complexity
    # Start: O(1)
    # End: O(lgN)
    # Print: O(K)
    def __init__(self):
        self.start_hash = {}
        #self.end_hash = defaultdict(list)
        self.heap = []

     # * When a process starts, it calls 'start' with processId and startTime.
    def start(self,processId,startTime):
        if processId not in self.start_hash:
            self.start_hash[processId] = startTime

        #else:


    # * When the same process ends, it calls 'end' with processId and endTime.
    def end(self,processId,endTime):
        if processId in self.start_hash:
            heapq.heappush(self.heap,(processId,self.start_hash[processId],endTime))
            del self.start_hash[processId]


    # * Prints the logs of this system sorted by the start time of processes in the below format
    # 	* {processId} started at {startTime} and ended at {endTime}
    def Print(self):
        while self.heap:
            id,start,end = heapq.heappop(self.heap)
            print(id, "started at ",start, "and ended at ",end)


# Example:
log = Logger()
log.start("1", 100)
log.start("2", 201)
log.end("2", 102)
log.start("3", 103)
log.end("1", 104)
log.end("1", 108)
log.end("3", 105)
log.Print()
log.start("2", 101)
log.end("2", 102)
log.Print()

# Output:
# 1 started at 100 and ended at 104
# 2 started at 101 and ended at 102
# 3 started at 103 and ended at 105