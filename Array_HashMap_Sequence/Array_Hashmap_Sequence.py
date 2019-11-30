
# 11/30/2019
# 1. Two Sum


# 459. Repeated Substring Pattern
def repeatedSubstringPattern(s):

    # optimal solution
    """
    algorithm
    1, it is basically to find repeated substring
    2, impossible to have a substring >(len(s)/2), that can be repeated
        so concatenate original input
    3, and get rid of first and last character
    So, when ss = s + s , we will have at least 4 parts of "repeated substring" in ss.
        e,g, 'aba' -> ab,a,aba,ba,a
     4, we are removing 1st char and last char => Out of 4 parts of repeated substring,
        2 part will be gone (they will no longer have the same substring).
        e,g,'aba' -> b,ba,ab,a
    """

    # aba will be true if I do not remove 1 and -1
    ss = s+s[1:-1]
    return s in ss

    # brute force
    # time complexity: O(n*m)
    cur = ''
    for ch in s:
        cur += ch
        temp = cur
        while len(temp) <= len(s):
            temp += cur
            if temp == s:
                return True

    return False

test = "abab"
test1 = "aba"
test2 = "abcabcabcabc"
test3 = "helloworld"
print(repeatedSubstringPattern(test))
print(repeatedSubstringPattern(test1))
print(repeatedSubstringPattern(test2))
print(repeatedSubstringPattern(test3))
print()

# 11/17/2019
# 394. Decode String
def decodeString(s):

    """
    algorithm
    1, use stack, and take care of inner most bracket first
    2, and expand to outer bracket
    3,
    """
    # time complexity O(n)
    stack = []
    for ch in s:
        if ch == "]":
            # get character in innermost bracket
            cur = ''
            while stack:
                val = stack.pop()
                # keep popping until we see open bracket
                if val == "[":
                    break
                # new val should come in the front
                cur = val + cur

            # get number that belongs to the bracket we are on now
            num = ''
            # we don't want to pop top value in stack yet,
            # as we can only pop it when we know last value in stack is digit
            while stack and stack[-1].isdigit():
                # popped value should come first for edge case '123[ab]]
                num = stack.pop() + num

            stack.append(int(num)*cur)

        else:
            stack.append(ch)

    return ''.join(stack)


test = "3[a]2[bc]" # return "aaabcbc".
test1 = "3[a2[c]]" # return "accaccacc".
test2 = "2[abc]3[cd]ef" # return "abcabccdcdcdef".
test3 = '12[leetcode]'
print(decodeString(test))
print(decodeString(test1))
print(decodeString(test2))
print(decodeString(test3))
print()

# 11/16/2019
# 238. Product of Array Except Self
def product(arr):
    """
    algorithm
    Instead of dividing the product of all the numbers in the array by the number at a given index to get the corresponding product,
    we can make use of the product of all the numbers to the left and all the numbers to the right of the index.
    Multiplying these two individual products would give us the desired result as well.
    """

    res = []
    l,r,ans = [1]*len(arr),[1]*len(arr),[1]*len(arr)
    for i in range(1,len(l)):
        l[i] = l[i-1] * arr[i-1]

    for j in range(len(arr)-2,-1,-1):
        r[j] = r[j+1]*arr[j+1]

    for i in range(len(ans)):
        ans[i] = l[i] *r[i]

    return ans

arr = [1, 2, 3, 4]
# [1,1,2,6]
print(product(arr))
print()

from collections import deque
class NestedIterator:
    def __init__(self, nestedList):
        self.stack = deque([])
        self.stack.append(nestedList)
        self.value = None

    def next(self):
        temp = self.value
        self.value = None
        return temp

    def hasNext(self):
        stack = self.stack
        while stack:
            # get top of stack
            # be careful with index
            top = stack.popleft()
            if len(top) == 1:
                # # check if value on top is integer r not
                if type(top[0]) == int:
                    self.value = top[0]
                    return True

                # if current list is nested, flatten it
                # be carful with how to flatten it
                else:
                    stack.appendleft(top[0])

            # if top is greater than 1, create nested list for each list
            # should reverse it, edge case for [1,[4,[6]]]
            else:
                for li in reversed(top):
                    stack.appendleft([li])

        return False


test = [[1,1],2,[1,1]]
test1 = [1,[4,[6]]]
i, v = NestedIterator(test), []
res_ = []
res_1 = []
while i.hasNext(): res_.append(i.next())
print(res_)

i, v = NestedIterator(test1), []
while i.hasNext(): res_1.append(i.next())
print(res_1)
print()

# 11/15/2019

# 647. Palindromic Substrings
def countSubstrings(s):

    #optimal solution
    """
    algorithm
    1, If looking down to the assignment of left and right, for example a string of length 4, it will be:
        left 0, right 0 (right = left + 0)
        left 0, right 1 (right = left + 1)
        left 1, right 1 (right = left + 0)
        left 1, right 2 (right = left + 1)
        left 2, right 2 (right = left + 0)
        left 2, right 3 (right = left + 1)
        left 3, right 3 (right = left + 0) # your code stops here
        left 3, right 4 (right = left + 1) # this is additional if written in my way showed below, but it is still safe
    2, if there is a match we expand search by decrementing left and incrementing right
    3, basically left is iterated in range(0, len(s)), and right is either left + 0 or left + 1
    """

    res = 0
    for i in range(len(s)):
        for j in range(2):
            l = i
            r = i + j
            while l >= 0 and r < len(s) and s[l] == s[r]:
                res +=1
                l -= 1
                r += 1

    return res

    # dp
    # time complexity O(n2)
    n = len(s)
    dp = [[False for i in range(n)] for _ in range(n)]
    cnt = 0
    # single letter is palindrome
    for i in range(n):
        dp[i][i] = True
        cnt +=1

    for i in range(n-1):
        if s[i] == s[i+1]:
            dp[i][i+1] = True
            cnt+=1

    for k in range(3,n+1):
        search_till = n - k +1
        for i in range(search_till):
            left = i
            right = i + k -1
            if dp[left+1][right-1] and s[left] == s[right]:
                dp[left][right] = True
                cnt+=1

    return cnt

test = "abc"
test1 = "aaa"
test2 = 'abcaaeddde'
test3 = "abcaa"

print(countSubstrings(test))
print(countSubstrings(test1))
print(countSubstrings(test2))
print(countSubstrings(test3))
print()

# 18. 4Sum
def fourSum(nums,target):

    # optimal solution
    """
    algorithm
    1, get every two subset,
    2, if length of subset is 2, we can just do 2 sum
    3,
    """
    # time complexity O(n3)

    def getFourSum(nums,t,n,sub,res):

        if n == 2:
            # get 2 sum that equal to cur_total
            l, r = 0, len(nums) - 1
            while l < r:
                cur_total = nums[l] + nums[r]
                if cur_total == t:
                    res.append(sub+[nums[l],nums[r]])
                    # when there is a match, move left pointer
                    l += 1
                    while l < r and nums[l-1] == nums[l]:
                        l +=1

                elif cur_total < t:
                    l += 1

                else:
                    r -= 1

        else:
            for i in range(len(nums)):
                if i == 0 or (i > 0 and nums[i] != nums[i-1]):
                    getFourSum(nums[i + 1:], t-nums[i], n - 1, sub + [nums[i]], res)

        return

    res = []
    sub = []
    nums.sort()
    getFourSum(nums,target,4,sub,res)
    return res

    # brute force
    # res = []
    # nums.sort()
    # for i in range(len(nums)):
    #     for j in range(i+1,len(nums)):
    #         for k in range(j + 1, len(nums)):
    #             for o in range(k + 1, len(nums)):
    #                 if nums[i] + nums[j] + nums[k] + nums[o] == target:
    #                     res.append([nums[i],nums[j],nums[k],nums[o]])
    #
    # final_res = []
    # for li in res:
    #     if li not in final_res:
    #         final_res.append(li)
    #
    # return final_res

# print(fourSum([-3,-2,-1,0,0,1,2,3],0)) # [[-3,-2,2,3],[-3,-1,1,3],[-3,0,0,3],[-3,0,1,2],[-2,-1,0,3],[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
print(fourSum([1,0,-1,0,-2,2],0)) # [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
print(fourSum([-1,0,-5,-2,-2,-4,0,1,-2],-9)) # [[-5,-4,-1,1],[-5,-4,0,0],[-5,-2,-2,0],[-4,-2,-2,-1]]
print()

# 11/14/2019
# 15. 3Sum
def threeSum(nums):

    # optimal solution
    """
    key is how to skip duplicates
    algorithm
    1, sort the array first
    2, create left and right pointer
    3, add up current, left and right values and if target is 0, append 3 values
    """
    nums.sort()
    res = []
    for i in range(len(nums)):
        # if current value is greater than 0, it will not be 0
        if nums[i] > 0: break

        # i > 0 for [0,0,0]
        if i > 0 and nums[i-1] == nums[i]:
            continue

        # We always start the left pointer from i+1 because the combination of 0~i has already been tried. [2]
        l = i+1
        r = len(nums)-1
        while l < r:
            cur_total = nums[i] + nums[l] + nums[r]

            # #[1,-1,-1,0]
            #If the total is less than zero, we need it to be larger, so we move the left pointer.
            if cur_total < 0:
                l +=1
            # If the total is greater than zero, we need it to be smaller, so we move the right pointer. [4]
            elif cur_total > 0:
                r -= 1

            else:
                res.append([nums[i],nums[l],nums[r]])
                # skip duplicates
                while l < r and nums[l] == nums[l+1]:
                    l +=1

                while r > 0 and nums[r] == nums[r-1]:
                    r -=1
                # these while loops stop at the end of duplicates

                # update left and right when target is 0
                l +=1
                r-= 1

    return res

    # time complexity O(n3) power
    # brute force
    res = []
    seen = set()
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            for k in range(j+1,len(nums)):
                if (nums[i] + nums[j] + nums[k]) == 0:
                    if tuple(sorted([nums[i], nums[j], nums[k]])) not in seen:
                        res.append([nums[i],nums[j],nums[k]])
                        seen.add(tuple(sorted([nums[i], nums[j], nums[k]])))

    return res


# print(threeSum([0,0,0,0]))
# print(threeSum([0,0,0]))
print(threeSum([-1, 0, 1, 2, -1, -4]))
print(threeSum([-2,0,1,1,2]))
print(threeSum([1,-1,-1,0]))
print(threeSum([-2,0,0,2,2]))
print()
[[0, 0, 0]]
[[0, 0, 0]]
[[-1, 0, 1], [-1, 2, -1]]
[[-2, 0, 2], [-2, 1, 1]]
[[1, -1, 0]]
[[-2, 0, 2]]

def generate(numRows):
    if numRows == 0:
        return None

    # res = [1] # [1, [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    res = [[1]]
    for i in range(2, numRows + 1):
        cur_level = [1] * i
        for j in range(1, len(cur_level) - 1):
            cur_level[j] = res[-1][j] + res[-1][j - 1]

        res.append(cur_level)

    return res

print(generate(5))

# 11/13/2019
# 581. Shortest Unsorted Continuous Subarray
def findUnsortedSubarray(nums):
    """
    algorithm
    1, we have two pointer that finds first non-increasing, l, and first non-descending, r in array
    2, but any numbers outside could be violating and they should be included to the answer
    3,
    """

    l,r = 0,len(nums)-1
    # get first non-increasing and non-decreasing
    while l < len(nums)-1 and nums[l] <= nums[l+1]:
        l += 1

    while r > 0 and nums[r-1] <= nums[r]:
        r -= 1

    # if array is already sorted, return 0
    #  # [1,2,3,4]
    # l will be bigger
    if l > r:
        return 0

    temp = nums[l:r+1]
    smallest = min(temp)
    biggest = max(temp)

    # check any numbers outside temp
    # [1, 3, 7, 2, 5, 4, 6, 10]
    #  decrement l value until we find something that is lower than minimum.
    while l > 0 and nums[l-1] > smallest:
        l -= 1

    # len(nums) -1 for [1,2]
    # increment r value until we find something that is greater than maximum.
    while r < len(nums) -1 and nums[r+1] < biggest:
        r += 1

    return (r - l) +1

test = [1, 3, 7, 2, 5, 4, 6, 10]
test1 = [1,2,3,4] # 0
test2 = [2,1] # 2
test3 = [2,6,4,8,10,15] # 2
test4 = [1,3,2,2,2] # 2
test5 = [1,3,2,3,3] # 4
test6 = [1,2,3,3,3] # 0
test7 = [2, 6, 4, 8, 10, 9, 15] # 5
print(findUnsortedSubarray(test))
print(findUnsortedSubarray(test1))
print(findUnsortedSubarray(test2))
print(findUnsortedSubarray(test3))
print(findUnsortedSubarray(test4))
print(findUnsortedSubarray(test5))
print(findUnsortedSubarray(test6))
print(findUnsortedSubarray(test7))
print()


# 11/11/2019

def minMoves(nums):

    # optimal solution
    return sum(nums) - len(nums)*min(nums)

    # brute force
    # time complexity O(nlogn)
    """
    1, Visualize the nums array as a bar graph where the value at each index is a bar of height nums[i].
        Sort the array such that the bar at index 0 is minimum height and the bar at index N-1 is highest.
    2, Now in the first iteration, make a sequence of moves such that the height at index 0 is equal to height at index N-1.
        Clearly this takes nums[N-1]-nums[0] moves. After these moves, index N-2 will be the highest and index 0 will still be the minimum and nums[0] will be same as nums[N-1].
    3, In the next iteration, lets do nums[N-2]-nums[0] moves.
        After this iteration, nums[0], nums[N-2], and nums[N-1] will be the same.

    it is basically dp, each iteration, we will solve sub problem,
    e,g, [1,2] we just need 1 move to make 1 and 2 same
         [1,3] we just need 2 moves to make 1 and 3 same
         [1,4] we just need 3 moves to make 1 and 4 same
         and so on...
    """
    nums.sort()
    cnt = 0
    for i in range(len(nums)):
        cnt += nums[i] - nums[0]

    return cnt

    # brute force
    # time complexity O(n2)
    # each iteration, find minimum value and max value,
    # increment all the values other than max_value by 1
    # and when mini value becomes same as max_value, return counter
    # be careful with duplicates
    cnt = 0
    while True:
        min_val,max_val = min(nums),max(nums)
        if max_val == min_val:
            break

        # for duplicates
        loc = nums.index(max_val)
        for i in range(len(nums)):
            if i != loc:
                nums[i] +=1

        cnt+=1

    return cnt

test = [1,2,3]
test1 = [4,6,2,7,8,5,3,1]
print(minMoves(test))
print(minMoves(test1))
print()

from collections import defaultdict
def subarraySum(nums,k):

    """
    key is to store accmulative sum and if accumulative - k exists in dictionary
    increment counter. and how to store values in key
    algorithm
    1, basically this is 2 sum question, if sum - target exists in dictionary, get the value
    """

    dic = {}
    # # sum of zero defaults to 1 count. It's necessary.
    # # nums[1,1] ,k =2  , then tsum=2
    # # Thus count = hmap[tsum-k]=hmap[0]=1
    # # Example:- Consider array [1,2,3] and target = 6. So when you are in last iteration. Your value of running sum will be 6.
    # # So you want to add +1 to the answer. So when you calculate running sum-target = 0.
    dic[0] = 1
    Sum = 0
    res = 0
    for num in nums:

        Sum += num

        # increment count only if there is valid key in hmap
        # valueAtK = Sum-k ; only if valueAtK exists as key in hmap
        if Sum - k in dic:
            res += dic[Sum-k]
            #dic[Sum-k] +=1

        # if key exists increment value by 1 at key, else set it to 1
        # The only reason why we have this condition is
        # to support 0 and negative values in nums that reduce tsum
        dic[Sum] = dic[Sum] + 1 if Sum in dic else 1

    return res

    # brute force
    # res = 0
    # for i in range(len(nums)):
    #     Sum = 0
    #     for j in range(i, len(nums)):
    #         Sum += nums[j]
    #         if Sum == k:
    #             res += 1
    #
    # return res

nums = [1,1,1] # 2
k = 2
num1 = [1] # 1
k1 = 1
num2 = [1,2,1,2,1] # 4
k2 = 3
num3 = [1,2,3] # 2
k3 = 3
num4 = [0,0,0,0,0,0,0,0,0,0] # 55
k4 = 0
num5 = [3,4,7,2,-3,1,4,2] # 4
k5 = 7
print(subarraySum(nums,k))
print(subarraySum(num1,k1))
print(subarraySum(num2,k2))
print(subarraySum(num3,k3))
print(subarraySum(num4,k4))
print(subarraySum(num5,k5))


def missingNumber(nums):
    # nums.sort()
    # nums_set = set(nums)
    # for i in range(len(nums)+1):
    #     if i not in nums_set:
    #         return i
    n = len(nums)

    # get the sum which is suppoused to be the sum
    expected_sum = n * (n + 1) // 2
    # given sum
    actual_sum = sum(nums)
    return expected_sum - actual_sum


# 448. Find All Numbers Disappeared in an Array
def findDisappearedNumbers(nums):

    seen = set()
    res = []
    for num in nums:
        if num not in seen:
            seen.add(num)

    for i in range(1, len(nums) + 1):
        if i not in seen:
            res.append(i)

    return res

[4,3,2,7,8,2,3,1]
[1,1]