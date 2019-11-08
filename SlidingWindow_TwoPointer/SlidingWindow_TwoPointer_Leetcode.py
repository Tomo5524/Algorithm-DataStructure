# Leet Code
"""
Edge Case
    1, Empty sequence
    2, Sequence with 1 or 2 elements
    3, Sequence with repeated elements
"""

# 209. Minimum Size Subarray Sum
# Minimum Size Subarray Sum
"""Given an array of n positive integers and a positive integer s,
    find the minimal length of a contiguous subarray of which the sum ≥ s.
    If there isn't one, return 0 instead.

"""
def minSubArrayLen(arr,s):

    """
    algorithm
    1, use sliding window, initialize two pointers (start and i), start is 0 i is for loop
    2, initialize minimum as len(nums)+1 cuz if subarray is whole array, minimum length is whole array
        but if whole array is less than target, it is 0, but since mini is initialized as len(arr), it will return arr
        to avoid this mistake, set it as len(arr)+1
    3, if current window's sum is greater than target, subtract arr[start_point]
    4, keep doing process 3, until current window is less than target
    """

    # # optimal solution
    # time complexity O(n)
    start_p,window = 0,0
    # + 1 cuz for edge case, 1,2,3,4,5, t = 15 answer is 5 and and 6 is not equal 5 so return 5
    # whereas, 1,1, t = 4, smallest is 0 but initiated as len(arr) + 1 so smallest is still len(arr) + 1 thus answer is 0
    min_length = len(arr) + 1
    for i in range(len(arr)):
        window += arr[i]
        while window >= s:
            # +1 to match the gap between length and index
            # e,g, [1,2,3,4,5] j - i is 4 but answer is 5 so +1 needed
            min_length = min(min_length, i - start_p +1)
            window -= arr[start_p]
            # shrink left
            start_p += 1

    return 0 if min_length == len(arr) +1 else min_length

    # brute force
    # + 1 cuz for edge case, 1,2,3,4,5, t = 15 answer is 5 and and 6 is not equal 5 so return 5
    # whereas, 1,1, t = 4, smallest is 0 but initiated as len(arr) + 1 so smallest is still len(arr) + 1 thus answer is 0
    # smallest = len(arr) + 1
    # for i in range(len(arr)):
    #     total = 0
    #     for j in range(i,len(arr)):
    #         total += arr[j]
    #         if total >= s: # should be equal cuz if cur_val alone is same as s, that is the answer
    #             smallest = min(smallest,j - i+1) # +1 to match the gap between length and index
    #                                                 # e,g, [1,2,3,4,5] j - i is 4 but answer is 5 so +1 needed
    #             break
    #
    # return 0 if smallest == len(arr)+1 else smallest

#15
meow = [1,2,3,4,5] # 5
#4
meow1 = [1,4,4] # 1
#11
meow2 = [1,2,3,4,5] # 3
#7
meow3 = [2,3,1,2,4,3]# 2
#100
meow4 = [] # 0
#213
meow5 = [12,28,83,4,25,26,25,2,25,25,25,12] # 8
#6
meow6 = [10,2,3] # 1
#3
meow7 = [1,1] # 0

print(minSubArrayLen(meow,15))
print(minSubArrayLen(meow1,4))
print(minSubArrayLen(meow2,11))
print(minSubArrayLen(meow3,7))
print(minSubArrayLen(meow4,100))
print(minSubArrayLen(meow5,213))
print(minSubArrayLen(meow6,6))
print(minSubArrayLen(meow7,3))


# 11. Container With Most Water

def maxArea(height):

    # Optimal solution
    """
    algorithm
    ### key is to find out cur_max water by finding smaller value out of left and right
        and multiply it by right - left
    1, create 2 pointers, wanna get current max for each value
    2, to get cur max water, find out which value(left and right) is smaller
        and multiply it by the difference between right and left
    3, if left pointer is smaller, move it to right by incrementing it
        if right pointer is smaller, move it left by decrementing it

    ## if there is way greater value, let's say 1000
        e,g, [1,8,6,2,1000,4,8,3,7], 1000 does not matter  as it cannot contain anything
        key is to find out most water not the biggest value

    """
    max_water = 0
    l,r = 0,len(height)-1
    while l < r:
        # be careful with parenthesis!!
        # if there is no parenthesis, multiplication will be operated first
        max_water = max(max_water, (r-l) * min(height[l],height[r]))
        if height[l] > height[r]:
            r -= 1
        else:
            l += 1

    return max_water

    # brute force # time complexity O(n2)
    """
    algorithm
    1, loop over each value and cur_max is right point(j) - left point(i) * smaller element (min(height[i],height[j]))
    2, and keep track of max value
    """
    # max_water = 0
    # for i in range(len(height)):
    #     for j in range(i,len(height)):
    #         cur_max = (j - i) * min(height[i],height[j])
    #         max_water = max(max_water,cur_max)
    #
    # return max_water

h =  [1,8,6,2,5,4,8,3,7] # 49
h1 = [3,1,2,4,5]
h2 = [1,2]
h3 = [1,8,6,2,1000,4,8,3,7] # does 9 matter?

print ()
print(maxArea(h))
print(maxArea(h1))
print(maxArea(h2))
print(maxArea(h3))

# 42. Trapping Rain Water

def TrappingRainWater(heights):
    """
    ## key is find left max and right max of current bar, get smaller one out of left max and right max
    and subtract current value with whichever is smaller
    """

    # optimized solution
    # great explanation
    # https://leetcode.com/problems/trapping-rain-water/discuss/17391/Share-my-short-solution.
    """
    algorithm
    1, initiate two pointer, left and right
    2, if left bar is smaller, that means there is a greater or equal bar
        on the right side, thus we can fill water left max to current left bar
    3, if right bar is smaller than left, do the same
    3, move smaller value. e,g, if left is greater, move right to left (r -=1) along with filling water
        if right is greater, move left to right along with filling water
    4, it doesnt matter what is in the middle, as all that matters is that
        when dealing with left bar, if there is a greater bar on the right side
        so doesnt matter how greater the greater bar on the right side is
        e,g [2,0,1,0,3,1,0,2,2,0,5] left is 2 right is 5.
        when current left bar is 0 and left max is 2 and right max is 5
        so left bar can be filled with 2 water and
        left bar is being compared to 3 not 5. so that is why it dosent matter what is in the middle

    """
    l,r = 0,len(heights)-1
    max_water = 0
    left_max = right_max = 0
    while l < r:
        # if left bar is smaller, we know we can fill water from left max to current bar
        # as it is guaranteed that there is a bigger or equal bar on the right
        if heights[l] <= heights[r]:
            left_max = max(left_max,heights[l])
            max_water += left_max - heights[l]
            l +=1

        else:
            right_max = max(right_max, heights[r])
            max_water += right_max - heights[r]
            r -= 1

    return max_water

    # dynamic programming
    # get left max
    # ans = 0
    # left_max = [0 for _ in range(len(heights))]
    # right_max = [0 for _ in range(len(heights))]
    #
    # # get left max for each value
    # left_max[0] = heights[0]
    # for i in range(1,len(heights)):
    #     left_max[i] = max(heights[i],left_max[i-1])
    #
    # # get right max for each value
    # right_max[-1] = heights[-1]
    # for i in range(len(heights)-2,-1,-1):
    #     right_max[i] = max(heights[i],right_max[i+1])
    #
    # for i in range(len(heights)):
    #     ans += min(left_max[i],right_max[i]) - heights[i]
    #
    # return ans

    # brute force
    # time complexity O(n2)
    """
    algorithm
    1, think of this question like contain water.
        so we need to find how much water each bar can contain.
    2, initialize max_left and max_right and get minimum value out of left and right
        subtract current value with minimum value out of left and right
    3,  and max_water gets the value calculated in process 2
    """

    # max_water = 0
    # for i in range(len(heights)):
    #     # get left max
    #     left_max = 0
    #     for l in range(i,-1,-1):
    #         left_max = max(left_max,heights[l])
    #
    #     # get right max
    #     right_max = 0
    #     for r in range(i,len(heights)):
    #         right_max = max(right_max,heights[r])
    #
    #     cur_max = min(left_max,right_max) - heights[i]
    #     # make sure cur_max is not negative
    #     if cur_max > 0:
    #         max_water += cur_max
    #
    # return max_water

meow = [0,1,0,2,1,0,1,3,2,1,2,1]
meow1 = [2,0,2]
meow2 = [4,2,3]
meow3 = [2,0,1,0,3,1,0,2,2,0,5]

print()
print(TrappingRainWater(meow))
print(TrappingRainWater(meow1))
print(TrappingRainWater(meow2))
print(TrappingRainWater(meow3))

# 3. Longest Substring Without Repeating Characters
"""
Given a string, find the length of the longest substring without repeating characters.

"""
from collections import Counter
def lengthOfLongestSubstring(s):

    # optimal solution
    # we only iterate through the string once so
    # Time complexity is O(N)
    # key is when finding duplicates, where to start off next
    """
    algorithm
    1, keep counting non unique characters and if we run into duplicates,
        update start pointer.
    2, update current longest substring and dictionary (current value and index) each iteration
    3,
    """
    # initialize dictionary
    dic = {}
    start,longest = 0,0
    # i is right pointer and start is left pointer
    # use enumerate of readability
    for i,char in enumerate(s):
        # if char in dic, update start point
        if char in dic:
            # update start of string index
            # in "dvdf case, when encountering d for the second time,
            # v should be next start , so pick dic[val] +1
            # "abba" this case
            # if slow = dic[char]+1,
            # last 'a' will make slow index 1 as at that point, a:0 and slow will get 1
            # and longest will get i - slow, which is 3
            # and len(s) - slow will be 3 which is false
            start = max(start, dic[char]+1)

        dic[char] = i
        longest = max(longest,i - start + 1)

    return longest

    # brute force
    """
    algorithm
    1, initiate hash set and iterate each letter
    2, put to hash set the words that are not in hashset and increment counter
    3, if current letter is in hashset, reset hashset and reset counter
    4, do 2 and 3 untile for loop is done
    """

    # seen = set()
    # cnt = 0
    # longest = 0
    # for i in range(len(s)):
    #     for j in range(i,len(s)):
    #         if s[j] in seen:
    #             # edge case for " ", answer should be 1 but longest is 0
    #             # if longest is here,
    #             #longest = max(longest,cnt)
    #             seen = set()
    #             cnt = 0
    #             break
    #
    #         # current letter is not in seen
    #         seen.add(s[j])
    #         cnt+=1
    #         longest = max(longest,cnt)
    #
    # return longest
    #return cnt if cnt > longest else longest

meow = "abcabcbb" # 3
meow1 = " " # 1
meow2 = "pwwkew" # 3
meow3 = "au" # 2
meow4 = "dvdf" # 3
meow5 = "abba" # 2
meow6 = "abbbbbba" # 2
meow7 = ""
meow8 = "aab"

print()
print(lengthOfLongestSubstring(meow))
print(lengthOfLongestSubstring(meow1))
print(lengthOfLongestSubstring(meow2))
print(lengthOfLongestSubstring(meow3))
print(lengthOfLongestSubstring(meow4))
print(lengthOfLongestSubstring(meow5))
print(lengthOfLongestSubstring(meow6))
print(lengthOfLongestSubstring(meow7))
print(lengthOfLongestSubstring(meow8))

# 76. Minimum Window Substring

def minWindow(s,t):

    """
    algorithm
    Key is how to check all letters in t are found, how to check all 0s in hashmap
    and how to update counter and how to shrink window
    and how to keep head of window at the beginning of window
    1, using two pointers, keep going until all letters are found,
        then shrink window
    2, if all letter are found, shrink window as long as window has all letters
    3, every time we encounter a letter in t, update hashmap
        and decrease counter if hashmap[t] is greater than 0
        other wise, just update dic and dont do anything with counter
    4, when shrinking window, if we run into a letter in t,
        increase counter and update dic
    5, keep track of head of window so to make it easy to return window
    6, from the beginning of window to beginning of window plus minimum lenght is the window
    """
    # Time complexity, O(∣S∣+∣T∣) where |S| and |T| represent the lengths of strings S and T.
    dic = {}
    for char in t:
        if char not in dic:
            dic[char] = 1
        else:
            dic[char] +=1

    start, end = 0,0
    # initialize counter as length of target, when it becomes 0, meaning all letters are found
    missing = len(t)
    min_length = len(s) + 1
    while end < len(s):

        # if current char in dic, update dic, # cur_char = s[end]
        if s[end] in dic:
            # if tartget is not found yet, decrement counter
            # e,g, 'B' at index 9, this will get 'B': -1 so it is already in window
            # don't decrement counter
            if dic[s[end]] > 0:
                missing -= 1

            # otherwise, just update dictionary
            dic[s[end]] -= 1

        end +=1 # placing end here, we dont need to worry about difference between index and length
        # all letter are found now, so shrink window and get minimum length
        while missing == 0:
            # get minimum length
            # min_length = min(min_length, end - start) # dont need +1 as end increments at the very end so no gap between index and length
            # this dosent work for edge case,
            # S5 = "cabwefgewcwaefgcf",T5 = "cae"
            # as start keeps updated
            if min_length > end - start:
                min_length = end - start
                # head only gets updated when new smaller window is found
                # stays at the beginning of window
                head = start

            # if current letter is target, update dic and counter
            if s[start] in dic:
                # when letter's value is 0, that means there is only one letter of the letter in window so update counter
                dic[s[start]] += 1
                # if value is greater than 0, we are missing that letter s[start]
                if dic[s[start]] > 0:
                    missing += 1
                    # if current letter's value is less  or than 0, meaning there are duplicates of that letter
                    # so no need to increment counter

            start+=1

    # head to minimum length is the window so yeah
    return "" if min_length == len(s)+1 else s[head:head+min_length]


    # brute force
    # if len(t) > len(s): return ""
    # mini = len(s)+1
    # ans = ""
    # for i in range(len(s)):
    #     match = ""
    #     cnt = 0
    #     seen = set()
    #     for j in range(i, len(s)):
    #         match += s[j]
    #         if s[j] not in seen and s[j] in t:
    #             cnt += 1
    #             if cnt == len(t):
    #                 if len(match) < mini:
    #                     ans = match
    #
    #                 mini = min(mini, len(match))
    #         seen.add(s[j])
    #
    # return ans

S = "ADOBECODEBANC"
T = "ABC"
S1 = "a"
T1 = "aa" # ''
S2 = "aa"
T2 = "aa" # 'aa'
S3 = 'azjskfzts'
T3 = 'sz'
S4 = 'ab'
T4 = 'a'
S5 = "cabwefgewcwaefgcf" # "cwae"
T5 = "cae"
print()
# print(minWindow(S,T))
# print(minWindow(S1,T1))
# print(minWindow(S2,T2))
# print(minWindow(S3,T3))
# print(minWindow(S4,T4))
print(minWindow(S5,T5))

