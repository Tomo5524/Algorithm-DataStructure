# Window Sliding Technique


"""
1, We compute the sum of first k elements out of n terms
using a linear loop and store the sum in variable window_sum.

2, Then we will graze linearly over the array till it reaches the end
and simultaneously keep track of maximum sum.

3, To get the current sum of block of k elements
just subtract the first element from the previous block and
add the last element of the current block .

"""

# return maximum sum of a contiguous sub array of size k
def maxSumSizek(arr,k):

    # optimized solution
    max_sum = 0
    cur_sum = 0
    for i in range(len(arr)):
        cur_sum += arr[i]
        if (i - k) >= 0:
            cur_sum -= arr[i-k]

        max_sum = max(max_sum,cur_sum)

    return max_sum

    # # brute force
    # max_sum = 0
    # for i in range(len(arr)):
    #     cur_sum = 0
    #     for j in range(i,k+i):
    #         cur_sum += arr[j]
    #
    #     max_sum = max(max_sum,cur_sum)
    #
    # return max_sum

arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
# print(maxSumSizek(arr,4))


# Sliding Window

# Trapping Rain Water


def TrappingRainWater(heights):
    """ algorithm
        1, traverse every element and find tallest on left (0,i) and right (i,len(arr)-1)
        2, it is guranteed that if leftmax is smaller than right max, you can store leftmax - cur_element
            because it does not matter the difference between leftmax and rightmax
        3, find out smller ele out of leftmax and right max and subtract if it is greater than 

"""
    # optimized solution
    l,r = 0,len(heights)-1
    total = 0
    leftmax,rightmax = 0,0
    while l < r:
        if heights[l] < heights[r]:
            if leftmax < heights[l]:
                leftmax = heights[l]
            else:
                total += leftmax - heights[l]
            l +=1
        else:
            if rightmax < heights[r]:
                rightmax = heights[r]
            else:
                total += rightmax - heights[r]

            r -=1
            
    return total
                
    # brute force
    # time complexity O(n2)
##    total = 0
##    for i in range(len(heights)):
##        cur = heights[i]
##        left_max = 0
##        right_max = 0
##        for left in range(i,-1,-1):
##            left_max = max(left_max,heights[left])
##        for right in range(i,len(heights)):
##            right_max = max(right_max,heights[right])
##        
##        rain = min(left_max, right_max) - cur
##        if rain > 0:
##            total += rain
##
##    return total

meow = [0,1,0,2,1,0,1,3,2,1,2,1]
meow1 = [2,0,2]

print(TrappingRainWater(meow))
print(TrappingRainWater(meow1))

# Minimum Size Subarray Sum
"""Given an array of n positive integers and a positive integer s,
    find the minimal length of a contiguous subarray of which the sum ≥ s.
    If there isn't one, return 0 instead.

"""

def minSubArrayLen(nums,s):

    # fast / slow
    
    windowSum, windowStart = 0,0
    # this len(nums)+1 does not matter
    # this is merely the closest value of possible biggest length
    minLength = len(nums)+1
    for windowEnd in range(len(nums)):
        windowSum+=nums[windowEnd]
        while windowSum >= s:
            minLength = min(minLength, windowEnd - windowStart + 1)
            # plus one cuz finding length which starts off with 1
            # shrink left 
            windowSum -= nums[windowStart]
            windowStart+=1

    # this is faster than the second one as 0 is placed in the begging
    # if all vals combined are not greater or equal to t, return 0
    return 0 if minLength == len(nums)+1 else minLength
    #return minLength if minLength != len(nums)+1 else 0

##    # bruteforce
##    min_len = len(arr)+1
##    if not arr: return 0
##    for start in range(len(arr)):
##        cur_max = arr[start]
##        # edge case when start value is greater or equal
##        if cur_max >= s:
##            return 1
##
##        # avoid iterating same value
##        for end in range(start+1,len(arr)):
##            cur_max += arr[end]
##            if cur_max >= s:
##                min_len = min(min_len, (end-start)+1)
##                break
##    return 0 if min_len == len(arr)+1 else min_len
            


##15
meow = [1,2,3,4,5] # 5

#4
meow1 = [1,4,4] # 1

#11
meow2 = [1,2,3,4,5] # 3

#7
meow3 = [2,3,1,2,4,3]# 2f

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


def KadaneAlgorithm(arr):

    # brueforce
##    maxsum = 0
##    for i in range(len(arr)):
##
##        for j in range(i+1,len(arr)):
##            cur_max = arr[i]+arr[j]
##            print(cur_max)
##            maxsum = max(maxsum,cur_max)
##
##    return maxsum

    max_cur = max_sum = arr[0]
    for i in range(1,len(arr)):
        max_cur+= arr[i]
        if arr[i] > max_cur:
            max_cur = arr[i]

        max_sum = max(max_sum,max_cur)

##        max_cur = max(arr[i],max_cur+arr[i])
##        max_sum = max(max_sum,max_cur)

    return max_sum


#meow = [-2,3,2,-1]
meow = [-2,1,-3,4,-1,2,1,-5,4]
print('KadaneAlgorithm')
print(KadaneAlgorithm(meow))

# Maximize number of 0s by flipping a subarray

# this is BS dont even bother
##def flip(arr):
##    max_zero = 0
##    ogzero = 0
##    
##
##    for i in range(len(arr)):
##        
##        if arr[i] == 0:
##            ogzero +=1
##
##        cnt0,cnt1 = 0,0
##        
##        for j in range(len(arr)):
##        #for j in range(i,len(arr)):
##            if arr[j] == 1:
##                cnt1 +=1
##            else:
##                cnt0+=1
##                
##            max_zero = max(max_zero,cnt1-cnt0)
##
##
##    return max_zero + ogzero
##            
##
##meow = [0,1,0,0,1,1,0]
##print(flip(meow))
##        

# 3. Longest Substring Without Repeating Characters

"""
Given a string, find the length of the longest substring without repeating characters.

"""

def lengthOfLongestSubstring(s):

    # fast and slow
    dic = {}
    slow,max_len = 0,0
    for fast,val in enumerate(s):
        # when val already in dictionary
        if val in dic:
            # get the current length from start val to current val
            cur_m = fast - slow
            # store the max length
            max_len = max(max_len,cur_m)            
            # update start of string index to the next index
            # slow = max(fast,dic[val]+1)
            # in "dvdf case, when encountering d for the second time,
            # v should be next start , so pick dic[val] +1
            # "abba" this case
            # if slow = dic[val]+1, slow will be 1
            # and len(s) - slow will be 3 which is false
            slow = max(slow,dic[val]+1)
        # add/update char to/of dictionary 
        dic[val] = fast
    # answer is either in the begining/middle OR some mid to the end of string
    # edge case for when there is no duplicate values, "au"
    # start always keeps track of middle val
    return max(max_len, len(s)-slow)


    
    # brute force 
##    n = len(s)
##    ans = 0
##    for i in range(n):
##        for j in range(n):
##            ans = max(ans,allUnique(s,i,j))
##
##    return ans
##                
##
##def allUnique(s,front,end):
##    seen = set()
##    for i in range(front,end+1):
##        if s[i] in seen:
##            return 0
##
##        seen.add(s[i])
##
##    return len(seen)

meow = "abcabcbb" # 3
meow1 = " " # 1
meow2 = "pwwkew" # 3
meow3 = "au" # 2
meow4 = "dvdf" # 3
meow5 = "abba" # 2
meow6 = "abbbbbba" # 2


print('meow')
print(lengthOfLongestSubstring(meow))
print(lengthOfLongestSubstring(meow1))
print(lengthOfLongestSubstring(meow2))
print(lengthOfLongestSubstring(meow3))
print(lengthOfLongestSubstring(meow4))
print(lengthOfLongestSubstring(meow5))
print(lengthOfLongestSubstring(meow6))


# Window Sliding Technique

"""
1, We compute the sum of first k elements out of n terms
using a linear loop and store the sum in variable window_sum.

2, Then we will graze linearly over the array till it reaches the end
and simultaneously keep track of maximum sum.

3, To get the current sum of block of k elements
just subtract the first element from the previous block and
add the last element of the current block .

"""

def max_sum(arr,n,k):
    # return the biggest sum of subarray of k lengths

    # get first window
    window_sum = sum([arr[i] for i in range(k)])
    max_sum = 0
    # goes until last window
    for i in range(n-k):
        # 
        window_sum = (window_sum - arr[i]) + arr[i+k]

        max_sum = max(window_sum,max_sum)

    return max_sum

# brute force

##def maxSum(arr,n,k):
##
##    max_sum = 0
##
##    for i in range(n-k+1):
##        cur_m = 0
##        for j in range(k):
##            cur_m = cur_m + arr[i+j]
##
##        max_sum = max(cur_m,max_sum)
##            
##            
##    return max_sum
    
arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
#print(maxSum(arr,len(arr),4))

    
meow = [1, 4, 2, 10, 2, 3, 1, 0, 20]
#meow = [1, 2, 3, -7, 7, 2, -12, 6]
#print(max_sum(meow,len(meow),4))


def profit(prices):

    cur_min = float('inf')
    max_p = 0
    for price in prices:
        cur_min = min(cur_min,price)
        max_p = max(max_p,price-cur_min)

    return max_p

    
    
prices = [9,11,8,5,7,10]
#prices = [7,6,4,3,1]
print(profit(prices))

def maxWater(height):

    l,r = 0,len(height)-1
    max_p = 0
    while l < r:
        max_p = max(max_p, min(height[l],height[r]) * (r-l))
        
        if height[l] < height[r]:
            l +=1
        else:
            r -=1
            
    return max_p

    # my work

##    start,end = 0,len(height)-1
##    cm = 0
##    max_sum = 0
##    while start < end:
##        cm = min(height[start],height[end]) * (end - start)
##        max_sum = max(max_sum,cm)
##        if height[start] < height[end]:
##            start+=1
##        else:
##            end-=1
##
##    return max_sum

                      
h =  [1,8,6,2,5,4,8,3,7] # 49
h1 = [3,1,2,4,5]
h2 = [1,2]

print(maxWater(h))
print(maxWater(h1))
print(maxWater(h2))

    
