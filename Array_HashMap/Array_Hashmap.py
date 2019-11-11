
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