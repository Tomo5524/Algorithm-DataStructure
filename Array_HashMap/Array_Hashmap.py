
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

    dic = {}
    res = []
    for num in nums:
        if num not in dic:
            dic[num] = 1

    for i in range(1, len(nums) + 1):
        if i not in dic:
            res.append(i)

    return res

[4,3,2,7,8,2,3,1]
[1,1]