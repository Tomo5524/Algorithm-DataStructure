# Leetcode

# key is to find out the first left and right bound. how to find them

# 410. Split Array Largest Sum


# 11/18/2019
# 1231. Divide Chocolate
def maximizeSweetness(sweetness,k):

    """
    key is to Find the maximum among the minimum subarray sums
    algorithm
    1,  the desired total sweetness must fall in the range from the minimum
        sweetness to the average sweetness of the (K + 1) pieces
        meaning we know l and r now
    2, if current chocolate bars with current target is less than k+1 (number of people including myself),
        some of us cannot have chocolate so we need to increase number of bars by increasing left bound getting mid
        otherwise, keep local solution, and to get more optimal solution, decrease right bound

    3, even current sweetness will be greater than target
        e,g, [1,2,3,4,5,6,7,8,9], I thought it will get 7 bars with target 6 but
        sweetness of each bar can be greater than target, so it will end up 5
        [1,2,3,4][5,6][7][8][9]
    """


    l,r = min(sweetness),sum(sweetness)//(k+1)
    while l < r:
        cur_bars = 0
        # without +1, in test case 1, it will get stuck when target is 6
        target = (l+r+1) // 2
        cur_sweetness = 0
        for val in sweetness:
            cur_sweetness += val
            # if the current piece has a total greater than or equal to the
            # given target, make a cut
            if cur_sweetness >= target:
                # reset cur_sweetness
                cur_sweetness = 0
                # take apart a piece of chocolate
                cur_bars +=1

        # we cannot give every one a piece of bar so decrease target by decreasing right bound
        if cur_bars < k+1:
            r = target - 1

        # otherwise, we try to increase target to get optimal solution (sneakily) as much as we can
        else:
            l = target

    return l

sweetness = [1,2,3,4,5,6,7,8,9]
K = 5
sweetness1 = [5,6,7,8,9,1,2,3,4]
K1 = 8
sweetness2 = [1,2,2,1,2,2,1,2,2]
K2 = 2
print(maximizeSweetness(sweetness,K))
print(maximizeSweetness(sweetness1,K1))
print(maximizeSweetness(sweetness2,K2))
print()


# 1011. Capacity To Ship Packages Within D Days

# return the least weight capacity of the ship that will result in all packages
# on the conveyor belt being shipped with in D days

def ship(weights,d): # time complexity O(NlogW) w is sum of list

    # brute force
    # from the problem statement, the minimum capacity must lie between biggest weight and sum of weight
    # given date is one day, just return the sum,
    # if given date is too big, return the biggest value
    # when to add current weight to total

    """
    algorithm,
    1, answer will lie between maximum value and sum of list because if given date is 100,
        you just need to ship one boat each day and smallest capacity is the largest value in list
        if given date is 1, you just need to ship all of boats in one day which is sum of list
    2, to find out current weight, you need to include a value that is not shipped with the same boat
        e,g, cur_capacity is 8, and [3,2,1,4,2,4], 3 + 2 + 1 + 4 is greater than cur_capacity
        but if we inclue 4, the ship will sink cuz cargo exceeds capacity.
    4, if days with current capacity(mid) is greater than given date(d), which means it is taking too long
        and does not satisfy requirement, so increase capacity by increasing left bound
    5, if days with current capacity(mid) is less, current capacity is valid and there is room for improvemnt
        dont disqualify current capacity and decrease capaciy by decreasing right bound

    ### to disqualify current capacity, increase left bound, if current answer (days) equals given days(d)
        it is disqualified
    """

    l,r = max(weights),sum(weights)
    while l < r:
        days = 0 ### days start off with 0 as no matter what, since left bound is biggest value
                        # and right bound is sum of list so there will be at least one day
        cargo  = 0
        mid = (l+r) // 2 # mid denotes current capacity
        for w in range(len(weights)):
            # to find out if cargo exceeds cur_capacity (mid)
            # you have to include the value that will not go into the same boat as previous cargo
            #e,g cur_capacity is 8, and [3,2,1,4,2,4], 3 + 2 + 1 + 4 is greater than cur_capacity
            # but if we inclue 4, the ship will sink cuz cargo exceeds capacity
            # so exclude 4 [1]
            cargo += weights[w]
            if cargo > mid:
                cargo = weights[w] # [1]
                days +=1

        # if days are grater than, its taking too slow, so we neet to increase capacity
        # by increasing left bound
        # should be equal cuz if days with current capacity is same as given days,
        # it is disqualifed.
        if days > d:
            l = mid +1

        # if given date is greater, current capacity(mid) is valid and to check if there is room for improvement
        # and answer will be the first smallest valid capacity
        else:
            r = mid
    #
    return l

meow = [3,2,1,4,2,4]
days = 3
meow1 = [1,2,3,4,5,6,7,8,9,10]
days1 = 5
meow2 = [1,2,3,1,1]
days2 = 50
# print(ship(meow,days))
print(ship(meow1,days1))
# print(ship(meow2,days2))
print()


# 875. Koko Eating Bananas

def minEatingSpeed(piles,H): # time complexity O(NlogW) w is maximum size of pile
    """
    algorihtm,
    1, if current amount of banana (eaten per hour) can finish all plates, within given H
        perform binary search on it
    2, if it cannot finish it, decrease amount of banana, meaning decrease right bound
    3, return the first (in binary search array) value of banana that can finish all plates
    """
    # how to deal with leftover
    # how and where to initiate low and high
    #if max(piles) // len(piles) > H: return max(piles)

    # max is the right bound as koko cant eat more than max amount
    l, r = 1, max(piles)
    # perform binary search
    while l < r:
        mid = (l+r) // 2 # mid denotes amount of banana eaten per hour
        hours = 0 # hours denotes total hours to finish all plates
        # check each plate and if mid can finish each plate at a time
        for i in range(len(piles)):
            # j denotes hours it takes to finish each plate with mid(cur amount koko can eat at a time)
            x = (piles[i] + mid -1)
            j = x // mid
            # j = (piles[i] + mid -1) // mid

            hours += j

        # if it takes longer to finish all plates than given hour,
        # meaning it cannot finish all plates with given hour,
        # so increase amount of banana by increasing left bound
        # all the left bounds are the values that cannot finish all the plates
        if hours > H:
            l = mid + 1

        # if it takes faster to finish all plates than given hour,
        # meaning cur amount of bananas can finish all plates withing given hour
        # eating too fast and to help koko slow down, decrease amout of banana  by decreasing right bound
        # answer is the smallest mid, so keep this amount (mid) qualified
        else:
            r = mid
            #l = mid+1

    return l

piles = [3,6,7,11]
H = 8
piles1 = [30,11,23,4,20]
H1 = 5
piles2 = [30,11,23,4,20]
H2 = 6
# print(minEatingSpeed(piles,H))
print(minEatingSpeed(piles1,H1))
# print(minEatingSpeed(piles2,H2))
print()

# 33. Search in Rotated Sorted Array
def BS_roatate(nums,target):
    """
    algorithm,
    ### if left to middle is sorted, perform standard binary search on sorted left sub array
    ### if middle to right is sorted, perform standard binary search on sorted right sub array
    1, Initiate left to be equal to 0, and riht to be equal to n - 1.
    2, perform standard binary search
    3, check if left element is less than middle element,
        left to middle is non rotated(sorted) and
        if it is and target is between left and middle, go to left
        otherwise go to right
    4 if left to middle is rotated and
        target is between middle to right, go to right
    5, otherwise, target is in roated array

    """
    if not nums: return -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid

        # this one determines if left to middle is sorted or not
        # if non rotated is in left to right, perform
        if nums[l] <= nums[mid]:
            # from mid to l is sorted
            # e,g [3,1] t = 1
            # should be eqaul not juut mid > l
            # when mid is 0 and l is 0
            # is its not equal, causes error

            # num[l] to nums[mid] is sorted and if target is in between, go to left
            # target cannot be middle itself as it should have been caught in ln 68
            # so no need equal for it
            if nums[l] <= target < nums[mid]:  # taget is in between l and mid
                # if target is in sorted sub array (left to right), go to left
                r = mid - 1

            # if target is greater than middle element, go to right
            else:  # e,g, 1,3,5, t = 5 is already sorted so you wanna go to the right
                # in right side
                l = mid + 1

        # pivot is somewhere between middle to right
        else:  # from mid to r is sorted
                # so you can simply do binary search from middle to right
            # e,g [5,1,3], t = 3
            # if nums[mid] > target >= nums[r], it will move left and return false
            # target cannot be middle itself as it should have been caught in ln 68
            # so no need equal for it
            if nums[mid] < target <= nums[r]:  # target is in between mid and r
                l = mid + 1
            else:  # in left sub array
                r = mid - 1

    return -1

# test cases
meow = [5, 6, 7, 8, 9, 10,11,23,45,1, 2, 3]
t = 23
meow1= [1,3,5]
t1 = 5
meow2 = [5,1,3]
t2 = 3
meow3 = [1,3]
t3 = 3
meow4 = [3,1]
t4 = 1
meow5 = [4,5,6,7,0,1,2]
t5 = 4
meow6= [9,0,1,2,3,4,5,6,7]
t6 = 9
print(BS_roatate(meow,t)) # 8
print(BS_roatate(meow1,t1)) # 2
print(BS_roatate(meow2,t2)) # 2
print(BS_roatate(meow3,t3)) # 1
print(BS_roatate(meow4,t4)) # 1
print(BS_roatate(meow5,t5)) # 0
print(BS_roatate(meow6,t6)) # 0
print()

# 35. Search Insert Position
def searchInsert(nums,target):
    """
    algorihtm
    1, assuming there is no duplicates, I can just do old fashion binary search
    2, if target does not exist, binary search always terminates right before the place
        where target is supposed to be.
        e,g, [1,3,5,6,8], t = 4
        first iteration, middle is 2, second time middle is 0, third time is 1, forth time, l = 2 and r = 1
        so just return l
    3, so just simply return left if target doesnt exist
    """
    def bs(nums,t,l,r):

        while l <= r:
            mid = (l+r) // 2
            if nums[mid] == t:
                return mid

            if t < nums[mid]:
                r = mid -1

            else:
                l = mid + 1

        return l

    if not nums: return 0
    # create helper function
    return bs(nums,target,0,len(nums)-1)

meow = [1,3,5]
t = 5
meow4 = [1,3]
t4 = 3
meow6 = [1,3,5,6,8]
t6 = 4
meow7 = [1,3,5,6]
t7 = 0
meow8 = [1,3,5,6]
t8 = 7

# print(searchInsert(meow,t))
# print(searchInsert(meow4,t4))
# print(searchInsert(meow6,t6))
# print(searchInsert(meow7,t7))
# print(searchInsert(meow8,t8))
# print()

# 153. Find Minimum in Rotated Sorted Array
def findmin(nums):
    """
    algorithm
    ### smallest value is alaways the value right after pivot
    1, the main idea for our checks is to place the left and right bounds on the start of the pivot,
        and never disqualify the index for a possible minimum value
    2, if middle element is greater than right, pivot must be somewhere between middle and right
        as it is not sorted, move to right. e,g [3,4,5,6,7,8,9,1,2]
    3,  if right element is greater than middle element, pivot must be somewhere between left and middle
        as left to middle is not sorted, move to left [8,9,1,2,3,4,5,6,7]
        ### to never disqualify current middle, right gets middle so middle element is still a potential answer
    4, after left bound is equal or greater than right bound, left is at smallest value so just return it
    """

    l, r = 0, len(nums)-1
    # DO NOT use left <= right because that would loop forever
    while l < r:
        # find the middle value between the left and right bounds (their average)
        mid = (l+r) // 2

        if nums[mid] > nums[r]:
            # we KNOW the pivot must be to the right of the middle:
            # we KNOW that the pivot/minimum value must have occurred somewhere
            # between middle and right which is why the values wrapped around and became smaller.

            # example:  [3,4,5,6,7,8,9,1,2]
            # in the first iteration, when we start with mid index = 4, right index = 9.
            # if nums[mid] > nums[right], we know that at some point to the right of mid,
            # the pivot must have occurred, which is why the values wrapped around
            # so that nums[right] is less then nums[mid]

            # we know that the number at mid is greater than at least
            # one number to the right, so we can use mid + 1 and
            # never consider mid again; we know there is at least
            # one value smaller than it on the right

            # pivot must have happened between middle to right so simply move towards right
            l = mid + 1

        else:
            # here, nums[mid] <= nums[right]:
            # we KNOW the pivot must be the middle or to the left of the middle:
            # if nums[mid] <= nums[right], we KNOW that the pivot was not encountered
            # to the right of middle, because that means the values would wrap around
            # and become smaller (which is caught in the above if statement).
            # this leaves the possible pivot point to be at index <= mid.

            # example: [8,9,1,2,3,4,5,6,7]
            # in the first iteration, when we start with mid index = 4, right index = 9.
            # if nums[mid] <= nums[right], we know the numbers continued increasing to
            # the right of mid, so they never reached the pivot and wrapped around.
            # therefore, we know the pivot must at index <= mid.

            # we know that nums[mid] <= nums[right].
            # therefore, we know it is possible for the mid index to store a smaller
            # value than at least one other index in the list (at right), so we do
            # not discard it by doing right = mid - 1. it still might have the minimum value.

            # mid index is potentially the target so dont disqualify mid
            # by r = mid not r = mid-1
            r = mid

    # at this point, left and right converge to a single index (for minimum value)
    # our if/else block forces the bounds of left/right to shrink each iteration:

    # left is located at smallest value at this point
    # so simply return l
    # edge case, if there is only one value, since l = 0, this will cover it
    return l

print()
print('findmin')
meow1 = [3,4,5,1,2] #1
meow2 = [4,5,6,7,0,1,2] #0
meow3 = [3,4,5,6,7,8,9,1,2] #
meow4 = [8,9,1,2,3,4,5,6,7] #
meow5 = [4,5,1,2,3]
print(findmin(meow1))
print(findmin(meow2))
print(findmin(meow3))
print(findmin(meow4))
print(findmin(meow5))