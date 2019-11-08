# Binary Search
# only applies to sorted list
# Time Complexity lon(On)

def BSsearchIterative(arr,t,l,r):
    """
    algorithm
    1, find middle element, if target lies in left subarray, move on to left
        if target lies in right, move to right
    2, do this process until middle element is target
    """

    # make sure when if operation is just greater or greater and equal
    # understand which applies to what
    # when target is last value or first value
    # meaning, want to check every element in array
    # probably need to use equal? figure it out
    # check search algorithm and compare them to Binary search
    # to figure out application of while <= , or while < /.
    #
    # iterative
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == t:
            return mid

        if arr[mid] > t:
            r = mid - 1  # move to left

        else:
            l = mid + 1  # move to right

    return None

meow = [1, 2, 3, 5, 6, 7, 8, 9, 10,11,23,45]
print(BSsearchIterative(meow,10,0,len(meow)-1))


def BSsearchRecursive(arr,t,l,r):

    if l <= r:

        mid = (l+r) // 2
        if arr[mid] == t:
            return mid

        if arr[mid] > t:
            return BSsearchRecursive(arr,t,l,mid-1)

        else:
            return BSsearchRecursive(arr,t,mid+1,r)

    return None

meow = [1, 2, 3, 5, 6, 7, 8, 9, 10,11,23,45]
print(BSsearchRecursive(meow,45,0,len(meow)-1))