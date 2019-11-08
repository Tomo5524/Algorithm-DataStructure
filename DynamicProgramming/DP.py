# DynamicProgamming

# Be careful when bottomup starts off with 1 like Fibionacci


def fib(n):
    # Brute Force
    # time complexity O(2^N)
    if n == 1 or n == 2:
        return 1

    return fib(n-1) + fib(n-2)

print(fib(10))

def memoization(n):

    def fib_memo(n):
        # don't forget base case
        if n == 1 or n == 2:
            return 1
        # if already current val is visited, just return it
        if n in memo:
            return memo[n]

        result = fib_memo(n-1) + fib_memo(n-2)
        memo[n] = result
        return memo[n]

    memo = {}
    fib_memo(n)
    #print(memo)
    return memo[n]

print(memoization(35))

def fib_bottom_Up(n):

    # O(1) space
    """
    algorithm
    1, initiate two 1s (they are the base case in recursion solution)
    2, one of them gets the sum and another one gets what previous sum
    3,
    """
    # initiate two 1s,
    val1 = 1 # denotes previous sum
    val2 = 1 # denotes current sum

    # this for loop goes until 4 that is why return val1 + val2
    for i in range(3,n):
        # populate current sum
        cur_sum = val1 + val2
        # gets previous sum
        val1 = val2
        # gets current sum
        val2 = cur_sum

    return val1+ val2

    # O(n) space
    # bottom_up = [0 for i in range(n+1)]
    # bottom_up[1] = 1
    # bottom_up[2] = 1
    # for i in range(3,n+1):
    #     bottom_up[i] = bottom_up[i-1] + bottom_up[i-2]
    #
    # return bottom_up[-1]

print(fib_bottom_Up(10))