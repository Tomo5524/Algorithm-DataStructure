# cookpad
def main(s, n):
    if 0 < n <= 4:
        return helper_(s, n)

    elif 4 < n <= 7:
        return helper(s,n)

    #elif 7 < n <= 11

def helper_(s, n):

    ans = ""
    for i in range(n):
        ans += s[i]
    return ans

def helper(s,n):
    cnt = 0
    ans = ""
    for i in range(1,len(s)):
        ans += s[i]
        cnt +=1

    j = 0
    while n > cnt:
        ans += s[0] + s[j]
        j += 1
        cnt +=1

    return ans


test = 'ABCD'
n = 3
test1 = 'ABCD'
n1 = 6
print(main(test,n))
print(main(test1,n1))

# 11/14/2019
def titleToNumber(s):

    ans = 0
    for ch in s:
        ans = ans * 26 + (ord(ch) - 64)

    return ans

test  ="A" # -> 1
test1 = "B" # -> 2
test2 = "C" # -> 3
test3 = "Z" # -> 26
test4 = "AA" # -> 27
test5 = "AB" # -> 28
test6 = "AAA" # -> 703
test7 = "ZY" # 701
print(titleToNumber(test))
print(titleToNumber(test1))
print(titleToNumber(test2))
print(titleToNumber(test3))
print(titleToNumber(test4))
print(titleToNumber(test5))
print(titleToNumber(test6))
print(titleToNumber(test7))
print()


def fizzBuzz(n):
    res = []
    for i in range(1, (n) + 1):

        # when number is divisible by 3 and 5
        if i % 3 == 0 and i % 5 == 0:
            res.append("FizzBuzz")

        # when number is divisible by 3
        elif i % 3 == 0:
            res.append("Fizz")

        # when number is divisible by 5
        elif i % 5 == 0:
            res.append("Buzz")

        else:
            res.append(str(i))

    return res

print(fizzBuzz(1))
print(fizzBuzz(15))

