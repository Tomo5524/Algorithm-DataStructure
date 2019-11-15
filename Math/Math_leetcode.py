

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

