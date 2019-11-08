def GoogleHard(n):
# 8/22/2019

    
    """Given an array of strictly the characters 'R', 'G', and 'B',
        segregate the values of the array
        so that all the Rs come first, the Gs come second, and the Bs come last.
        You can only swap elements of the array.
        Do this in linear time and in-place.
        it should become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
        
        """
    # time Complexity O(n)

    Bstart = Rstart = 0
    Bend = Rend = len(n)-1

    while Bstart < Bend or Rend > Rstart:
        if n[Rend] == "R":
            n[Rend],n[Rstart] = n[Rstart],n[Rend]
            Rstart +=1
        else:
            Rend -=1

        if n[Bstart] == "B":
            n[Bend],n[Bstart] = n[Bstart],n[Bend]
            Bend -=1
        else:
            Bstart+=1

    return n

n = ["B","B","R","R","B","R","R"]
print(GoogleHard(n))
