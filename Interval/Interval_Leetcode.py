"""
    ### Intervals ###

    input is usually nested and index 0 is always start and end is index1 [1,3] s = 1, e = 3
    so all we have to do is check previous last val to current val
    if meetings[i-1][1] > meetings[i][0] = overlappping
    if previous last val is larger, it is overlapping
    An example of an interval array: [[1, 2], [4, 7]] = Non=overlapping
    [[1, 5], [4, 7]], overlapping.

"""


def is_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def merge_overlapping_intervals(a, b):
    return [min(a[0], b[0]), max(a[1], b[1])]

a = [1, 2]
b = [4, 7]
c = [1, 5]
d = [4, 7]
print(is_overlap(a,b))
print(merge_overlapping_intervals(a,b))
print(is_overlap(c,d))
print(merge_overlapping_intervals(c,d))

"""
    Corner cases
    1, Single interval
    2, Non-overlapping intervals
    3, An interval totally consumed within another interval
    4, Duplicate intervals
    
"""
import heapq
# 252. Meeting Rooms
def canAttendMeetings(intervals):
    ## Given an array of meeting time intervals consisting of start and end times
    ## [[s1,e1],[s2,e2],...] (si < ei),
    ## determine if a person could attend all meetings.



    # find if there is any overlapping
    intervals.sort()
    # edge case, single interval could pass test as for starts off with 1
    # and single interval will be ignored and return true
    for i in range(1, len(intervals)):

        # ,[2,4]],[[7,10] # no overlapping as current list of first value (start time)
        # is bigger than the previous list of last value (end time)
        # meaning the previous meeting is alredy done so can attend next meeting
        if intervals[i - 1][1] > intervals[i][0]:
            return False

    return True

meow = [[0,30],[5,10],[15,20]] # false
meow1 = [[7,10],[2,4]] # true

print(canAttendMeetings(meow))
print(canAttendMeetings(meow1))

# 253. Meeting Rooms II

def meetingRooms(intervals):
    # Time Complexity (O(nlogn))

    """
    Algorithm

    1, MOST IMPORTANT -> Separate out the start times and the end times in their separate arrays.
    2. Sort the start times and the end times separately.
        Note that this will mess up the original correspondence of start times and end times. They will be treated individually now.
    3, We consider two pointers: s_ptr and e_ptr which refer to start pointer and end pointer.
        The start pointer simply iterates over all the meetings and the end pointer helps us track if a meeting has ended
         and if we can reuse a room.
    4, When considering a specific meeting pointed to by s_ptr,
        we check if this start timing is greater than the meeting pointed to by e_ptr.
        If this is the case then that would mean some meeting has ended by the time the meeting at s_ptr had to start.
        So we can reuse one of the rooms. Otherwise, we have to allocate a new room.
        If a meeting has indeed ended i.e. if start[s_ptr] >= end[e_ptr], then we increment e_ptr.
        Repeat this process until s_ptr processes all of the meetings.
    """

    #  # Separate out the start and the end timings and sort them individually.
    # this sorting takes O(NlogN)
    start_times = sorted([i[0] for i in intervals])
    end_times = sorted([i[1] for i in intervals])
    rooms = 0
    start_p, end_p = 0, 0
    # check all the meeting by checking all the start times
    while start_p < len(intervals):
        # if there is a meeting that has ended by the time at start_p starts
        # comparing 15 to 10, when 15 starts, meeting that ended at 10,
        # so free up one room by subtracting
        # if start and end are the same, we will free up one room, as the previous meeting has ended
        if start_times[start_p] >= end_times[end_p]:
            # Free up a room and increment the end_pointer.
            # one of meetings has just ended and now is avilable
            rooms -= 1
            end_p += 1

        # If a room got free, then this used_rooms += 1 wouldn't have any effect. used_rooms would
        # remain the same in that case. If no room was free, then this would increase used_rooms
        # no matter what, someone will use a room
        # if one of rooms has just become available, it now just become ocupied
        rooms += 1
        # process one meeting
        start_p += 1

    return rooms

print()
meow = [[0, 30],[5, 10],[15, 20]] # 2
meow1 = [[7,10],[2,4]] # 1
meow2 = [[1,5],[8,9],[8,9]] # 2
meow3 = [[13,15],[1,13]]# 1
meow4  =[[9,10],[4,9],[4,17]] # 2
print(meetingRooms(meow))
print(meetingRooms(meow1))
print(meetingRooms(meow2))
print(meetingRooms(meow3))
print(meetingRooms(meow4))

# 435. Non-overlapping Intervals

def eraseOverlapIntervals(intervals):
    # time complexity O(nlog(n))

    """algorithm,
        1, if we encounter, overlapping, update current interval end val, interval[1] and increment cnt
        takeaway is that, we dont actually delete anything
        but we delete it by updating end val
        so overlapping that has larger end will be deleted

    """
    if not intervals: return 0
    intervals.sort()  # O(nlog(n)). Sorting takes O(nlog(n)) time.
    cnt = 0
    for i in range(1, len(intervals)):
        # if intervals[i-1][1] > intervals[i][0]: # # cur_end is not updated so 100 will be compaired
        # check if it is overlapping,
        # we can do intervals[i][0] < cur_end
        # cuz if overlapping,  i[0] should be somewhere
        # between interval[i-1][0] and cur_end
        # e,g, 1,11 and 2,12 are overlapping as 2 is somewhere between
        # previous[0] and cur_end as it is sorted so i[0] is greater thatn previous[0] fosho
        # it would naturally fall under overlapping area
        if intervals[i - 1][1] > intervals[i][0]:
            cnt += 1
            # edge case
            # [[1,100],[11,22],[1,11],[2,12]]
            # cur_end needs to update so it is not 100 but 11
            intervals[i][1] = min(intervals[i - 1][1], intervals[i][1])  # update cur_end, get the smaller end
            # so we can accurately find out all overlapping

    return cnt
meow = [[1,100],[11,22],[1,11],[2,12]] # 2
meow1 = [[1,2],[2,3]] # 0
meow2 = [[1,2],[2,3],[3,4],[1,3]] # 1
meow3 =  [[1,2],[1,2],[1,2]] # 2
print()
print(eraseOverlapIntervals(meow))
print(eraseOverlapIntervals(meow1))
print(eraseOverlapIntervals(meow2))
print(eraseOverlapIntervals(meow3))

# 56. Merge Intervals

""" Given a collection of intervals, merge all overlapping intervals. """

# Other than the sort invocation, we do a simple linear scan of the list,
# so the runtime is dominated by the O(nlgn) complexity of sorting.

def MergeIntervals(intervals):
    """
    algorithm
    1, sort the intervals
    2, comparison over previous interval and current interval is based off last value in result
        as last interval in result is the current longest merger
        so just update current merge by updating last value of result[1]
    3,

    """
    # edge case
    if not intervals: return None
    intervals.sort() # O(NlogN)
    res = []
    idx = 0
    # check all intervals
    for interval in intervals:
        # edge case [[1,4],[2,3]]
        # keep greater end val
        # if res[-1][1] is greater than current interval end value, keep res[-1][1]
        # same values are considered overlapping
        if res and res[-1][-1] >= interval[0]:
            res[-1][-1] = max(res[-1][-1], interval[1])

        else:
            # get first value no matter what for edge case when there is only one interval
            # if not overlapping, append current interval to list
            res.append(interval)

    return res

meow = [[1,4],[0,4]] # not sorted
meow1 = [[1,4],[5,6]] # loop started from 1
meow2 = [[1,3]]
meow3 = [[1,4],[4,5]] # duplicats considered as over lapping, also started from 1
meow4 = [[1,4],[2,3]] # keep greater end val
meow5 = [[1,3],[2,6],[8,10],[15,18]]
print()
print(MergeIntervals(meow))
print(MergeIntervals(meow1))
print(MergeIntervals(meow2))
print(MergeIntervals(meow3))
print(MergeIntervals(meow4))
print(MergeIntervals(meow5))

meow6 = [[2,3],[4,5],[6,7],[8,9],[1,10]]
print(MergeIntervals(meow6))


# 57. Insert Interval
def InsertInterval(intervals,newInterval):
    """
    algorithm
    1, Add to the output all the intervals starting before newInterval based on start val
    2, Add newInterval to the output. Merge it with the last added interval if newInterval starts before the last added interval.
    3, Add the next intervals one by one. Merge with the last added interval if the current interval starts before the last added interval.
    """
    new_start, new_end = newInterval
    idx, n = 0, len(intervals)
    res = []

    # add all intervals starting before newInterval
    while idx < n and new_start > intervals[idx][0]:
        # no overlapping
        res.append(intervals[idx])
        idx += 1

    # add newInterval as new_start is somewhere between
    # if there is no overlap, just add the interval
    # what if not output or output[-1][1] < new_start:
    # edge case, meow2 = [[1,5],[6,8]], newInterval2 = [0,9]
    # edge [[1,5]], t = [6,8], just "if not res", will just return 1,8
    # new_start 6 is greater than cur_end 5, it is overlapping thus append it
    # if not res or ns > intervals[idx][0], in [[1,5]] , t = [2,3] case,  it will be out ouf range
    # cuz idx is 1 and length is just 0
    ### if newstart is greater than cur end in list, it is not overlapping
    if not res or new_start > res[-1][-1]:  # #meow1 = [[1,2],[5,6],[6,7],[8,10],[12,16]], t= 4,8
        # # 4(ns) will be greater than 2 so it will get 4 in above case
        res.append(newInterval)  # we found a new merging point in the case above, will get 4,8 instead of 5,6
    # if there is an overlap, merge with the last interval
    else:
        # edge case
        # [[1,5]], [2,3]
        #  # if max(ne,intervals[idx][1]) [[1,3],[6,9]], t = [2,5] it will be 1,9
        res[-1][1] = max(res[-1][1], new_end)

    # add next intervals, merge them if needed
    while idx < n:
        cur_start, cur_end = intervals[idx]
        # if there is no overlap, just add an interval
        # not eqaul cuz 8 and 8 are overlapping
        if res[-1][1] < cur_start:
            # append non overlapping, 12,16
            res.append(intervals[idx])
        # if there is an overlap, merge with the last interval
        else:
            res[-1][1] = max(res[-1][1], cur_end)

        idx += 1

    return res

meow = [[1,3],[6,9]]
newInterval = [2,5]

meow1 = [[1,2],[3,5],[6,7],[8,10],[12,16]]
newInterval1 = [4,8]

meow2 = [[1,5],[6,8]] # edge case, new_start is less
newInterval2 = [0,9]

meow3 = [[1,5]]
newInterval3 = [1,7]

meow46 = []
newInterval4 = [5,7]

meow5 = [[1,5]]
newInterval5 = [2,3]

meow6 = [[1,2],[3,5],[4,17],[6,7],[8,10],[12,16]]
newInterval6 = [4,8]

meow7 = [[1,2],[5,6],[6,7],[8,10],[12,16]]
newInterval7 = [4,8]

meow8 = [[1,5]]
newInterval8 = [6,8]

print()
print(InsertInterval(meow,newInterval)) # [[1, 5], [6, 9]]
print(InsertInterval(meow1,newInterval1)) # [[1, 2], [3, 10], [12, 16]]
print(InsertInterval(meow2,newInterval2)) # [[0, 9]]
print(InsertInterval(meow3,newInterval3)) # [[1, 7]]
print(InsertInterval(meow46,newInterval4))# [[5, 7]]
print(InsertInterval(meow5,newInterval5)) # [[1, 5]]
print(InsertInterval(meow6,newInterval6)) # [[1, 2], [3, 17]]
print(InsertInterval(meow7,newInterval7)) # [[1, 2], [4, 10], [12, 16]]
print(InsertInterval(meow8,newInterval8)) # [[1, 5], [6, 8]]

