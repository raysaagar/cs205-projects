a = range(0,5,2)     # [0,2,4], range(beg,end,step), step is int
a.append(5)          # [0,2,4,5]
b = [2*i for i in a] # [0,4,8,10]
c = a + b            # [0,2,4,5,0,4,8,10], Concat, NOT addition
d = a * 2            # [0,2,4,5,0,2,4,5], Concat, NOT mult
e = a[1:3]           # [2,4], Excludes last index
g = a[2:]            # [4,5], Index 2 through last
h = a[-1]            # [5], Index backwards from end
i = a[:-2]           # [0,2], All but the last two elements
