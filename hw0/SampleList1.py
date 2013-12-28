mylist = [1, 2.5, "Hello", [3, "World"]]
print(mylist[0])     # Prints 1
mylist[3][1]= ("World!!!").upper()
print(mylist[3])     # Prints [3, 'WORLD!!!']
mylist.reverse()
print(mylist)        # Prints [[3, 'WORLD!!!'], 'Hello', 2.5, 1]
mylist[0].append(7)
print(mylist[0])     # Prints [3, 'WORLD!!!', 7]
