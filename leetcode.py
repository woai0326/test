x = 120
string = str(x)
if x<0:
    new = string[:0:-1]
    new = string[0]+new
else:
    new = string[::-1]

print(int(new))