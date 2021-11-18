from os import listdir
from os.path import isfile, join

mypath = "assets/NOAH - Voice for TBKK\TBKK - Thai 35 Users"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)