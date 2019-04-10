import sys
import pandas
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python processOutput.py <Log Output File> <File to store CSV>")
    exit(0)
f = open(sys.argv[1], "r")
lines = f.readlines()
count = 0
fout = open(sys.argv[2], "w")
fout.write("Episode,Train Rewards,Test Rewards\n")
for i in range(len(lines)):
    #print(i, lines[i])
    trainRewards = ""
    testRewards = ""
    if lines[i] != "------------------------------------------\n":
        continue
    i+=1
    if lines[i].find("Episode") == -1 or lines[i].find("epsilon") == -1:
        continue
    i+=1
    #print(i)
    while i < len(lines): 
        if lines[i].find("INFO") == -1 or lines[i].find("Train") == -1:
            i+=1
        else:
            break
    line = lines[i].split(":")
    #print(i)
    if line[2] == "Train rewards":
        trainRewards = line[3].strip()
    i+=1
    while i < len(lines): 
        if lines[i].find("INFO") == -1:
            i+=1
        else:
            break
    line = lines[i].split(":")
    #print(i)
    if line[2] == "Test rewards":
        testRewards = line[3].strip()
    #print(count, trainRewards, testRewards)
    outline = str(count) + "," + trainRewards + "," + testRewards + "\n"
    fout.write(outline)
    count+=1
    #print("++++++++++++++")

fout.close()
df = pandas.read_csv(sys.argv[2])
plt.plot("Episode", "Train Rewards", data = df, marker = 'x', color = 'red')
plt.plot("Episode", "Test Rewards", data = df, marker = 'o', color = 'blue')
plt.show()

