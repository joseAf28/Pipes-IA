import os
import numpy as np
import time
import matplotlib.pyplot as plt

pathFolder = os.path.dirname(os.path.abspath(__file__))

modelV1 = os.path.join(pathFolder, "pipeSlidesV1.py")
modelV2 = os.path.join(pathFolder, "pipeSlidesV2.py")
modelV3 = os.path.join(pathFolder, "pipeSlidesV3.py")


testArr = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "15", "20", "25", "30", "35", "40", "45"]

# testArr = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "15"]


timeInit = time.time()

def testModel(model, idTest):
    testTxt = os.path.join(pathFolder, f"test/test-{testArr[idTest]}.txt")
    commandTest = f"python3.11 {model} < {testTxt}"
    os.system(commandTest)


sizeTest = 15

arrayTimeV2 = np.zeros((len(testArr), sizeTest), dtype=float)
arrayTimeV3 = np.zeros((len(testArr), sizeTest), dtype=float)


for i in range(len(testArr)):
    for j in range(sizeTest):
        # testModel(modelV1, i)
        timeInitV2 = time.time()
        testModel(modelV2, i)
        arrayTimeV2[i,j] += time.time() - timeInitV2
        
        timeInitV3 = time.time()
        testModel(modelV3, i)
        arrayTimeV3[i,j] += time.time() - timeInitV3
    
    print(f"Test {i+1}/{len(testArr)}", end="\r")


meeanTimeV2 = np.mean(arrayTimeV2, axis=1)
meeanTimeV3 = np.mean(arrayTimeV3, axis=1)

stdTimeV2 = np.std(arrayTimeV2, axis=1)
stdTimeV3 = np.std(arrayTimeV3, axis=1)



with open(os.path.join(pathFolder, "timesResults.txt"), "w") as file:
    for i in range(len(testArr)):
        file.write(f"{testArr[i]}   {meeanTimeV2[i]}    {stdTimeV2[i]}\n")
    file.write("\n")
    for i in range(len(testArr)):
        file.write(f"{testArr[i]}   {meeanTimeV2[i]}    {stdTimeV2[i]}\n")
# testModel(modelV2, 0)

plt.figure(1)
plt.plot(testArr, meeanTimeV2, 'o', label="model V2")
plt.plot(testArr, meeanTimeV3, 'o', label="model V3")
plt.xlabel("Test")
plt.ylabel("Time [sec]")
plt.legend()
plt.grid()
plt.savefig("PlotTimeResults.png")

