def readTraincsv(file):
    # skip first line
    file.readline()
    # relevant indexes 
    # Pclass (2), Sex (5), Age (6), SibSp(7), Parch(8), Fare (10)
    # Name takes up two spaces since they have a comma
    # Survived (1)
    features = []
    survived = []

    for line in file:
        [_, survival, pClass, _, _, sex, age, sibSp, parch, _, fare, _, _] = line.split(",")
        feature = [pClass, sex, age, sibSp, parch, fare]
        if feature[1] == 'female':
            feature[1] = 0
        else:
            feature[1] = 1

        if feature[2] == '':
            feature[2] = 30 # average age

        feature = [float(f) for f in feature]
        features.append(feature)
        survival = [float(f) for f in survival]
        survived.append(survival)

    return [features, survived]

def readTestcsv(file):
    file.readline()
    features = []
    pIDs = []

    for line in file:
        [pID, pClass, _, _, sex, age, sibSp, parch, _, fare, _, _] = line.split(",")
        feature = [pClass, sex, age, sibSp, parch, fare]
        if feature[1] == 'female':
            feature[1] = 0
        else:
            feature[1] = 1

        if feature[2] == '':
            feature[2] = 30 # average age

        if feature[-1] == '':
            feature[-1] = 14.5 # median fare

        feature = [float(f) for f in feature]
        features.append(feature)
        pIDs.append(pID)

    return [features, pIDs]

import torch
import torch.nn
import numpy as np

train = open('titanic/train.csv', 'r')
[trainFeatures, trainSurvived] = readTraincsv(train)
test = open('titanic/test.csv', 'r')
[testFeatures, pIDs] = readTestcsv(test)

dtype = torch.float
device = torch.device("cpu")

trainTensor = torch.Tensor(trainFeatures, device=device)
testTensor = torch.Tensor(testFeatures, device=device)
survivedTensor = torch.Tensor(trainSurvived, device=device)

# N is number of people
# D_in number of features (6 features)
N, D_in, H, D_out = np.shape(trainFeatures)[0], np.shape(trainFeatures)[1], 100, 1

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, H, device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-5
for t in range(5000):
    # 304722.0
    # Forward pass
    y_pred = trainTensor.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)

    # print(y_pred)
    # print(survivedTensor)
    loss = (y_pred - survivedTensor).abs().sum()
    # criterion = torch.nn.CrossEntropyLoss()
    # loss = criterion(y_pred, survivedTensor)

    if t % 100 == 99:
        print(t, loss.item())

    # Backward pass
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w3 -= learning_rate * w3.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()

# Apply the prediction to test case
test_pred = testTensor.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)

# Round each value in test_pred to 1 or 0
for i in range(0, len(test_pred)):
    if(test_pred[i] < 0.5):
        test_pred[i] = 0
    else:
        test_pred[i] = 1

outputFile = open('titanic/gender_submission.csv', 'w')
outputFile.write('PassengerId,Survived\n')
for i in range(0, len(test_pred)):
    outputFile.write("%s,%d\n" %(pIDs[i], test_pred[i]))

outputFile.close()