local criterion = require("./criterion")
local prep = require("./prep")
local build = require("./build")
local test = require("./test")
local train = require("./train")
local mnist = require("./mnist")
local batch = require("./minibatch")
local data = require("./dataset")

require("os")
math.randomseed(os.time())

printedData = 0
dataSize = 4294967296
useGPU = false
epochs = 25
transforms = 0
minibatch = -1
for i=1, table.getn(arg) do
    if arg[i] == "--make_images" then
        if i < table.getn(arg) then
            printedData = tonumber(arg[i+1])
        end

        printedData = printedData or 100

        printedData = math.floor(printedData)
        assert(dataSize >= 0, "Must have a image print size")
    end

    if arg[i] == "--data_size" and i < table.getn(arg) then
        dataSize = math.floor(tonumber(arg[i+1]))
        assert(dataSize >= 0, "Must have a positive data set size")
    end

    if arg[i] == "--epochs" and i < table.getn(arg) then
        epochs = math.floor(tonumber(arg[i+1]))
        assert(epochs >= 0, "Must have a positive number of epochs")
    end

    if arg[i] == "--transforms" and i < table.getn(arg) then
        transforms = math.floor(tonumber(arg[i+1]))
        assert(transforms >= 0, "Must have a positive number of transforms")
    end

    if arg[i] == "--cuda" then
        useGPU = true
    end

    if arg[i] == "--minibatch" then
        minibatch = 10
        if i < table.getn(arg) then
            minibatch = math.floor(tonumber(arg[i+1]))
            assert(epochs >= 0, "Must have a positive size minibatch")
        end
    end
end

print("Loading training data...")
trainData = mnist.read_data("./data/train", dataSize)

--Regardless of how much training data is used
--read the whole test set; training is the long part
print("\nLoading testing data...")
testData = mnist.read_data("./data/t10k", 4294967296)

if printedData > 0 then
    local images = require("./images")
    require "lfs"

    lfs.mkdir("./images")
    lfs.mkdir("./images/testingImages")
    lfs.mkdir("./images/trainingImages")

    print("\nSaving up to "..printedData.." images from each set")
    for i=1, printedData do
        if testData:size() >= i then
            images.save_tensor("./images/testData/"..i..".bmp", testData[i][1])
        end
        if trainData:size() >= i then
            images.save_tensor("./images/trainData/"..i..".bmp", trainData[i][1])
        end
    end
end

print("\nCreating neural net...")
net = build.build_mnist_net()
crit = criterion.build_mnist_criterion()

print("\nCreating "..transforms.." transformed copies of each image...")
transformed = prep.transformSet(trainData, transforms)

trainData.data = torch.cat(trainData.data, transformed.data, 1)
trainData.labels = torch.cat(trainData.labels, transformed.labels, 1)

if minibatch < 0 then minibatch = trainData:size() end
miniData = torch.Tensor(minibatch, 1, trainData[1][1]:size(2), trainData[1][1]:size(3))
miniLabels = torch.Tensor(minibatch)
minibatchData = data.make_dataset(miniData, miniLabels)

if useGPU then
    print("\nConverting to CUDA types...")
    local data = require("./dataset")
    require("cunn")

    minibatchData.data = minibatchData.data:cuda()
    minibatchData.labels = minibatchData.labels:cuda()
    net = net:cuda()
    crit = crit:cuda()
end

print("\nNormalizing training data...")
mean, stdev = prep.normalize(trainData.data)

print("\nTraining...")
train.train_minibatch_nn(net, crit, trainData, minibatchData, epochs)

print("\nNormalizing testing data...")
testData.data[{{},{},{},{}}]:add(-mean)
testData.data[{{},{},{},{}}]:div(stdev)

print("\nTesting...")
scores = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
k=1
for i=1, math.ceil(testData:size()/minibatchData:size()) do
    minibatchData = batch.make_minibatch(testData, minibatchData, i)

    local guess = net:forward(minibatchData.data)

    for j=1, minibatchData:size() do
        local truth = testData[k][2]

        local confidences, indices = torch.sort(guess[j], true)

        counts[truth] = counts[truth] + 1
        if truth == indices[1] then
            scores[truth] = scores[truth] + 1
        end

        k = k+1
    end
end

scorestr = ""
for i=1, 10 do
    scorestr = scorestr .. tonumber(string.format("%.2f",100*scores[i]/counts[i])) .. "% "
end

print("Class Scores: " .. scorestr)

correct = 0
for i, s in ipairs(scores) do
    correct = correct + s
end

print("Score: " .. tonumber(string.format("%.2f", 100*correct/testData:size())) .. "%")
