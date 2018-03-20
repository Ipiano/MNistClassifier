local criterion = require("./criterion")
local prep = require("./prep")
local build = require("./build")
local test = require("./test")
local train = require("./train")
local mnist = require("./mnist")
local batch = require("./minibatch")
local data = require("./dataset")

require("os")
require("optim")

math.randomseed(os.time())
torch.setdefaulttensortype('torch.FloatTensor')

printedData = 0
dataSize = 4294967296
useGPU = false
epochs = 25
transforms = 0
minibatch = -1
makeGraph = false
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

    if arg[i] == "--graph" then
        makeGraph = true
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

print("\nNormalizing testing data...")
testData.data[{{},{},{},{}}]:add(-mean)
testData.data[{{},{},{},{}}]:div(stdev)

if makeGraph then
    --Make loggers to output graph files
    trainLogger = optim.Logger(paths.concat("./logs", 'train.log'))
    testLogger = optim.Logger(paths.concat("./logs", 'test.log'))

    trainLogger:display(false)
    testLogger:display(false)
    
    for i=1, epochs do
        print("\nTraining...")
        train.train_minibatch_nn(net, crit, trainData, minibatchData, 1, trainLogger)

        print("\nTesting...")
        test.test_nn(net, testData, minibatchData, testLogger)
    end
else
    print("\nTraining...")
    train.train_minibatch_nn(net, crit, trainData, minibatchData, epochs)

    print("\nTesting...")
    test.test_nn(net, testData, minibatchData, testLogger)
end