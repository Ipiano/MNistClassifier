local criterion = require("./criterion")
local prep = require("./prep")
local build = require("./build")
local test = require("./test")
local train = require("./train")
local mnist = require("./mnist")

printedData = 0
dataSize = 4294967296
useGPU = false
for i=1, table.getn(arg) do
    if arg[i] == "--make_images" then
        printedData = nil
        if i < table.getn(arg) then
            printedData = tonumber(arg[i+1])
        end

        if printedData == nil then
            printedData = 100
        end

        printedData = math.floor(printedData)
        assert(dataSize >= 0, "Must have a image print size")
    end

    if arg[i] == "--data_size" and i < table.getn(arg) then
        dataSize = math.floor(tonumber(arg[i+1]))
        assert(dataSize >= 0, "Must have a positive data set size")
    end

    if arg[i] == "--cuda" then
        useGPU = true
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
    lfs.mkdir("./images/testData")
    lfs.mkdir("./images/trainData")

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

print("\nNormalizing training data...")
mean, stdev = prep.normalize(trainData.data)

print("\nCreating neural net...")
net = build.build_mnist_net()
crit = criterion.build_mnist_criterion()

trainSet = trainData
if useGPU then
    local data = require("./dataset")
    require("cunn")

    trainSet = data.make_dataset(trainData.data:cuda(), trainData.labels:cuda())
    net = net:cuda()
    crit = crit:cuda()
end

print("\nTraining...")
train.train_nn(net, crit, trainSet)

print("\nNormalizing testing data...")
testData.data[{{},{},{},{}}]:add(-mean)
testData.data[{{},{},{},{}}]:div(stdv)

print("\nTesting...")
scores = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1, testData:size() do
    local truth = testData[i][2]
    local guess = net:forward(testData[i][1])
    local confidences, indices = torch.sort(guess, true)

    if truth == indices[1] then
        scores[truth] = scores[truth] + 1
    end
end

scorestr = ""
for i, s in ipairs(scores) do
    scorestr = scorestr .. s .. " "
end

print("Class Scores: " .. scorestr)

correct = 0
for i, s in ipairs(scores) do
    correct = correct + s
end

print("Score: " .. 100*correct/testData:size() .. "%")