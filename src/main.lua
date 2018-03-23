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

--Default to using all possible data
dataSize = 4294967296

--Default to 25 epochs
epochs = 25

for i=1, table.getn(arg) do

    --Make .bmp versions of first n training data
    if arg[i] == "--make_images" then
        if i < table.getn(arg) then
            printedData = tonumber(arg[i+1])
        end

        printedData = printedData or 100

        printedData = math.floor(printedData)
        assert(dataSize >= 0, "Must have a positive image print count")
    end

    --Limit training data to at most n
    if arg[i] == "--data_size" and i < table.getn(arg) then
        dataSize = math.floor(tonumber(arg[i+1]))
        assert(dataSize >= 0, "Must have a positive data set size")
    end

    --Do n epochs
    if arg[i] == "--epochs" and i < table.getn(arg) then
        epochs = math.floor(tonumber(arg[i+1]))
        assert(epochs >= 0, "Must have a positive number of epochs")
    end

    --Create n affine transformed versions of each training data
    if arg[i] == "--transforms" and i < table.getn(arg) then
        transforms = math.floor(tonumber(arg[i+1]))
        assert(transforms >= 0, "Must have a positive number of transforms")
    end

    --Create n elastic deformed versions of each training data
    --Optional sigma and alpha; default 34, 4
    if arg[i] == "--deforms" and i < table.getn(arg) then
        deforms = math.floor(tonumber(arg[i+1]))
        assert(deforms >= 0, "Must have a positive number of deforms")

        if(i < table.getn(arg)-1) then
            sigma = tonumber(arg[i+2])
        end

        if(i < table.getn(arg)-2) and sigma then
            alpha = tonumber(arg[i+3])
        end

        alpha = alpha or 34
        sigma = sigma or 4
    end

    --Multiply learning rate by a every b epochs
    --Default 0.3, 100
    if arg[i] == "--learning_multiplier" and i < table.getn(arg) then
        if(i < table.getn(arg)) then
            epoch_learning_mul = tonumber(arg[i+1])
        end

        if(i < table.getn(arg)-1) and epoch_learning_mul then
            epoch_learning_mod = tonumber(arg[i+2])
        end

        epoch_learning_mul = epoch_learning_mul or 0.3
        epoch_learning_mod = math.floor(epoch_learning_mod or 100)

        assert(epoch_learning_mul > 0, "Must have positive non-0 epoch learning multiplier")
        assert(epoch_learning_mod > 0, "Must have positive non-0 epoch learning mod")
    end

    --Run on GPU with cunn
    if arg[i] == "--cuda" then
        useGPU = true
    end

    --Output training and testing score after each epoch
    if arg[i] == "--graph" then
        makeGraph = true
    end

    --Output timing statistics of each part of training algorithm
    if arg[i] == "--profile" then
        profile = true
    end

    --Use miniBatches of size n
    --Default 10
    if arg[i] == "--minibatch" then
        if i < table.getn(arg) then
            miniBatch = math.floor(tonumber(arg[i+1]))
        end

        miniBatch = miniBatch or 10
        assert(epochs > 0, "Must have a positive non-0 size miniBatch")
    end
end

--Make logging folder based on parameters regardless of if it'll be used
logFolder = epochs.."-epochs_"..dataSize.."-data"
if miniBatch then
    logFolder = logFolder.."_"..miniBatch.."-minibatch"
end

if transforms then
    logFolder = logFolder.."_"..transforms.."-transforms"
end

if deforms then
    logFolder = logFolder.."_"..deforms.."-deforms-"..sigma.."-"..alpha
end

if epoch_learning_mul then
    logFolder = logFolder.."_learningmultiplier-"..epoch_learning_mul.."-"..epoch_learning_mod
end


print("Loading training data...")
trainData = mnist.read_data("./data/train", dataSize)

--Regardless of how much training data is used
--read the whole test set; training is the long part
print("\nLoading testing data...")
testData = mnist.read_data("./data/t10k", 4294967296)

print("\nCreating neural net...")
net = build.build_mnist_net()
crit = criterion.build_mnist_criterion()

print("\nCreating "..(transforms or 0).." transformed copies of each image...")
transformed = prep.transformSet(trainData, transforms or 0)

print("\nCreating "..(deforms or 0).." deformed copies of each image...")
deformed = prep.deformSet(trainData, deforms or 0, sigma, alpha)

if printedData then
    local images = require("./images")
    require "lfs"

    lfs.mkdir("./images")
    lfs.mkdir("./images/testingImages")
    lfs.mkdir("./images/trainingImages")
    lfs.mkdir("./images/deformedImages")
    lfs.mkdir("./images/transformedImages")

    print("\nSaving up to "..printedData.." images from each set")
    for i=1, printedData do
        if testData:size() >= i then
            images.save_tensor("./images/testingImages/"..i..".bmp", testData[i][1])
        end
        if trainData:size() >= i then
            images.save_tensor("./images/trainingImages/"..i..".bmp", trainData[i][1])
        end
        if transformed:size() >= i then
            images.save_tensor("./images/transformedImages/"..i..".bmp", transformed[i][1])
        end
        if deformed:size() >= i then
            images.save_tensor("./images/deformedImages/"..i..".bmp", deformed[i][1])
        end
    end
end

trainData.data:cat({transformed.data, deformed.data}, 1)
trainData.labels:cat({transformed.labels, deformed.labels}, 1)

--If no minibatch specified, make 'batch' of entire data set
miniBatch = miniBatch or trainData:size()

--Build data set to hold minibatches
miniData = torch.Tensor(miniBatch, 1, trainData[1][1]:size(2), trainData[1][1]:size(3))
miniLabels = torch.Tensor(miniBatch)
miniBatchData = data.make_dataset(miniData, miniLabels)

if useGPU then
    print("\nConverting to CUDA types...")
    local data = require("./dataset")
    require("cunn")

    miniBatchData.data = miniBatchData.data:cuda()
    miniBatchData.labels = miniBatchData.labels:cuda()
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
    trainLogger = optim.Logger(paths.concat("./logs/"..logFolder, 'train.log'))
    testLogger = optim.Logger(paths.concat("./logs/"..logFolder, 'test.log'))

    trainLogger:display(false)
    testLogger:display(false)

    for i=1, epochs do
        print("\nTraining...")
        train.train_minibatch_nn(net, crit, trainData, miniBatchData, 1, trainLogger, nil, nil, nil, profile, epoch_learning_mul, epoch_learning_mod)

        print("\nTesting...")
        test.test_nn(net, testData, miniBatchData, testLogger)
    end
else
    print("\nTraining...")
    train.train_minibatch_nn(net, crit, trainData, miniBatchData, epochs, nil, nil, nil, nil, profile, epoch_learning_mul, epoch_learning_mod)

    print("\nTesting...")
    test.test_nn(net, testData, miniBatchData, testLogger)
end
