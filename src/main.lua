local criterion = require("./criterion")
local prep = require("./prep")
local build = require("./build")
local test = require("./test")
local train = require("./train")
local mnist = require("./mnist")

print("Loading training data...")
trainset = mnist.read_data("./data/train", false)

print("\nLoading testing data...")
testset = mnist.read_data("./data/t10k", false)

print("\nNormalizing training data...")
mean, stdev = prep.normalize(trainset.data)

print("\nCreating neural net...")
net = build.build_mnist_net()
crit = criterion.build_mnist_criterion()

print("\nTraining...")
train.train_nn(net, crit, trainset)

print("\nNormalizing testing data...")
testset.data[{{},{},{},{}}]:add(-mean)
testset.data[{{},{},{},{}}]:div(stdv)

print("\nTesting...")
scores = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1, testset:size() do
    local truth = testset[i][2]
    local guess = net:forward(testset[i][1])
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

print("Score: " .. 100*correct/testset:size() .. "%")