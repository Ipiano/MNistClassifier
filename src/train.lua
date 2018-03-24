require("nn")
require("optim")
require("torch")
prep = require("./prep")

local M = {}

function M.train_nn(net, criterion, data, epochs, learningRate, decay)
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = learningRate or 0.005
    trainer.learningRateDecay = decay or 5e-7
    trainer.maxIteration = epochs

    trainer:train(data)
end

-- Training method adapted from https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua
M._global_epochs = 1
function M.train_minibatch_nn(net, criterion, data, minibatch, epochs, logger, learningRate, momentum, decay, timer, learning_mul, learning_mod)
    -- Confusion Matrix to rank performance
    local classes = {'1','2','3','4','5','6','7','8','9','10'}
    local matrix = optim.ConfusionMatrix(classes)

    -- Starting values of gradiant parameters
    parameters, gradParameters = net:getParameters()

    -- Init all variables needed for profiling
    local times = torch.Tensor(6)
    local pTime = 0;
    local time = 0;
    local clockCount = 0;

    -- Create clock function that can be called
    -- during sgd to update timing variables
    -- Default it to empty so it doesn't take cpu time
    -- when profiling not requested
    local clock = function(i, inc) end

    if timer then
        -- Check if cutorch is available
        -- if so, we synchronize at each step to make sure
        -- we get accurate timings when using the GPU
        local hascutorch, cutorch = pcall(require,"cutorch")
        clock = function(i, inc)
            if hascutorch then
                cutorch.synchronize()
            end
                
            time = sys.clock()
            times[i] = (times[i] or 0) + (time - pTime)
            pTime = time

            if inc then clockCount = clockCount + 1 end
        end
    end

    -- Closure to run sgd on a minibatch
    local trainBatch = function(mini)

        -- Closure to evaluate f(X) and df/dX for a minibatch
        -- this is used by the gradient descent
        local feval = function(x)
            clock(1)

            -- reset gradients
            gradParameters:zero()

            -- evaluate function for complete mini batch
            local outputs = net:forward(mini.data)
            clock(2)
            local f = criterion:forward(outputs, mini.labels)
            clock(3)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, mini.labels)
            clock(4)
            net:backward(mini.data, df_do)
            clock(5)

            matrix:batchAdd(outputs, mini.labels)
            clock(6, true)

            -- return f and df/dX
            return f,gradParameters
        end

        -- Perform SGD step on current mini-batch:
        sgdState = sgdState or {
            learningRate = (learningRate or 0.005) * math.pow(learning_mul or 1, math.floor((M._global_epochs-2) / (learning_mod or 100))),
            momentum = momentum or 0,
            learningRateDecay = decay or 5e-7
        }
        pTime = sys.clock()
        optim.sgd(feval, parameters, sgdState)
    end

    for e=1, epochs or 25 do

        --Generate random suffle of data
        shuffle = torch.randperm(data:size(), 'torch.LongTensor')

        --Increment epoch counter
        print("\n# epoch "..M._global_epochs)
        M._global_epochs = M._global_epochs+1

        --Reset timing data
        times:zero()
        pTime = 0;
        time = 0;
        clockCount = 0;

        --Run the trainBatch closure above for
        --each minibatch in the source data set, using the given shuffle
        --Data will be stored into the minibatch object passed
        --in, so minimal allocations are needed
        minibatch:forMinibatches(trainBatch, data, true, shuffle)
    
        --After training, output profile info if requested
        if timer then
            local str = ""
            for i=1, times:size(1) do
                str = str..((times[i]/clockCount)*1000).." ms\t"
            end
            print("Timepoints: "..str)
        end

        --Output score
        matrix:updateValids()
        local score = matrix.totalValid * 100
        if logger ~= nil then
            logger:add{['% mean class accuracy (train set)'] = score}
        end
        print("Score: "..score.."%")
        matrix:zero()
    end
end

return M
