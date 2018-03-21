require("nn")
require("optim")
require("torch")
batch = require("./minibatch")

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
function M.train_minibatch_nn(net, criterion, data, minidata, epochs, logger, learningRate, momentum, decay, timer)
    local classes = {'1','2','3','4','5','6','7','8','9','10'}
    local matrix = optim.ConfusionMatrix(classes)

    parameters, gradParameters = net:getParameters()

    local times = torch.Tensor(6)
    local pTime = 0;
    local time = 0;
    local clockCount = 0;
    local clock = function(i, inc) end
    local hascutorch, cutorch = pcall(require,"cutorch")

    if timer then
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

    for e=1, epochs or 25 do
        t = 0

        local shuffle = torch.randperm(data:size(), 'torch.LongTensor')

        print("\n# epoch "..M._global_epochs)
        M._global_epochs = M._global_epochs+1

        xlua.progress(t, data:size())

        times:zero()
        pTime = 0;
        time = 0;
        clockCount = 0;

        for i=1, math.ceil(data:size()/minidata:size()) do
            minidata = batch.make_minibatch(data, minidata, i, shuffle)

            -- create closure to evaluate f(X) and df/dX
            local feval = function(x)
                clock(1)

                -- reset gradients
                gradParameters:zero()

                -- evaluate function for complete mini batch
                local outputs = net:forward(minidata.data)
                clock(2)
                local f = criterion:forward(outputs, minidata.labels)
                clock(3)

                -- estimate df/dW
                local df_do = criterion:backward(outputs, minidata.labels)
                clock(4)
                net:backward(minidata.data, df_do)
                clock(5)

                matrix:batchAdd(outputs, minidata.labels)
                clock(6, true)

                -- return f and df/dX
                return f,gradParameters
            end
    
            -- Perform SGD step on current mini-batch:
            sgdState = sgdState or {
                learningRate = learningRate or 0.005,
                momentum = momentum or 0,
                learningRateDecay = decay or 5e-7
            }
            pTime = sys.clock()
            optim.sgd(feval, parameters, sgdState)

            --Show progress bar
            t = math.min(t + minidata:size(), data:size())
            xlua.progress(t, data:size())
        end

        if timer then
            local str = ""
            for i=1, times:size(1) do
                str = str..((times[i]/clockCount)*1000).." ms\t"
            end
            print("Timepoints: "..str)
        end

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