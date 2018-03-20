require("nn")
require("optim")
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
function M.train_minibatch_nn(net, criterion, data, minidata, epochs, learningRate, momentum, decay)
    parameters, gradParameters = net:getParameters()

    for e=1, epochs or 25 do
        t = 0

        print("\n# epoch "..e)
        xlua.progress(t, data:size())
        for i=1, math.ceil(data:size()/minidata:size()) do
            minidata = batch.make_minibatch(data, minidata, i)

            -- create closure to evaluate f(X) and df/dX
            local feval = function(x)
                -- reset gradients
                gradParameters:zero()

                -- evaluate function for complete mini batch
                local outputs = net:forward(minidata.data)
                local f = criterion:forward(outputs, minidata.labels)

                -- estimate df/dW
                local df_do = criterion:backward(outputs, minidata.labels)
                net:backward(minidata.data, df_do)

                -- return f and df/dX
                return f,gradParameters
            end
    
            -- Perform SGD step on current mini-batch:
            sgdState = sgdState or {
                learningRate = learningRate or 0.005,
                momentum = momentum or 0,
                learningRateDecay = 5e-7
            }
            optim.sgd(feval, parameters, sgdState)

            --Show progress bar
            t = math.min(t + minidata:size(), data:size())
            xlua.progress(t, data:size())
        end
    end
end

return M