require("nn")

local M = {}

function M.train_nn(net, criterion, data, epochs)
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.005
    trainer.maxIteration = epochs

    trainer:train(data)
end

return M