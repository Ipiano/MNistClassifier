require("nn")

local M = {}

function M.train_nn(net, criterion, data)
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 10

    trainer:train(data)
end

return M