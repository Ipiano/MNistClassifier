require("torch")
require("optim")

local M = {}

function M.test_nn(net, data, minidata, logger)
    local scores = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    local counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    local k=1

    -- Closure to test a specific minibatch
    local testBatch = function(mini)
    
        local guess = net:forward(mini.data)

        for j=1, mini:size() do
            local truth = data[k][2]
    
            local confidences, indices = torch.sort(guess[j], true)
    
            counts[truth] = counts[truth] + 1
            if truth == indices[1] then
                scores[truth] = scores[truth] + 1
            end
    
            k = k+1
        end
    end

    -- Test all minibatches, using minidata given as
    -- batch size and persistent storage
    minidata:forMinibatches(testBatch, data, true)
    
    -- Output scores
    local scorestr = ""
    for i=1, 10 do
        scorestr = scorestr .. tonumber(string.format("%.2f",100*scores[i]/counts[i])) .. "% "
    end
    
    print("Class Scores: " .. scorestr)
    
    local correct = 0
    for i, s in ipairs(scores) do
        correct = correct + s
    end
    
    print("Score: " .. tonumber(string.format("%.2f", 100*correct/data:size())) .. "%") 

    if logger ~= nil then
        logger:add{['% mean class accuracy (test set)'] = 100*correct/data:size()}
    end
end

return M
