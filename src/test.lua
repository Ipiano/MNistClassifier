require("torch")

local M = {}

function M.test_nn(net, data, minidata, logger)
    scores = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    k=1
    t=1
    xlua.progress(t, data:size())
    for i=1, math.ceil(data:size()/minidata:size()) do
        minidata = batch.make_minibatch(data, minidata, i)
    
        local guess = net:forward(minidata.data)
    
        for j=1, minidata:size() do
            local truth = data[k][2]
    
            local confidences, indices = torch.sort(guess[j], true)
    
            counts[truth] = counts[truth] + 1
            if truth == indices[1] then
                scores[truth] = scores[truth] + 1
            end
    
            k = k+1
        end
        --Show progress bar
        t = math.min(t + minidata:size(), data:size())
        xlua.progress(t, data:size())
    end
    
    scorestr = ""
    for i=1, 10 do
        scorestr = scorestr .. tonumber(string.format("%.2f",100*scores[i]/counts[i])) .. "% "
    end
    
    print("Class Scores: " .. scorestr)
    
    correct = 0
    for i, s in ipairs(scores) do
        correct = correct + s
    end
    
    print("Score: " .. tonumber(string.format("%.2f", 100*correct/data:size())) .. "%") 

    if logger ~= nil then
        logger:add{['% mean class accuracy (test set)'] = 100*correct/data:size()}
    end
end

return M