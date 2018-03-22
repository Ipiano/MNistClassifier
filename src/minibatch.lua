local M = {}

function M.make_minibatch(data, minidata, batchNum, shuffle)
    local batchSize = minidata:size()
    local start = (batchNum-1) * batchSize + 1
    
    k=1
    for i=start, math.min(start+batchSize-1, data:size()) do
        if shuffle ~= nil then
            minidata.data[k] = data.data[shuffle[i]]:clone()
            minidata.labels[k] = data.labels[shuffle[i]]
        else
            minidata.data[k] = data.data[i]:clone()
            minidata.labels[k] = data.labels[i]
        end
        k=k+1
    end

    return minidata
end

return M
