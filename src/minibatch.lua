local M = {}

function M.make_minibatch(data, minidata, batchNum)
    local batchSize = minidata:size()
    local start = (batchNum-1) * batchSize + 1
    
    minidata.data = data.data[{{start, start+batchSize-1}, {}, {}, {}}]:clone()
    minidata.labels = data.labels[{{start, start+batchSize-1}}]:clone()

    return minidata
end

return M