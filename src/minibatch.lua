local M = {}

function M.make_minibatch(data, minidata, batchNum, shuffle)
    local batchSize = minidata:size()
    local start = (batchNum-1) * batchSize + 1
    
    if shuffle ~= nil then
        k=1
        for i=start, start+batchSize-1 do
            minidata.data[k] = data.data[shuffle[i]]
            minidata.labels[k] = data.labels[shuffle[i]]

            k=k+1
        end
    else
        minidata.data = data.data[{{start, start+batchSize-1}, {}, {}, {}}]:clone()
        minidata.labels = data.labels[{{start, start+batchSize-1}}]:clone()
    end

    return minidata
end

return M