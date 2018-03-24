require("torch")

local M = {}
local hascunn, cunn = pcall(require,"cunn")

--Creates a dataset type from a Tensor of data
--and a Tensor of labels
M.Dataset = function(data, labels, defaultLimit)
    data = data or torch.Tensor((labels or torch.Tensor(1)):size(1), 1, 1, 1)
    labels = labels or torch.Tensor((data or torch.Tensor(1)):size(1))

    assert(data:size(1) == labels:size(1), "Inconsistent dataset sizes")

    --Create the base object with data and labels
    --as well as size()
    local out = {
        data = data,
        labels = labels,
        _limit = defaultLimit or data:size(1),

        --Gets the amount of data actually used
        size = function(self) 
            return math.min(self._limit, self.data:size(1))
        end,

        --Specifies that not all of the allocated space
        --is used
        limit = function(self, l)
            self._limit = l
        end,

        --Checks how much allocated space the dataset has
        reserved = function(self)
            return self.data:size(1)
        end,

        --Copies a subset of src into dst to make the nth minibatch
        --if shuffle is specified, it should be an array of indexes
        --for indexing into src
        minibatch = function(dst, src, batchNum, shuffle)
            local batchSize = dst:reserved()
            local startInd = (batchNum-1) * batchSize + 1
            local endInd = math.min(startInd + batchSize - 1, src:size())
        
            k=1
            for i=startInd, endInd do
                if shuffle then
                    dst.data[k] = src.data[shuffle[i]]
                    dst.labels[k] = src.labels[shuffle[i]]
                else
                    dst.data[k] = src.data[i]
                    dst.labels[k] = src.labels[i]
                end
        
                k=k+1
            end

            dst:limit(k)
        
            return dst
        end,

        --Perform some function for all minibatches in a set
        --Optionally have a message and progress bar
        forMinibatches = function (dst, lambda, src, progress, shuffle)
            local t = 0
        
            if progress then
                xlua.progress(t, src:size())
            end
        
            for i=1, math.ceil(src:size()/dst:reserved()) do
                dst:minibatch(src, i, shuffle)
        
                lambda(dst)
        
                t = math.min(t+dst:size(), src:size())
        
                if progress then
                    xlua.progress(t, src:size())
                end
            end
        end,

        float = function(self)
            return M.Dataset(self.data:float(), self.labels:float())
        end
    }

    --If cunn is available, add the :cuda() function
    if hascunn then
        out.cuda = function(self)
            return M.Dataset(self.data:cuda(), self.labels:cuda())
        end
    end

    --Add the metafunction __index to allow indexing
    setmetatable(out,
    {
        __index = function(self, i)
            return {self.data[i], self.labels[i]}
        end
    })

    return out
end

return M