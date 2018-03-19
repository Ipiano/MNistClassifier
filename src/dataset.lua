local M = {}

M.make_dataset = function(data, labels)
    --The StochasticGradient object requires a specific type of dataset
    --specifically, the data objct just have a size() method, and be indexable

    --Create the base object with data and labels
    --as well as size()
    local out = {
        data = data,
        labels = labels,

        size = function(self) 
            return self.data:size(1) 
        end
    }

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