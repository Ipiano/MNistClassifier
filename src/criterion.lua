require("nn")

local M = {}

function M.build_mnist_criterion()
    --Build a negative log-likelihood criterion
    criterion = nn.ClassNLLCriterion()
    
    return criterion
end

return M