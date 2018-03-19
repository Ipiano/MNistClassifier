require("torch")

local M = {}

function M.normalize(imageSet)
    mean = imageSet[{{},{},{},{}}]:mean()
    imageSet[{{},{},{},{}}]:add(-mean)

    stdv = imageSet[{{},{},{},{}}]:std()
    imageSet[{{},{},{},{}}]:div(stdv)
    
    return mean, stdv
end

return M