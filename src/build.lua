require("nn");

local M = {}

function M.build_mnist_net(batchnorm)
    net = nn.Sequential()
    
    --Add first layer of spatial convolution
    net:add(nn.SpatialConvolution(1, 5, 5, 5, 2, 2))

    --Batch-Norm and Non-Linearity
    if batchnorm then net:add(nn.SpatialBatchNormalization(5)) end
    net:add(nn.ReLU())
    
    --Add second layer of spatial convolution
    net:add(nn.SpatialConvolution(5, 50, 5, 5, 2, 2))

    --Batch-Norm and Non-Linearity
    if batchnorm then net:add(nn.SpatialBatchNormalization(50)) end
    net:add(nn.ReLU())
    
    --Reshape 3D tensor into 1D tensor for linear layers to use
    net:add(nn.View(50*5*5))

    --Two hidden layers, 1 output
    net:add(nn.Linear(50*5*5, 100))
    if batchnorm then net:add(nn.BatchNormalization(100)) end
    net:add(nn.ReLU())
    net:add(nn.Linear(100, 10))
    if batchnorm then net:add(nn.BatchNormalization(10)) end
    net:add(nn.ReLU())
    net:add(nn.LogSoftMax())

    return net
end

return M
