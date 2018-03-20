require("nn");

local M = {}

function M.build_mnist_net()
    net = nn.Sequential()
    
    --Add first layer of spatial convolution
    net:add(nn.SpatialConvolution(1, 5, 5, 5, 2, 2))

    --Batch-Norm and Non-Linearity
    net:add(nn.SpatialBatchNormalization(5))
    net:add(nn.ReLU())
    
    --Add second layer of spatial convolution
    net:add(nn.SpatialConvolution(5, 50, 5, 5, 2, 2))

    --Batch-Norm and Non-Linearity
    net:add(nn.SpatialBatchNormalization(50))
    net:add(nn.ReLU())
    
    --Reshape 3D tensor into 1D tensor for linear layers to use
    net:add(nn.View(50*5*5))

    --Two hidden layers, 1 output
    net:add(nn.Linear(50*5*5, 100))
    net:add(nn.BatchNormalization(100))
    net:add(nn.ReLU())
    net:add(nn.Linear(100, 10))
    net:add(nn.BatchNormalization(10))
    net:add(nn.ReLU())
    net:add(nn.LogSoftMax())

    return net
end

return M
