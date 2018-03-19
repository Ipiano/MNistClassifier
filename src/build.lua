require("nn");

local M = {}

function M.build_mnist_net()
    net = nn.Sequential()
    
    --Add first layer of spatial convolution
    net:add(nn.SpatialConvolution(1, 5, 13, 13))

    --Non-linearity and max-pooling
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    --Add second layer of spatial convolution
    net:add(nn.SpatialConvolution(5, 50, 5, 5))

    --Non-linearity and max-pooling
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    --Reshape 3D tensor into 1D tensor for linear layers to use
    net:add(nn.View(50*2*2))

    --Two hidden layers, 1 output
    net:add(nn.Linear(50*2*2, 100))
    net:add(nn.ReLU())
    net:add(nn.Linear(100, 10))
    net:add(nn.ReLU())
    net:add(nn.LogSoftMax())

    return net
end

return M