require("torch")
require("image")

local data = require("./dataset")

local M = {}

function M.normalize(dataSet)
    local mean = dataSet[{{},{},{},{}}]:mean()
    dataSet[{{},{},{},{}}]:add(-mean)

    local stdv = dataSet[{{},{},{},{}}]:std()
    dataSet[{{},{},{},{}}]:div(stdv)
    
    return mean, stdv
end

function M.transformImage(imageTensor, mode)
    local matrix1 = torch.Tensor(2, 2):fill(0)
    local matrix2 = torch.Tensor(2, 2):fill(0)

    local angle = math.random() * math.pi/8 - math.pi/16

    --print("Rotate to angle "..angle*180/math.pi)
    matrix1[1][1] = math.cos(angle)
    matrix1[2][1] = math.sin(angle)
    matrix1[1][2] = -math.sin(angle)
    matrix1[2][2] = math.cos(angle)

    local shearx = math.random()*0.25 - 0.125
    local sheary = math.random()*0.25 - 0.125

    --print("Shear "..shearx ..", "..sheary)
    matrix2[2][1] = shearx
    matrix2[1][2] = sheary

    local scalex = math.random() * 0.25 + 0.75
    local scaley = math.random() * 0.25 + 0.75

    --print("Scale "..scalex..", "..scaley)
    matrix2[1][1] = 1/scalex
    matrix2[2][2] = 1/scaley

    local matrix = torch.mm(matrix1, matrix2)

    return image.affinetransform(imageTensor, matrix, mode)
end

function M.deformImage(imageTensor, sigma, alpha, mode)
    sigma = sigma or 4
    alpha = alpha or 34
    mode = mode or 'bilinear'

    local field = torch.Tensor(2, imageTensor:size(2), imageTensor:size(3)):uniform(-1, 1)

    local gaussian = image.gaussian(field:size(2), sigma, 1)

    field = image.convolve(field,gaussian, 'same')

    field:div(field:norm())
    field:mul(alpha)

    --[[Print vector field in gnuplot-readable form
    print("X\tY\tdX\tdY")
    for i=1, field:size(2) do
        for j=1, field:size(2) do
            print(i.."\t"..j.."\t"..field[2][i][j].."\t"..field[1][i][j])
        end
    end--]]

    return image.warp(imageTensor, field, mode)
end

function M.transformSet(dataSet, transforms, mode)
    mode = mode or 'bilinear'

    local newData = torch.Tensor(dataSet.data:size(1) * transforms, dataSet.data:size(2), dataSet.data:size(3), dataSet.data:size(4))

    local newLabels = torch.Tensor(dataSet.labels:size(1) * transforms)

    local newI = 1
    for i=1, dataSet:size() do
        for j=1, transforms do
            newData[newI] = M.transformImage(dataSet[i][1], mode)
            newLabels[newI] = dataSet[i][2]

            newI = newI+1
        end
    end

    return data.make_dataset(newData, newLabels)
end

function M.deformSet(dataSet, deforms, sigma, alpha)
    local newData = torch.Tensor(dataSet.data:size(1) * deforms, dataSet.data:size(2), dataSet.data:size(3), dataSet.data:size(4))

    local newLabels = torch.Tensor(dataSet.labels:size(1) * deforms)

    local newI = 1
    for i=1, dataSet:size() do
        for j=1, deforms do
            newData[newI] = M.deformImage(dataSet[i][1], sigma, alpha, "bilinear")
            newLabels[newI] = dataSet[i][2]

            newI = newI+1
        end
    end

    return data.make_dataset(newData, newLabels)
end

return M