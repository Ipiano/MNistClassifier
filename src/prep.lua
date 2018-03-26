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

function M.transformImage(imageTensor, resultTensor, transformTensor1, transformTensor2, transformTensor3, mode)
    mode = mode or 'bilinear'

    transformTensor1:fill(0)
    local matrix2 = torch.Tensor(2, 2):fill(0)

    local angle = math.random() * math.pi/8 - math.pi/16

    --print("Rotate to angle "..angle*180/math.pi)
    transformTensor1[1][1] = math.cos(angle)
    transformTensor1[2][1] = math.sin(angle)
    transformTensor1[1][2] = -math.sin(angle)
    transformTensor1[2][2] = math.cos(angle)

    --print("Shear "..shearx ..", "..sheary)
    transformTensor2[2][1] = math.random()*0.25 - 0.125
    transformTensor2[1][2] = math.random()*0.25 - 0.125

    --print("Scale "..scalex..", "..scaley)
    transformTensor2[1][1] = 1/(math.random() * 0.25 + 0.75)
    transformTensor2[2][2] = 1/(math.random() * 0.25 + 0.75)

    transformTensor3:mm(transformTensor1, transformTensor2)

    return image.affinetransform(resultTensor, imageTensor, transformTensor1, mode)
end

function M.deformImage(imageTensor, resultTensor, dirTensor, fieldTensor, gaussTensor, sigma, alpha, mode)
    sigma = sigma or 4
    alpha = alpha or 34
    mode = mode or 'bilinear'

    dirTensor:uniform(-1, 1)
    image.gaussian{sigma=sigma, amplitude=1, tensor=gaussTensor}
    fieldTensor = image.convolve(fieldTensor, dirTensor, gaussTensor, "same")
    fieldTensor:div(fieldTensor:norm())
    fieldTensor:mul(alpha)

    --[[Print vector field in gnuplot-readable form--
    print("X\tY\tdX\tdY")
    for i=1, fieldTensor:size(2) do
        for j=1, fieldTensor:size(2) do
            print(i.."\t"..j.."\t"..fieldTensor[2][i][j].."\t"..fieldTensor[1][i][j])
        end
    end--]]

    return image.warp(resultTensor, imageTensor, fieldTensor, mode)
end

local function operateOnSet(src, processImage, hitsPerImage, useGPU)
    --Process items 1000 at a time
    --And when processing each one, make at most 1000 modified elements at once
    local miniSet = data.Dataset(torch.Tensor(100, src.data:size(2), src.data:size(3), src.data:size(4)))
    local miniResults = torch.Tensor(100, src.data:size(2), src.data:size(3), src.data:size(4))

    if useGPU then
        require("cutorch")
        miniSet = miniSet:cuda()
        miniResults = miniResults:cuda()
    end

    local outputImages = torch.Tensor()
    local outputLabels = torch.Tensor()

    local batchNum = 1
    --Processing a single batch of the input
    local process1Batch = function(batch)

        --For each image, call process1Image
        --until it's returned enough total amount
        for i=1, batch:size() do
            local todo = hitsPerImage
            while todo > 0 do
                local did = math.min(todo, miniResults:size(1))
                for j=1, did do
                    processImage(batch.data[i], miniResults[j])
                end

                --Copy results off gpu
                outputImages = outputImages:cat(miniResults[{{1, did}}], 1)

                --Reduce target goal
                todo = todo - did
            end

            --Add a bunch of labels to label set
            outputLabels = outputLabels:cat(torch.Tensor(hitsPerImage):fill(batch.labels[i]), 1)
        end
        batchNum = batchNum + 1
    end

    --Load batches of original data onto gpu
    --and process them
    miniSet:forMinibatches(process1Batch, src, true)
    --print(outputImages:size())
    --print(outputLabels)
    return data.Dataset(outputImages, outputLabels)
end

function M.transformSet(dataSet, useGPU, transforms, mode)
    local transformTensor1 = torch.Tensor(2, 2)
    local transformTensor2 = torch.Tensor(2, 2)
    local transformTensor3 = torch.Tensor(2, 2)

    if useGPU then
        require("cutorch")
        transformTensor1 = transformTensor1:cuda()
        transformTensor2 = transformTensor2:cuda()
        transformTensor3 = transformTensor3:cuda()
    end

    local process1Image = function(image, storage)
        M.transformImage(image, storage, transformTensor1, transformTensor2, transformTensor3, mode)
    end

    return operateOnSet(dataSet, process1Image, transforms, useGPU)
end

function M.deformSet(dataSet, useGPU, deforms, sigma, alpha, mode)
    local dirTensor = torch.Tensor(2, dataSet.data:size(3), dataSet.data:size(4))
    local gaussTensor = torch.Tensor(dataSet.data:size(3), dataSet.data:size(4))
    local fieldTensor = torch.Tensor(2, dataSet.data:size(3), dataSet.data:size(4))

    if useGPU then
        require("cutorch")
        dirTensor = dirTensor:cuda()
        gaussTensor = gaussTensor:cuda()
        fieldTensor = fieldTensor:cuda()
    end

    local process1Image = function(image, storage)
        M.deformImage(image, storage, dirTensor, fieldTensor, gaussTensor, sigma, alpha, mode)
    end

    return operateOnSet(dataSet, process1Image, deforms, useGPU)
end

return M