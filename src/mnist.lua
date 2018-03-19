require("struct")
require("torch")
require("itorch")

local data = require("./dataset")

local M = {}

--[[Reads a set of images from a binary ubtye file --]]
function M.read_images( fileName, dataSize )
    local inp = assert(io.open(fileName, "rb"), "Unable to open image file " .. fileName)

    local MAGICNUMBER = 2051

    local header = inp:read(16)
    local magicMatch, count, h, w = struct.unpack(">i4i4i4i4", header)

    assert(magicMatch == MAGICNUMBER, "Magic number mismatch in " .. fileName .. ": " .. MAGICNUMBER .. " vs. " .. magicMatch)

    print(count .. " items size " .. w .. "x" .. h .. " in file " .. fileName)
    count = math.min(count, dataSize)

    local images = {}

    for i=1, count do
        local image = {}
        for j=1, h do
            local row = {}
            for k=1, w do
                row[k] = inp:read(1):byte()
            end
            image[j] = row
        end
        images[i] = {image}
    end

    local out = torch.Tensor(images)

    return out
end

--[[Reads a set of labels from a binary ubtye file --]]
function M.read_labels( fileName, dataSize )
    local inp = assert(io.open(fileName, "rb"), "Unable to open labels file " .. fileName)

    local MAGICNUMBER = 2049

    local header = inp:read(8)
    local magicMatch, count = struct.unpack(">i4i4", header)

    assert(magicMatch == MAGICNUMBER, "Magic number mismatch in " .. fileName .. ": " .. MAGICNUMBER .. " vs. " .. magicMatch)

    print(count .. " labels  in file " .. fileName)
    count = math.min(dataSize, count)

    local labels = {}
    
    --Gotta +1 all labels because they are the actual number there; but lua is 1-indexed
    --so label 0 is not valid
    inp:read(count):gsub(".", function(c) table.insert(labels, c:byte() + 1) end)

    local out = torch.Tensor(labels)

    return out
end

--[[Reads a pair image file and label file with the same base name, ending in '-images-idx3-ubyte' and '-labels-idx1-ubyte' respectively --]]
function M.read_data( fileBase, dataSize )
    local imgFile = fileBase .. "-images-idx3-ubyte"
    local labelFile = fileBase .. "-labels-idx1-ubyte"

    local images = M.read_images(imgFile, dataSize)
    local labels = M.read_labels(labelFile, dataSize)

    assert(images:size(1) == labels:size(1), "Image set and label set are different sizes")

    out = data.make_dataset(images, labels)

    return out
end

return M