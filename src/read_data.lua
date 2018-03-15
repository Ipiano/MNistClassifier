require("struct")

--[[Reads a set of images from a binary ubtye file --]]
function read_images( fileName, printData )
    local inp = assert(io.open(fileName, "rb"), "Unable to open image file " .. fileName)

    local MAGICNUMBER = 2051

    local header = inp:read(16)
    local magicMatch, count, h, w = struct.unpack(">i4i4i4i4", header)

    assert(magicMatch == MAGICNUMBER, "Magic number mismatch in " .. fileName .. ": " .. MAGICNUMBER .. " vs. " .. magicMatch)

    print(count .. " items size " .. w .. "x" .. h .. " in file " .. fileName)

    local images = {}

    for i=1, count do
        local image = {}
        for j=1, h do
            local row = {}
            for k=1, w do
                row[k] = inp:read(1)
            end
            image[j] = row
        end
        images[i] = image
    end

    if printData then
        for i=1, math.min(10, count) do
            for j=1, h do
                local row = ""
                for k=1, w do
                    row = row .. string.format("%3i", string.byte(images[i][j][k])) .. " "
                end
                print(row)
            end
            print("")
        end
    end

    return images
end

--[[Reads a set of labels from a binary ubtye file --]]
function read_labels( fileName, printData )
    local inp = assert(io.open(fileName, "rb"), "Unable to open image file " .. fileName)

    local MAGICNUMBER = 2049

    local header = inp:read(8)
    local magicMatch, count = struct.unpack(">i4i4", header)

    assert(magicMatch == MAGICNUMBER, "Magic number mismatch in " .. fileName .. ": " .. MAGICNUMBER .. " vs. " .. magicMatch)

    print(count .. " labels  in file " .. fileName)

    local labels = {};
    inp:read(count):gsub(".", function(c) table.insert(labels, c) end)

    if printData then
        for i=1, math.min(10, count) do
            print(string.byte(labels[i]))
        end
    end

    return labels
end

--[[Reads a pair image file and label file with the same base name, ending in '-images-idx3-ubyte' and '-labels-idx1-ubyte' respectively --]]
function read_data( fileBase, printData )
    local imgFile = fileBase .. "-images-idx3-ubyte"
    local labelFile = fileBase .. "-labels-idx1-ubyte"

    local images = read_images(imgFile, printData)
    local labels = read_labels(labelFile, printData)

    assert(table.getn(images) == table.getn(labels), "Image set and label set are different sizes")

    return {images, labels}
end