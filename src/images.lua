local gm = require("graphicsmagick")

local M = {}

function M.save_tensor(path, tensor)
    local image = gm.Image(tensor, "I", "DHW")

    image:save(path)
end

return M