require("nn")

function build_criterion()
    --Build a negative log-likelihood criterion
    criterion = nn.ClassNLLCriterion()
    
    return criterion
end
