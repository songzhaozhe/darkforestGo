local nn = require 'nn'
require 'classic'
require 'classic.torch' -- Enables serialisation

local Body = classic.class('Body')
require 'ParallelCriterion2'
-- Constructor
function Body:_init()
end

function Body:create_net_and_crit(opt)
  -- Number of input frames for recurrent networks is always 1
	local input_feature_num = opt.input_feature_num;
	local net = nn.Sequential()  
	net:add(nn.SpatialConvolution(input_feature_num,92,5,5,1,1,2,2))
	net:add(nn.ReLU(true))
	if opt.use_bn then
		net:add(nn.SpatialBatchNormalization(92))
	end
	local cur_input_dim = 92
	for i = 1,10 do
		net:add(nn.SpatialConvolution(cur_input_dim,384,3,3,1,1,1,1))
		cur_input_dim = 384
		net:add(nn.ReLU(true))
		if opt.use_bn then
			net:add(nn.SpatialBatchNormalization(cur_input_dim))
		end 	
	end
	net:add(nn.SpatialConvolution(cur_input_dim,opt.nstep,3,3,1,1,1,1))
	cur_input_dim = opt.nstep

    local model = nn.Sequential()
    model:add(net):add(nn.View(opt.nstep, 19*19):setNumInputDims(3)):add(nn.SplitTable(1, 2))
    local softmax = nn.Sequential()
    -- softmax:add(nn.Reshape(19*19, true))
    softmax:add(nn.LogSoftMax())
    -- )View(-1):setNumInputDims(2))

    local softmaxs = nn.ParallelTable()
    -- Use self-defined parallel criterion 2, which can handle targets of the format nbatch * #target
    local criterions = nn.ParallelCriterion2()
    for k = 1, opt.nstep do
        softmaxs:add(softmax:clone())
        local w = 1.0 / k
        criterions:add(nn.ClassNLLCriterion(), w)
    end
    model:add(softmaxs)
  return model, criterions
end

return Body


