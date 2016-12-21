require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'logroll'

local tnt = require 'torchnet'

local pl = require 'pl.import_into'()

local tnt = require 'torchnet'

local opt = pl.lapp[[
    --alpha          (default 0.1)
    --nthread        (default 8)
    --batchsize      (default 256)
    --num_forward_models  (default 4096)       Number of forward models.
    --progress                                 Whether to print the progress
    --epoch_size          (default 12800)      Epoch size
    --epoch_size_test     (default 128000)      Epoch size for test.
    --data_augmentation                        Whether to use data_augmentation

    --nGPU                (default 1)          Number of GPUs to use.
    --nstep               (default 3)          Number of steps.
    --model_name          (default 'model-12-parallel-384-n-output-bn')
    --datasource          (default 'kgs')
    --feature_type        (default 'extended')
    --net                 (default 'regular_net')
    --use_bn              (default 'false')     
    --maxepoch            (default 1000)
    --name                (default '')
]]

opt.use_bn = opt.use_bn == 'true'

local flog = logroll.file_logger(paths.concat('exp_'..opt.net..'_log.txt'))
local plog = logroll.print_logger()
log = logroll.combine(flog, plog)

local regular_net = require(opt.net)

local model = regular_net(opt)

local net, crit = model:create_net_and_crit(opt)
net:cuda()
crit:cuda()
local flog = logroll.file_logger(paths.concat('exp_'..opt.net..'_log.txt'))
local plog = logroll.print_logger()
log = logroll.combine(flog, plog)
log.info("haahh")
--local a = torch.ones(4,25,19,19)
-- print(#a)
-- local b = net:forward(a)
-- local y = torch.ones(4,3)
-- print(net.output)
-- print(y)
-- local errs = crit:forward(net.output,y)
-- print(errs)
-- local grad = crit:backward(net.output,y)
-- print(grad)
-- print(b)
local epoch = 0
local acc_errs = 0
local t = 0
-- while epoch < opt.maxepoch do
--     net:training()
--     acc_errs = 0
--     t = 0

--     for sample in state.iterator() do
--         print(sample)
--         state.sample = sample
--         self.hooks("onSample", state)

--        -- This includes forward/backward and parameter update.
--        -- Different RL will use different approaches.
--         local errs = state.agent:optimize(sample)
--         accumulate_errs(state, errs)

--         state.t = state.t + 1
--         self.hooks("onUpdate", state)
--     end

--     -- Update the sampling model.
--     -- state.agent:update_sampling_model(update_sampling_before)
--     state.agent:update_sampling_model()

--     state.epoch = state.epoch + 1
--     self.hooks("onEndEpoch", state)
-- end



