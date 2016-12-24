package.path = package.path .. ";../?.lua"

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
    --mode                (default 'train')
    --optimiser           (default 'sharedRmsProp')
    --maxepoch            (default 50)
    --feature_type        (default 'extended')
    --foldername         (default '')
]]
--add models
--local network_maker = require('train.rl_framework.examples.go.models.' .. opt.model_name)
--local network, crit_not_used, outputdim, monitor_list = network_maker({1, 25, 19, 19}, opt)
--local model = network:cuda()
opt.userank = true
opt.use_bn = opt.use_bn == 'true'
if (opt.foldername == '') then
    opt.foldername = opt.net
end
if (opt.feature_type == 'extended') then
    opt.input_feature_num = 25
elseif (opt.feature_type == 'ours') then
    opt.input_feature_num = 35
end


if not paths.dirp('experiments') then
    paths.mkdir('experiments')
end
paths.mkdir(paths.concat('experiments', opt.foldername))


local flog = logroll.file_logger(paths.concat('experiments', opt.foldername,'_log.txt'))
local plog = logroll.print_logger()
log = logroll.combine(flog, plog)

local regular_net = require(opt.net)

local model = regular_net(opt)

local net, crit = model:create_net_and_crit(opt)
net:cuda()
crit:cuda()

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


local callbacks = {
    forward_model_init = function(partition)
        local tnt = require 'torchnet'
        return tnt.IndexedDataset{
            fields = { opt.datasource .. "_" .. partition },
            path = '../dataset'
        }
    end,
    forward_model_generator = function(dataset, partition)
        local fm_go = require 'train.rl_framework.examples.go.fm_go'
        return fm_go.FMGo(dataset, partition, opt)
    end
}
local Master = require 'Master'

local master = Master(opt,net,crit,callbacks)

if opt.mode == 'train' then
    master:train()
elseif opt.mode == 'test' then
    master:test()
end




