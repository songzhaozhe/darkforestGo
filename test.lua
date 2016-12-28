--local sgfloader = require 'utils.sgf'
local tnt = require'torchnet'
--print(sgfloader.parse(io.open("sgfs/alphago_leesedol_1.sgf"):read("*a")))
idr = tnt.IndexedDatasetReader('dataset/kgs_train.idx','dataset/kgs_train.bin')
idr.get(2)
