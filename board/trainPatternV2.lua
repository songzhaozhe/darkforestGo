require 'torch'
local tnt = require "torchnet"
local sgfloader = require'utils.sgf'
local pat = require'board.pattern_v2'
local board = require 'board.board'
local common = require'common.common'
local goutils = require 'utils.goutils'

local function protected_play(b, game)
    local x, y, player = sgfloader.parse_move(game:play_current(), false)
    if player ~= nil and board.play(b, x, y, player) then
        game:play_next()
        return true
    else
        return false
    end
end

function load_random_game(info, apply_random_moves)
    local game
    local sample_idx
    while true do
        sample_idx = math.random(info.dataset:size())
        local sample = info.dataset:get(sample_idx)
        --for k, v in pairs(sample) do
        --    sample = v
        --    break
        --end
        -- require 'fb.debugger'.enter()
        local content = sample.table.content
        local filename = sample.table.filename
        game = sgfloader.parse(content:storage():string(), filename)
        if game ~= nil and game:has_moves() and game:get_boardsize() == common.board_size and game:play_start() then
            board.clear(info.b)
            goutils.apply_handicaps(info.b, game)

            local game_play_through = true
            if apply_random_moves then
                local round = math.random(game:num_round()) - 1
                for j = 1, round do
                    if not protected_play(info.b, game) then
                        game_play_through = false
                        break
                    end
                end
            end
            if game_play_through then break end
        end
    end

    -- require 'fb.debugger'.enter()

    -- Get the final result of the game.
    game.player_won = game:get_result_enum()
    -- Then we start from the beginning.
    info.game = game
    info.sample_idx = sample_idx

    --[[
    print(string.format("New game: idx = %d/%d", self.sample_idx, self.dataset:size()))
    board.show(self.b, 'last_move')
    ]]
end


local pat_h = pat.init(nil,true,nil)
local grads =  pat.init_grads()
idr = tnt.IndexedDatasetReader('dataset/kgs_train.idx','dataset/kgs_train.bin')
local sSummary, pSummary
local info = {dataset=idr,b=board.new()}
for i = 1,20 do
    load_random_game(info,true)
    content = idr:get(i).table.content
    game = sgfloader.parse(content:storage():string(), filename)
    --print(game)
--    print(type(game))
    local opt = {komi = 6.5,rule = board.chinese_rule,player_won =info.game.player_won,iterations = 100, training = true}
    print(opt)
    sSummary,pSummary = pat.train_policy_gradient(pat_h, grads, info.b, opt, sSummary, pSummary)
    print(i)
    pat.update_params(pat_h)
    print(pat.params)
end

pat.save(pat_h,"firstTry2.bin")
