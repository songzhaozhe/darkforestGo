local classic = require 'classic'


local Master = classic.class('Master')


local function accmulate_gradients( ... )
    -- body
end

function Master:applyGradients()

  local feval = function()
    -- loss needed for validation stats only which is not computed for async yet, so just 0
    local loss = 0 -- 0.5 * tdErr ^2
    return loss, dTheta
  end

  --self.optimParams.learningRate = self.learningRateStart * (self.totalSteps - self.step) / self.totalSteps
  local optimiser_file = paths.concat('optimiser',self.optimiser)
  require (optimiser_file)
  self.optimiser(feval, theta, self.optimParams)

  dTheta:zero()
end


function Master:_init(opt,net,crit)
	-- body
	self.opt = opt
	self.net = net
	self.crit = crit
  	self.theta, self.dTheta = self.net:getParameters()	
  	self.dTheta:zero()
  	self.optimiser = opt.optimiser
  	local sharedG = self.theta:clone():zero()
  	self.optimParams = {
  		learningRate = opt.learningRate
  		momentum = opt.momentum
      	rmsEpsilon = opt.rmsEpsilon,	
      	g = sharedG
  	}


end


function Master:train()

	local epoch = 0
	local acc_errs = 0
	local t = 0
	local net = self.net
	local crit = self.crit
	while epoch < opt.maxepoch do
	    net:training()
	    acc_errs = 0
	    t = 0

	    for sample in state.iterator() do

			-- This includes forward/backward and parameter update.
			-- Different RL will use different approaches.
			net:forward(sample.s)
			local errs = crit:forward(net.output,sample.a)
			local grad = crit:backward(net.output,sample.a)

			net:backward(sample.s,grad)

	        acc_errs = acc_errs + errs

	        t = t + 1

	        self:applyGradients()


	    end

	    -- Update the sampling model.
	    -- state.agent:update_sampling_model(update_sampling_before)
	    state.agent:update_sampling_model()

	    state.epoch = state.epoch + 1

	end

end

function Master:test(opt)
	
end
