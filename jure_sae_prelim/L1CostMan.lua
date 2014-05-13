local L1CostMan, parent = torch.class('nn.L1CostMan', 'nn.Criterion')

function L1CostMan:__init()
   parent.__init(self)
end

function L1CostMan:updateOutput(input)
   return input:norm(1) -- implemented in cutorch
end

function L1CostMan:updateGradInput(input)
   self.gradInput:resizeAs(input)
   return self.gradInput:sign(input) -- impl in cutorch
end
