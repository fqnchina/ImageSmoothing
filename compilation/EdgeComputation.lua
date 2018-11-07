local THNN = require 'nn.THNN'
local EdgeComputation, parent = torch.class('nn.EdgeComputation', 'nn.Module')

function EdgeComputation:__init(scale)
  parent.__init(self)
  self.scale = scale or 1
end

function EdgeComputation:updateOutput(input)
  self.scale = self.scale or 1
  input = input / self.scale

  local bs,dim,height,width = input:size(1),input:size(2),input:size(3),input:size(4)
  input = torch.sum(input,2)
  self.output = torch.CudaTensor():resizeAs(input):fill(0)

  input.THNN.EdgeComputation_updateOutput(
    input:cdata(),
    self.output:cdata()
  )
  return self.output
end

function EdgeComputation:updateGradInput(input, gradOutput)
    local bs,dim,height,width = input:size(1),input:size(2),input:size(3),input:size(4)
    self.gradInput = torch.CudaTensor():resizeAs(gradOutput):zero()

    input.THNN.EdgeComputation_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata()
    )
    self.gradInput = torch.expand(self.gradInput,bs,dim,height,width) / self.scale
    
    return self.gradInput
end