local THNN = require 'nn.THNN'
local EdgeDetector, parent = torch.class('nn.EdgeDetector', 'nn.Module')

function EdgeDetector:__init(isSmoothing)
  parent.__init(self)
  self.isSmoothing = isSmoothing
end

function EdgeDetector:updateOutput(input)
  local input_image = input[{{},{1,3},{},{}}]
  local input_edge = input[{{},{4},{},{}}]
  local bs,dim,height,width = input_edge:size(1),input_edge:size(2),input_edge:size(3),input_edge:size(4)

  input_image = torch.sum(input_image,2)/3
  self.output_preserve = torch.CudaTensor():resizeAs(input_edge):fill(0)
  self.output_eliminate = torch.CudaTensor():resizeAs(input_edge):fill(0)
  self.output = torch.CudaTensor():resize(bs,dim*2,height,width):fill(0)

  input.THNN.EdgeDetector_updateOutput(
    input_image:cdata(),
    input_edge:cdata(),
    self.output_preserve:cdata(),
    self.output_eliminate:cdata(),
    self.isSmoothing
  )

  self.output[{{},{1},{},{}}] = self.output_preserve
  self.output[{{},{2},{},{}}] = self.output_eliminate

  return self.output
end