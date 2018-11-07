local THNN = require 'nn.THNN'
local SmoothAndEdgeTerm, parent = torch.class('nn.SmoothAndEdgeTerm', 'nn.Criterion')

function SmoothAndEdgeTerm:__init(sigma_color,sigma_space,window_size,lp,w_smooth,w_edge,w_L2,isDetailEnhancement,isStylization)
  parent.__init(self)

  self.sigma_color = - 1 / (sigma_color*sigma_color*2)
  self.sigma_space = - 1 / (sigma_space*sigma_space*2)
  self.window_size = window_size
  self.lp = lp
  self.w_smooth = w_smooth
  self.w_edge = w_edge
  self.w_L2 = w_L2
  self.isDetailEnhancement = isDetailEnhancement
  self.isStylization = isStylization
end

function SmoothAndEdgeTerm:updateOutput(input, target)
  local input_cnn = input[{{},{1,3},{},{}}] 
  local input_edge = input[{{},{4},{},{}}]
  local target_yuv = target[{{},{1,3},{},{}}]
  local target_edge = target[{{},{4},{},{}}]
  local target_edge_mask = target[{{},{5},{},{}}]

  self.output_smooth = torch.CudaTensor():resizeAs(input_cnn):zero()
  self.smooth_mask_pre = torch.CudaTensor():resizeAs(input_edge):zero()
  self.smooth_mask = torch.CudaTensor():resizeAs(input_edge):zero()
  self.w = torch.CudaTensor():zero()

  input.THNN.SmoothAndEdgeTerm_updateOutput(
    input_cnn:cdata(),
    input_edge:cdata(), 
    target_yuv:cdata(),
    target_edge:cdata(),
    target_edge_mask:cdata(),
    self.smooth_mask_pre:cdata(),
    self.smooth_mask:cdata(),
    self.w:cdata(),
    self.output_smooth:cdata(),
    self.sigma_color, 
    self.sigma_space, 
    self.window_size,
    self.lp,
    self.isDetailEnhancement,
    self.isStylization,
    self.w_L2
  )

  local loss_edge = 0
  if torch.sum(target_edge_mask) ~= 0 then
    local sub = torch.csub(input_edge,target_edge)
    self.gt = sub[target_edge_mask]
    loss_edge = torch.mean(torch.pow(self.gt,2))
  end
  local loss_smooth = torch.mean(self.output_smooth)

  self.output = loss_edge * self.w_edge + loss_smooth * self.w_smooth

  return self.output
end

function SmoothAndEdgeTerm:updateGradInput(input, target)
  local input_cnn = input[{{},{1,3},{},{}}]
  local input_edge = input[{{},{4},{},{}}]
  local target_yuv = target[{{},{1,3},{},{}}]
  local target_edge = target[{{},{4},{},{}}]
  local target_edge_mask = target[{{},{5},{},{}}]
  local gradInput_smooth = torch.CudaTensor():resizeAs(input_cnn):zero()
  local gradInput_edge = torch.CudaTensor():resizeAs(input_edge):zero()

  input.THNN.SmoothAndEdgeTerm_updateGradInput(
    input_cnn:cdata(),
    self.smooth_mask:cdata(),
    target_edge_mask:cdata(),
    self.w:cdata(),
    gradInput_smooth:cdata(),
    self.sigma_color, 
    self.window_size,
    self.lp,
    self.w_L2
  )

  if torch.sum(target_edge_mask) ~= 0 then
    gradInput_edge[target_edge_mask] = 2*self.gt/self.gt:size(1)
  end

  self.gradInput = torch.CudaTensor():resizeAs(input):zero()
  self.gradInput[{{},{1,3},{},{}}] = self.w_smooth * gradInput_smooth / (3 * input_cnn:size(3) * input_cnn:size(4))
  self.gradInput[{{},{4},{},{}}] = self.w_edge * gradInput_edge

  return self.gradInput
end

function SmoothAndEdgeTerm:clearState()
  nn.utils.clear(self, 'w', 'smooth_mask_pre', 'smooth_mask', 'gt', 'output_smooth')
  return parent.clearState(self)
end