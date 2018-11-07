require 'nn'
require 'optim'
require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'sys'
require 'cudnn'
require 'nngraph'
cudnn.fastest = true
cudnn.benchmark = true

local function subnet1()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1, 1))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet2()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 2, 2, 2, 2))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 2, 2, 2, 2))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet4()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 4, 4, 4, 4))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 4, 4, 4, 4))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet8()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 8, 8, 8, 8))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 8, 8, 8, 8))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet16()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 16, 16, 16, 16))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 16, 16, 16, 16))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

local function subnet32()

  sub = nn.Sequential()

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 32, 32, 32, 32))
  sub:add(cudnn.SpatialBatchNormalization(64))
  sub:add(cudnn.ReLU(true))

  sub:add(cudnn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 32, 32, 32, 32))
  sub:add(cudnn.SpatialBatchNormalization(64))

  cont = nn.ConcatTable()
  cont:add(sub)
  cont:add(nn.Identity())

  return cont
end

h0 = nn.Identity()()
h0_origin = nn.Identity()()
h1 = h0 - cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
h2 = h1 - cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
h3 = h2 - cudnn.SpatialConvolution(64, 64, 3, 3, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)

sub1 = h3 - subnet2() - nn.CAddTable() - cudnn.ReLU(true)
sub2 = sub1 - subnet2() - nn.CAddTable() - cudnn.ReLU(true)
sub3 = sub2 - subnet4() - nn.CAddTable() - cudnn.ReLU(true)
sub4 = sub3 - subnet4() - nn.CAddTable() - cudnn.ReLU(true)
sub5 = sub4 - subnet8() - nn.CAddTable() - cudnn.ReLU(true)
sub6 = sub5 - subnet8() - nn.CAddTable() - cudnn.ReLU(true)
sub7 = sub6 - subnet16() - nn.CAddTable() - cudnn.ReLU(true)
sub8 = sub7 - subnet16() - nn.CAddTable() - cudnn.ReLU(true)
sub9 = sub8 - subnet1() - nn.CAddTable() - cudnn.ReLU(true)
sub10 = sub9 - subnet1() - nn.CAddTable() - cudnn.ReLU(true)

h4 = sub10 - cudnn.SpatialFullConvolution(64, 64, 4, 4, 2, 2, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
h5 = h4 - cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1) - cudnn.SpatialBatchNormalization(64) - cudnn.ReLU(true)
h6 = h5 - cudnn.SpatialConvolution(64, 3, 1, 1)
h7 = {h6,h0_origin} - nn.CAddTable()
h7_edge = h7 - nn.EdgeComputation()
h7_grad = {h7,h7_edge} - nn.JoinTable(2)

model = nn.gModule({h0,h0_origin},{h7,h7_grad})
model = model:cuda()

criterion = nn.ParallelCriterion():add(nn.MSECriterion(),1):add(nn.SmoothAndEdgeTerm(0.1,7,10,0.8,1,0.1,5,0,1),1)
criterion = criterion:cuda()

model_computeEdge = nn.EdgeComputation()

for i,module in ipairs(model:listModules()) do
   local m = module
   if m.__typename == 'cudnn.SpatialConvolution' or m.__typename == 'cudnn.SpatialFullConvolution' then
      local stdv = math.sqrt(12/(m.nInputPlane*m.kH*m.kW + m.nOutputPlane*m.kH*m.kW))
      m.weight:uniform(-stdv, stdv)
      m.bias:zero()
   end
   if m.__typename == 'cudnn.SpatialBatchNormalization' then
      m.weight:fill(1)
      m.bias:zero()
   end
end



postfix = 'smooth_stylization'
max_iters = 30
batch_size = 1

model:training()
collectgarbage()

parameters, gradParameters = model:getParameters()

sgd_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-8,
  weightDecay = 0.0005,
  momentum = 0.9,
  dampening = 0,
  nesterov = true
}

adam_params = {
  learningRate = 0.01,
  weightDecay = 0.0005,
  beta1 = 0.9,
  beta2 = 0.999
}

rmsprop_params = {
  learningRate = 1e-2,
  weightDecay = 0.0005,
  alpha = 0.9
}

savePath = './smoothing/'

local file = './smoothing_codes/train_stylization.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
print('*******************train file*******************')
print(line)
print('*******************train file*******************')

local file = './data/VOC2012_train.txt'
local trainSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet

local file = './data/VOC2012_test.txt'
local testSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(testSet, line)
end
f:close()
local testsetSize = #testSet

local iter = 0
local epoch_judge = false
step = function(batch_size)
  local testCount = 1
  local current_loss = 0
  local current_testloss = 0
  local count = 0
  local testcount = 0
  batch_size = batch_size or 4
  local order = torch.randperm(trainsetSize)

  for t = 1,trainsetSize,batch_size do
    iter = iter + 1
    local size = math.min(t + batch_size, trainsetSize + 1) - t

    local feval = function(x_new)
      -- reset data
      if parameters ~= x_new then parameters:copy(x_new) end
      gradParameters:zero()

      local loss = 0
      for i = 1,size do
        local inputFile =  trainSet[order[t+i-1]]
        local inputEdgeFile = string.gsub(inputFile,'VOC2012_input','VOC2012_input_edge_default')
        inputEdgeFile = string.gsub(inputEdgeFile,'%.png','-edge.png')

        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local label_all = torch.CudaTensor(1, 5, height, width):fill(0)

        input[1] = tempInput
        local input_origin = input:clone()
        input = input * 255
        label = input:clone()
        local inputs = {input - 115,input}

        label_all[{{},{1},{},{}}] = image.rgb2y(tempInput)  
        label_all[{{},{2},{},{}}] = 0.492 * torch.csub(input_origin[{{},{3},{},{}}],label_all[{{},{1},{},{}}])
        label_all[{{},{3},{},{}}] = 0.877 * torch.csub(input_origin[{{},{1},{},{}}],label_all[{{},{1},{},{}}])
        label_all[{{},{4},{},{}}] = model_computeEdge:forward(input)
        label_all[{{},{5},{},{}}] = image.load(inputEdgeFile)
        local labels = {label,label_all}

        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        loss = loss + tempLoss
        local grad = criterion:backward(pred, labels)

        model:backward(inputs, grad)
      end
      gradParameters:div(size)
      loss = loss/size

      return loss, gradParameters
    end
    
    if epoch_judge then
      adam_params.learningRate = adam_params.learningRate*0.1
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params, adam_params)
      epoch_judge = false
    else
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params)
    end

    count = count + 1
    current_loss = current_loss + fs[1]
    print(string.format('Iter: %d Current loss: %4f', iter, fs[1]))

    if iter % 20 == 0 then
      local loss = 0
      for i = 1,size do
        local inputFile = testSet[testCount]
        local inputEdgeFile = string.gsub(inputFile,'VOC2012_input','VOC2012_input_edge_default')
        inputEdgeFile = string.gsub(inputEdgeFile,'%.png','-edge.png')

        local tempInput = image.load(inputFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local label_all = torch.CudaTensor(1, 5, height, width):fill(0)

        input[1] = tempInput
        local input_origin = input:clone()
        input = input * 255
        label = input:clone()
        local inputs = {input - 115,input}

        label_all[{{},{1},{},{}}] = image.rgb2y(tempInput)  
        label_all[{{},{2},{},{}}] = 0.492 * torch.csub(input_origin[{{},{3},{},{}}],label_all[{{},{1},{},{}}])
        label_all[{{},{3},{},{}}] = 0.877 * torch.csub(input_origin[{{},{1},{},{}}],label_all[{{},{1},{},{}}])
        label_all[{{},{4},{},{}}] = model_computeEdge:forward(input)
        label_all[{{},{5},{},{}}] = image.load(inputEdgeFile)
        local labels = {label,label_all}

        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        loss = loss + tempLoss
        testCount = testCount + 1
      end
      loss = loss/size
      testcount = testcount + 1
      current_testloss = current_testloss + loss

      print(string.format('TestIter: %d Current loss: %4f', iter, loss))
    end
  end

  return current_loss / count, current_testloss / testcount
end

netfiles = './smoothing/netfiles/'
timer = torch.Timer()
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local loss,testloss = step(batch_size,i)
    print(string.format('Epoch: %d Current loss: %4f', i, loss))
    print(string.format('Epoch: %d Current test loss: %4f', i, testloss))

    local filename = string.format('%smodel_%s_%d.net',netfiles,postfix,i)
    model:clearState()
    torch.save(filename, model)
    local filename = string.format('%sstate_%s_%d.t7',netfiles,postfix,i)
    torch.save(filename, adam_state_save)
    print('Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours')
  end
end
print('Time elapsed: ' .. timer:time().real/(3600*24) .. ' days')
