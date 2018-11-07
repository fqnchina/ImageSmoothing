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
local matio = require 'matio'
matio.compression = matio.ffi.COMPRESSION_wDiminish

imgPath = './images'
savePath = './test_stylization'

files = {}
for file in paths.files(imgPath) do
  if string.find(file,'input.png') then
    table.insert(files, paths.concat(imgPath,file))
  end
end

-- change the model file to any others to try different smoothing effects.
modelfile = './netfiles/model_smooth_stylization_30.net'
model = torch.load(modelfile)
model = model:cuda()
model:training()

for _,inputFile in ipairs(files) do
	print(inputFile)

	local inputImg = image.load(inputFile)
	local savColor = string.gsub(inputFile,imgPath,savePath)
	local height = inputImg:size(2)
	local width = inputImg:size(3)

	local input = torch.CudaTensor(1, 3, height, width)
	input[1] = inputImg:cuda()
	local input_origin = input:clone()
	input = input * 255

	local inputs = {input - 115,input}
	predictions = model:forward(inputs)
	predictions_final = predictions[1]

	for m = 1,3 do
	  local numerator = torch.dot(predictions_final[1][m], input[1][m])
	  local denominator = torch.dot(predictions_final[1][m], predictions_final[1][m])
	  local alpha = numerator/denominator
	  predictions_final[1][m] = predictions_final[1][m] * alpha
	end

	predictions_final = predictions_final/255
	local sav = string.gsub(savColor,'.png','-predict_30.png')
	image.save(sav,predictions_final[1])
end