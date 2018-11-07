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

-- structure extraction for edge-preserving smoothing
smoothing = '1'
-- structure extraction for texture removal
-- smoothing = '0'


imgPath = '/mnt/data/VOC2012_input/'

if smoothing == '1' then
  savePath = '/mnt/data/VOC2012_input_edge_default/'
else
  savePath = '/mnt/data/VOC2012_input_edge_texture/'
end

h0 = nn.Identity()()
h0_edge = h0 - nn.EdgeComputation()
h1 = {h0,h0_edge} - nn.JoinTable(2)
h2 = h1 - nn.EdgeDetector(smoothing)
model_edgeDetector = nn.gModule({h0},{h2})
model_edgeDetector = model_edgeDetector:cuda()

files = {}
for file in paths.files(imgPath) do
  if string.find(file,'.png') then
    table.insert(files, paths.concat(imgPath,file))
  end
end

for _,inputFile in ipairs(files) do
  
  local inputImg = image.load(inputFile)
  local savColor = string.gsub(inputFile,imgPath,savePath)

  local height = inputImg:size(2)
  local width = inputImg:size(3)

  local input = torch.CudaTensor(1, 3, height, width)
  input[1] = inputImg:cuda()
  input = input * 255

  edge_label = model_edgeDetector:forward(input)
  edge_label_preserve = edge_label[{{},{1},{},{}}]
  edge_label_eliminate = edge_label[{{},{2},{},{}}]

  local sav = string.gsub(savColor,'%.png','-edge.png')
  if smoothing == '1' then
    image.save(sav,edge_label_preserve[1])
  else
    image.save(sav,edge_label_eliminate[1])
  end
end