require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'jzt'
require 'prof-torch'

cutorch.setDevice(arg[1])

torch.manualSeed(42)

batch_size = 128
wc = 0.001
momentum = 0.9
learning_rate = 0.001

data_path = 'data_gcn_whitened/'
X = torch.FloatTensor(torch.FloatStorage(data_path .. 'X')):resize(60000, 3, 32, 32):cuda()
y = torch.FloatTensor(torch.FloatStorage(data_path .. 'y')):cuda()

net = nn.Sequential{bprop_min=2,debug=0}
net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing2(3, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.LPPooling(64, 2, 2, 2, 2, 2))

net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing2(64, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.LPPooling(64, 2, 2, 2, 2, 2))

net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing2(64, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.LPPooling(64, 2, 2, 2, 2, 2))

-- print(net:cuda():forward(X:narrow(1, 1, 128)):size())

size = 64 * 4 * 4
net:add(nn.Reshape(size))
net:add(nn.Linear(size, 10))
net:add(nn.LogSoftMax())

measure = nn.ClassNLLCriterion()

net = net:cuda()
measure = measure:cuda()

layers = {}
for i = 1,net:size() do
   if net:get(i).weight then
      layer = net:get(i)
      layer.weight_v = torch.CudaTensor():resizeAs(layer.weight):zero()
      layer.bias_v = torch.CudaTensor():resizeAs(layer.bias):zero()
      layer.gradWeight:zero()
      layer.gradBias:zero()
      layers[#layers + 1] = layer
   end
end

t = prof.time()
for epoch = 1,100 do
   -- train
   for t = 1,50000 - batch_size,batch_size do
      X_batch = X:narrow(1, t, batch_size)
      y_batch = y:narrow(1, t, batch_size)
      net:forward(X_batch)
      measure:forward(net.output, y_batch)
      measure:backward(net.output, y_batch)
      net:backward(X_batch, measure.gradInput)

      -- gradient step
      for _, layer in ipairs(layers) do
         layer.weight_v:mul(momentum):add(-learning_rate * wc, layer.weight):add(-learning_rate, layer.gradWeight)
         layer.weight:add(layer.weight_v)

         layer.bias_v:mul(momentum):add(-learning_rate * wc, layer.bias):add(-learning_rate, layer.gradBias)
         layer.bias:add(layer.bias_v)

         -- zeroGradParameters
         layer.gradWeight:zero()
         layer.gradBias:zero()
      end
   end

   -- test
   if epoch % 5 == 0 then
      torch.save(('net/%05d'):format(epoch), net)
      err = 0
      for t = 50001,60000 - batch_size,batch_size do
         X_batch = X:narrow(1, t, batch_size)
         y_batch = y:narrow(1, t, batch_size)
         net:forward(X_batch)
         _, i = net.output:float():max(2)
         err = err + i:float():ne(y_batch:float()):sum()
      end
      print(epoch, err / 10000, prof.time() - t)
   end
end
