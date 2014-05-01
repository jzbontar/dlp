require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'jzt'
require 'prof-torch'

cutorch.setDevice(1)
torch.manualSeed(42)

batch_size = 128
l1_coef = 1
learning_rate = 0.01

data_path = 'data_gcn_whitened/'
X = torch.FloatTensor(torch.FloatStorage(data_path .. 'X')):resize(60000, 3, 32, 32):cuda()

enc = nn.Sequential{bprop_min=1}
enc:add(nn.SpatialConvolutionRing2(3, 64, 5, 5))
enc:add(nn.Threshold())
enc:cuda()

enc_sparse = nn.Sequential()
enc_sparse:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2))
enc_sparse:cuda()

enc_reconstruct = nn.Sequential()
enc_reconstruct:add(nn.SpatialZeroPadding(4, 4, 4, 4))
enc_reconstruct:add(nn.SpatialConvolutionRing2(64, 16, 5, 5))
enc_reconstruct:cuda()

reconstruct_cost = nn.MSECriterion():cuda()
l1cost = jzt.L1Cost():cuda()

layers = {}
for _, net in ipairs({enc,enc_reconstruct,enc_sparse}) do
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
end
-- print(('Number of layers with parameters: %d'):format(#layers))

t = prof.time()
for epoch = 1,100 do
   -- train
   for t = 1,50000 - batch_size, batch_size do
      X_batch = X:narrow(1, t, batch_size)
      enc:forward(X_batch)

      enc_sparse:forward(enc.output)
      l1cost:forward(enc_sparse.output)
      l1cost:backward(enc_sparse.output)
      enc_sparse:backward(enc.output, l1cost.gradInput)

      enc_reconstruct:forward(enc.output)
      out = enc_reconstruct.output:narrow(2, 1, 3)
      reconstruct_cost:forward(out, X_batch)
      reconstruct_cost:backward(out, X_batch)
      enc_reconstruct:backward(enc.output, reconstruct_cost.gradInput)

      gradInput = enc_sparse.gradInput:add(enc_reconstruct.gradInput)
      enc:backward(X_batch, gradInput)

      ---


   end
   print(epoch, prof.time() - t)
end
