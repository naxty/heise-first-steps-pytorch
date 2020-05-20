data = torch.randn(1,3)
weights = torch.randn(3,3)
bias = torch.ones(3, 1)
y = torch.mm(data, weights) + bias