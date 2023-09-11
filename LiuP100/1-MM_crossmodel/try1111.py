import torch

def forward_hook(module, input, output):
    print("Inside forward hook")
    print("Input: ", input)
    print("Output: ", output)

def backward_hook(module, grad_input, grad_output):
    print("Inside backward hook")
    print("Grad Input: ", grad_input)
    print("Grad Output: ", grad_output)

model = torch.nn.Sequential(
          torch.nn.Linear(20, 10),
          torch.nn.ReLU(),
          torch.nn.Linear(10, 5),
          torch.nn.ReLU(),
          torch.nn.Linear(5, 1),
        )

model[0].register_forward_hook(forward_hook)
model[0].register_backward_hook(backward_hook)

x = torch.randn(1, 20)
y = model(x)
y.backward()