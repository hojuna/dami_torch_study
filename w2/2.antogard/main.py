import torch
from torchvision.models import resnet18, ResNet18_Weights



"""
pytorch에서 사용법
"""
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # 순전파 단계(forward pass)

loss = (prediction - labels).sum()
loss.backward() # 역전파 단계(backward pass)


optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


optim.step() # 경사하강법(gradient descent)


"""
Autograd에서 미분
"""


a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# 수집된 변화도가 올바른지 확인합니다.
print(9*a**2 == a.grad)
print(-2*b == b.grad)

# tensor([True, True])
# tensor([True, True])

"""
DAG에서 제외하기
"""

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients?: {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

# Does `a` require gradients?: False
# Does `b` require gradients?: True