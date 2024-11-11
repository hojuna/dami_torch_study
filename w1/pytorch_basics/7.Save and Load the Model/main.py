"""
Saving and Loading Model Weights
"""

# #Weights save
# model = models.vgg16(weights='IMAGENET1K_V1')
# torch.save(model.state_dict(), 'model_weights.pth')

# #Weights load
# model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
# model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
# model.eval()


"""Saving and Loading Models with Shapes"""
# #model save
# torch.save(model, 'model.pth')

# #model load
# model = torch.load('model.pth', weights_only=False)