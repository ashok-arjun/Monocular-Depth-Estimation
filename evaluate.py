import numpy as np
import torch 

from model.net import evaluate_predictions, combined_loss

def evaluate(model, dataloader_getter, batch_size):
  """
  Evaluates on a single iteration of the dataloader, with batch_size images
  :returns the loss value and metrics dict
  """
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = model.to(device)
  
  model.eval()

  test_dl = dataloader_getter(batch_size = batch_size, shuffle = True) 

  sample = next(iter(test_dl))
  images, depths = sample['img'], sample['depth']
  images = torch.autograd.Variable(images.to(device))
  depths = torch.autograd.Variable(depths.to(device))
  
  with torch.no_grad():
    predictions = model(images)

  loss = combined_loss(predictions, depths)
  metrics = evaluate_predictions(predictions, depths)

  return images, depths, predictions, loss, metrics
