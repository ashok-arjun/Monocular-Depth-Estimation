import torch
from model.loss import mean_l2_loss

def evaluate_predictions(predictions, truth):
  """
  Defines 5 metrics used to evaluate the depth estimation models
  """
  ratio_max = torch.max((predictions/truth),(truth/predictions)) #element wise maximum
  d1 = torch.mean((ratio_max < 1.25     ).float()) # mean of the bool tensor
  d2 = torch.mean((ratio_max < 1.25 ** 2).float()) # mean of the bool tensor
  d3 = torch.mean((ratio_max < 1.25 ** 3).float()) # mean of the bool tensor
  relative_error = torch.mean(torch.abs(predictions - truth)/truth)
  rmse = torch.sqrt(mean_l2_loss(predictions, truth))/100 # divide by 100 so that rmse is on data range(0-10)
  log10_error = torch.mean(torch.abs(torch.log10(predictions) - torch.log10(truth)))
  return {'d1_accuracy':d1, 'd2_accuracy':d2, 'd3_accuracy':d3, 'relative_err':relative_error, 'rmse':rmse, 'log10_error':log10_error}