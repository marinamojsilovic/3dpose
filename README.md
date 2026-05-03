This project contains model, training and utilities used to estimate joint positions in 3d based on
2d joint location. 
The model has a transformer architecture and the input data comes from HUMAN3.6M dataset(preprocessed).

Work progress:
1. Model is training on ground truth 2d data, positional encoding used is sin positional encoding. 
   Basic training no early stop lr scheduler etc. Val loss after 100 epochs is 74mm.