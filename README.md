# U-net in tensorflow
- A customed implementation of the U-net model for semantic segmenation. The model architecture is slightly modified, such as the number of channels for some layers. Most important, the loss function is a combination of pixelwise cross-entropy loss and IOU loss, which is different from loss used in the original paper.
- The project is developed on satelite imagery data, which is a five-class classification problem. The number of class can be changed in model.py accordingly in other scienario.

## Training
- Put train and val data under ./data directory (./data/train, ./data/trainannot, ./data/val, ./data/valannot).
- Run prepare.py to load train and val data and generating .csv files ready for training.
- Run train.py to start training. Default training settings: 500 epochs, 32 samples per batch, can be changed by calling flags. Model will be saved under the root directory

## Inference
- Put test data under ./testdata/raw directory
- Run test.py to load the trained model and do the inference. Results will be saved at ./testdata/pred
