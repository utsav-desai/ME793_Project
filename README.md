# ME793_Project

This project aims to create generative models for microstructure image generation.

## File Structure and How to run this code

* *data* directory contains data
* *checkpoints* directory contains saved model weights
* *config.json* contains training configurations such as learning rate, no. of epochs, etc..
* *ddpm.py* trains confitional diffusion model and saves model weighs in the directory *checkpoint*
* *inference.py* runs inference on trained model (Parameters can se be set in this file)
* *inference.ipynb* a demo created for generation


1. To run this code first train the model using *ddpm.py.* (Make sure *checkpoints* directory exists)
2. Run inference using inference.py or inference.ipynb
