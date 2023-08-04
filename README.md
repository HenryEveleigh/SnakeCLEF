# SnakeCLEF

This is a simple algorithm which illustrates the application of convolutional neural networks to image recognition problems in ecology. The task is taken from the 2023 SnakeCLEF competition, where the aim is to identify the species of a snake from an image and location data. The performance of the model is not great but is better than chance, showing that the algorithm does learn information about the data. We also demonstrate the use of Bayesian optimisation to optimise hyperparameters in a CNN. 

The following files and folders are required to run the algorithm:
* The SnakeCLEF 2023 small-size training set image folder should be stored as SnakeCLEF_training_set/SnakeCLEF2023-small_size.
* The SnakeCLEF 2023 small-size validation set image folder should be stored as SnakeCLEF_validation_set/SnakeCLEF2023-small_size.
* The SnakeCLEF 2023 training set metadata should be stored as SnakeCLEF2023-TrainMetadata-iNat.csv.
* The SnakeCLEF 2023 validation set metadata should be stored as SnakeCLEF2023-ValMetadata.csv.
* The SnakeCLEF 2023 venomous status list should be stored as venomous_status_list.csv.

All of these files are available from https://huggingface.co/spaces/competitions/SnakeCLEF2023. Note that the files are not secure, so you may get warnings from your browser when downloading them.
