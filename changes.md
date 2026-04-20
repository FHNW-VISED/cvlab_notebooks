the training / validation data plot is nowhere near what I want, I want it clear that the training data is for changing the model weights, and the validation is during training, and the test is for after. I need a sort of diagram with a model and arrows.

minimize some of the code cells in colab to occupy less space, there should be a way to do that in colab web 

when using the CNN from scratch, ABSOLUTELY remove the dataset creation once again. make them work with the same dataloader and dataset as before to keep them comparable

after a person run a full training, for any of the models, save the results to a common tensorboard file the can plot at the end of the notebook. it must be clear in the naming from which model and settings it comes from. use this to facilitate the comparisons with previous runs. alternatively, save the results in a dict or json and display them nicely on a simple iterative cell.

reduce as much as possible the code duplications.

create some beginner friendly cells with minimal code where it's very clear what they need to do and see visually how it affects what they see. the idea is to keep the information overload low and only show one variable to change. of course there can be also other more complex cells like there are already.

I copy-pasted the whole DinoV3 section, adapt the headers so that it fits nicely with the rest of the extra sections. also remove the "try with your own image" from the Advanced section, it can be done by anyone.

must fix the data downloading, save the zip somewhere easy, also github, then extract it and make easy download and extract

the data will now be stored with git lfs in here: https://github.com/FHNW-VISED/cvlab_notebooks/blob/main/faces_dataset.zip --> to prepare the dataset, download it from here, unzip the file, and put it in the correct folder structure