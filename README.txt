Dataset link: https://www.kaggle.com/datasets/liucong12601/stanford-online-products-dataset
The processing part of the dataset is explained in detail in Part 2 below.

############################################################
#############Part 1: Introduction to Project Structure#############
############################################################
Introduce the files in order:

cirtorch: The train.exe file required to start training, as well as the folder containing its supporting files.

data: The database used for training and the folder containing the generated data files can theoretically start training directly.
*Product images from real e-commerce websites
*120,053 images(12 objects)
*Train: 59,551 images 
*Test: 60,502 images 
*All images are standardized to 224x224 pixels


Matching base: The database used for matching during UI program startup (for demonstration purposes).

my_test_gallery (abandoned): This is the dataset that was originally presented to the teacher, but it has now been discarded because the images inside are too cluttered and do not meet our testing standards. But I still decided to keep it here and enable it if necessary.

Parameter Statistics: This is the relevant program and result used for all our data integration. Our final data is summarized in the Parameter Statistic.xlsx file above, which can be viewed if needed.

qury image: Some photos are used for testing during UI program startup (for demonstration purposes).

Step3~10: All training was successful, including baseline and full models, as well as training logs.

create_sop_db.py: Scan the SOP image dataset folder, create an index and category label for each image, and then save it as a sop.pkl file.

create_sop_gnd.py: Read the validation set val from the previously generated sop.pkl, and then generate the Ground Truth file gnd_sop.pkl for the validation/testing phase.

desktop_app.py: The UI program used for project demonstration, if you want to change the read model, you can modify it at the read path within the program.

requirements.txt: Instructions on how to install dependent environments.

setup.py: The code used for automatically configuring the environment, but it was not actually used, so it can be ignored.



############################################################
##########Part 2:                                                                ##########
##########1. How to run each script step-by-step               ##########
##########2. Any configuration files or parameters needed ##########
##########3. Expected outputs at each stage                      ##########
############################################################

1. Sort the dataset according to the format of 'create_sop_db.py' and 'create_sop_gnd.py'

2. Run 'create_sop_db.py' and 'create_sop_gnd.py' in sequence, and the dataset will be automatically processed.

(PS: The first two steps in theory can be skipped, we have already dealt with them.)

3. Using the following code, start train.exe directly in Python (note that both the environment and the entire file need to be loaded) to begin training.

baseline model:
python -m cirtorch.examples.train ./exp_sop_resnet50_v5 --gpu-id 0 --training-dataset sop --test-datasets sop --arch resnet50 --pool gem --loss contrastive --optimizer adam --lr 5e-7 --batch-size 16 --image-size 224 --neg-num 5 --epochs 10 --test-freq 1 --loss-margin 0.85 --workers 4

full model:
python -m cirtorch.examples.train ./exp_sop_resnet50_v6_full_features --gpu-id 0 --training-dataset sop --test-datasets sop --arch resnet50 --pool gem --loss contrastive --optimizer adam --lr 5e-7 --batch-size 8 --image-size 224 --neg-num 5 --epochs 10 --test-freq 1 --loss-margin 0.85 --workers 4 --regional --whitening --local-whitening

PS: You can adjust the opening parameter to adjust the model save position, or adjust the number of training rounds through epochs.

4. After the training is completed, simply launch desktop.app. py to use the model for matching.
