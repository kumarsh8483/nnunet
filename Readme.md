# ML Model Management Implementation of MIC-DKFZ/nnUNet

## Environment Variable Defined Data Mounts

1) nnUNet_raw_data_base: This is where nnU-Net finds the raw data and stored the cropped data. The folder located at nnUNet_raw_data_base must have at least the subfolder nnUNet_raw_data, which in turn contains one subfolder for each Task.

    This is set to `/input` in this image, so the directory containing the task dataset should be mounted as follows (assuming it is named "Task03_Liver"):

    ```
    /input/nnUNet_raw_data/Task_03_Liver/
        dataset.json
        imagesTr/
            img_001_0000.nii.gz
            img_002_0000.nii.gz
            ...
        imagesTs/
            img_003_0000.nii.gz
            img_004_0000.nii.gz
            ...
        labelsTr/
            img_001.nii.gz
            img_002.nii.gz
            ...
    ```

2) nnUNet_preprocessed: This is the folder where the preprocessed data will be saved. The data will also be read from this folder during training.

    This is set to `/preprocessed` in this image.

3) RESULTS_FOLDER: This specifies where nnU-Net will save the model weights. If pretrained models are downloaded, this is where it will save them.

    This is set to `/output` in this image.

## DICOM Dataset Conversion

If you need to convert DICOMs to the proper input format, use the [make_dataset.py](make_dataset.py) script:

```bash
usage: make_dataset.py [-h] [--dataset DATASET] [--input INPUT] [--output OUTPUT] [--split SPLIT] [--list] [--structures STRUCTURES] [--task-name TASK_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to JSON file that defines the task variables (in dataset.json)
  --input INPUT         Path to input directory containing DICOMs and RT Structs
  --output OUTPUT       Path to output directory
  --split SPLIT         Percent (integer) of the data that will go to the training set (default is 80)
  --list                Print out list of available RT structures and exit (does not create dataset)
  --structures STRUCTURES
  --task-name TASK_NAME
```

## Preprocess

Assuming you are still using the Task03_Liver dataset, run the following:

```bash
nnUNet_plan_and_preprocess -t 3 --verify_dataset_integrity [...ARGS]
```

Optional arguments:
```
usage: nnUNet_plan_and_preprocess [-h] [-t TASK_IDS [TASK_IDS ...]] [-pl3d PLANNER3D] [-pl2d PLANNER2D] [-no_pp] [-tl TL] [-tf TF] [--verify_dataset_integrity] [-overwrite_plans OVERWRITE_PLANS] [-overwrite_plans_identifier OVERWRITE_PLANS_IDENTIFIER]

Optional arguments:
  -h, --help            show this help message and exit
  -t TASK_IDS [TASK_IDS ...], --task_ids TASK_IDS [TASK_IDS ...]
                        List of integers belonging to the task ids you wish to run experiment planning and preprocessing for. Each of these ids must, have a matching folder 'TaskXXX_' in the raw data folder
  -pl3d PLANNER3D, --planner3d PLANNER3D
                        Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be
                        configured
  -pl2d PLANNER2D, --planner2d PLANNER2D
                        Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. Can be 'None', in which case this U-Net will not be configured
  -no_pp                Set this flag if you dont want to run the preprocessing. If this is set then this script will only run the experiment planning and create the plans file
  -tl TL                Number of processes used for preprocessing the low resolution data for the 3D low resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of RAM
  -tf TF                Number of processes used for preprocessing the full resolution data of the 2D U-Net and 3D U-Net. Don't overdo it or you will run out of RAM
  --verify_dataset_integrity
                        set this flag to check the dataset integrity. This is useful and should be done once for each dataset!
  -overwrite_plans OVERWRITE_PLANS
                        Use this to specify a plans file that should be used instead of whatever nnU-Net would configure automatically. This will overwrite everything: intensity normalization, network
                        architecture, target spacing etc. Using this is useful for using pretrained model weights as this will guarantee that the network architecture on the target dataset is the same as on
                        the source dataset and the weights can therefore be transferred. Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use the LiTS plans to run the
                        preprocessing of the HepaticVessel task. Make sure to only use plans files that were generated with the same number of modalities as the target dataset (LiTS -> BCV or LiTS ->
                        Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, LiTS has just one)). Also only do things that make sense. This functionality is beta withno support
                        given. Note that this will first print the old plans (which are going to be overwritten) and then the new ones (provided that -no_pp was NOT set).
  -overwrite_plans_identifier OVERWRITE_PLANS_IDENTIFIER
                        If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows where to look for the correct plans and data. Assume your identifier is called IDENTIFIER, the
                        correct training command would be: 'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER -pretrained_weights FILENAME
```

## Training

After preprocessing, you can run the training using:

```bash
for FOLD in $(seq 0 4); do
    nnUNet_train 3d_fullres nnUNetTrainerV2 3 $FOLD --npz [...ARGS]
done
```

Optional arguments:
```
usage: nnUNet_train [-h] [-val] [-c] [-p P] [--use_compressed_data] [--deterministic] [--npz] [--find_lr] [--valbest] [--fp32] [--val_folder VAL_FOLDER] [--disable_saving] [--disable_postprocessing_on_folds] [--val_disable_overwrite] [--disable_next_stage_pred] [-pretrained_weights PRETRAINED_WEIGHTS] network network_trainer task fold

positional arguments:
  network
  network_trainer
  task                  can be task name or task id
  fold                  0, 1, ..., 5 or 'all'

optional arguments:
  -h, --help            show this help message and exit
  -val, --validation_only
                        use this if you want to only run the validation
  -c, --continue_training
                        use this if you want to continue a training
  -p P                  plans identifier. Only change this if you created a custom experiment planner
  --use_compressed_data
                        If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data is much more CPU and RAM intensive and should only be used if you know what you are
                        doing
  --deterministic       Makes training deterministic, but reduces training speed substantially. I (Fabian) think this is not necessary. Deterministic training will make you overfit to some random seed. Don't
                        use that.
  --npz                 if set then nnUNet will export npz files of predicted segmentations in the validation as well. This is needed to run the ensembling step so unless you are developing nnUNet you should
                        enable this
  --find_lr             not used here, just for fun
  --valbest             hands off. This is not intended to be used
  --fp32                disable mixed precision training and run old school fp32
  --val_folder VAL_FOLDER
                        name of the validation folder. No need to use this for most people
  --disable_saving      If set nnU-Net will not save any parameter files (except a temporary checkpoint that will be removed at the end of the training). Useful for development when you are only interested in
                        the results and want to save some disk space
  --disable_postprocessing_on_folds
                        Running postprocessing on each fold only makes sense when developing with nnU-Net and closely observing the model performance on specific configurations. You do not need it when
                        applying nnU-Net because the postprocessing for this will be determined only once all five folds have been trained and nnUNet_find_best_configuration is called. Usually running
                        postprocessing on each fold is computationally cheap, but some users have reported issues with very large images. If your images are large (>600x600x600 voxels) you should consider
                        setting this flag.
  --val_disable_overwrite
                        Validation does not overwrite existing segmentations
  --disable_next_stage_pred
                        do not predict next stage
  -pretrained_weights PRETRAINED_WEIGHTS
                        path to nnU-Net checkpoint file to be used as pretrained model (use .model file, for example model_final_checkpoint.model). Will only be used when actually training. Optional. Beta. Use
                        with caution.
```

## Inference

Following training, the resulting model can be used for inference using the following command:

```bash
nnUNet_predict -i /path/to/test/images -o /path/to/output/directory -t 3 -f 0 1 2 3 4 [...ARGS]
```

```
usage: nnUNet_predict [-h] -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME [-tr TRAINER_CLASS_NAME] [-ctr CASCADE_TRAINER_CLASS_NAME] [-m MODEL] [-p PLANS_IDENTIFIER] [-f FOLDS [FOLDS ...]] [-z] [-l LOWRES_SEGMENTATIONS] [--part_id PART_ID] [--num_parts NUM_PARTS] [--num_threads_preprocessing NUM_THREADS_PREPROCESSING] [--num_threads_nifti_save NUM_THREADS_NIFTI_SAVE] [--disable_tta] [--overwrite_existing] [--mode MODE] [--all_in_gpu ALL_IN_GPU] [--step_size STEP_SIZE] [-chk CHK] [--disable_mixed_precision]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FOLDER, --input_folder INPUT_FOLDER
                        Must contain all modalities for each patient in the correct order (same as training). Files must be named CASENAME_XXXX.nii.gz where XXXX is the modality identifier (0000, 0001, etc)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        folder for saving predictions
  -t TASK_NAME, --task_name TASK_NAME
                        task name or task ID, required.
  -tr TRAINER_CLASS_NAME, --trainer_class_name TRAINER_CLASS_NAME
                        Name of the nnUNetTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution U-Net. The default is nnUNetTrainerV2. If you are running inference with the cascade and the
                        folder pointed to by --lowres_segmentations does not contain the segmentation maps generated by the low resolution U-Net then the low resolution segmentation maps will be automatically
                        generated. For this case, make sure to set the trainer class here that matches your --cascade_trainer_class_name (this part can be ignored if defaults are used).
  -ctr CASCADE_TRAINER_CLASS_NAME, --cascade_trainer_class_name CASCADE_TRAINER_CLASS_NAME
                        Trainer class name used for predicting the 3D full resolution U-Net part of the cascade.Default is nnUNetTrainerV2CascadeFullRes
  -m MODEL, --model MODEL
                        2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres
  -p PLANS_IDENTIFIER, --plans_identifier PLANS_IDENTIFIER
                        do not touch this unless you know what you are doing
  -f FOLDS [FOLDS ...], --folds FOLDS [FOLDS ...]
                        folds to use for prediction. Default is None which means that folds will be detected automatically in the model output folder
  -z, --save_npz        use this if you want to ensemble these predictions with those of other models. Softmax probabilities will be saved as compressed numpy arrays in output_folder and can be merged between
                        output_folders with nnUNet_ensemble_predictions
  -l LOWRES_SEGMENTATIONS, --lowres_segmentations LOWRES_SEGMENTATIONS
                        if model is the highres stage of the cascade then you can use this folder to provide predictions from the low resolution 3D U-Net. If this is left at default, the predictions will be
                        generated automatically (provided that the 3D low resolution U-Net network weights are present
  --part_id PART_ID     Used to parallelize the prediction of the folder over several GPUs. If you want to use n GPUs to predict this folder you need to run this command n times with --part_id=0, ... n-1 and
                        --num_parts=n (each with a different GPU (for example via CUDA_VISIBLE_DEVICES=X)
  --num_parts NUM_PARTS
                        Used to parallelize the prediction of the folder over several GPUs. If you want to use n GPUs to predict this folder you need to run this command n times with --part_id=0, ... n-1 and
                        --num_parts=n (each with a different GPU (via CUDA_VISIBLE_DEVICES=X)
  --num_threads_preprocessing NUM_THREADS_PREPROCESSING
                        Determines many background processes will be used for data preprocessing. Reduce this if you run into out of memory (RAM) problems. Default: 6
  --num_threads_nifti_save NUM_THREADS_NIFTI_SAVE
                        Determines many background processes will be used for segmentation export. Reduce this if you run into out of memory (RAM) problems. Default: 2
  --disable_tta         set this flag to disable test time data augmentation via mirroring. Speeds up inference by roughly factor 4 (2D) or 8 (3D)
  --overwrite_existing  Set this flag if the target folder contains predictions that you would like to overwrite
  --mode MODE           Hands off!
  --all_in_gpu ALL_IN_GPU
                        can be None, False or True. Do not touch.
  --step_size STEP_SIZE
                        don't touch
  -chk CHK              checkpoint name, default: model_final_checkpoint
  --disable_mixed_precision
                        Predictions are done with mixed precision by default. This improves speed and reduces the required vram. If you want to disable mixed precision you can set this flag. Note that yhis is
                        not recommended (mixed precision is ~2x faster!)
```
