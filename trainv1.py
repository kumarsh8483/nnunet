# training python script : train.py
import os
import subprocess
import socket
from typing import Union, Optional

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

nnUNet_raw_data_base=os.getenv('nnUNet_raw_data_base')
nnUNet_preprocessed=os.getenv('nnUNet_preprocessed')
RESULTS_FOLDER=os.getenv('RESULTS_FOLDER')

class Command():
    counter = 0

    def __init__(self, *cmd):
      self.cmd = cmd

    def run(self):
      result = subprocess.run(self.cmd, stdout=subprocess.PIPE)
      print(f"Command {self.counter} execution : ", end="")
      print(result.stdout.decode('ascii'))

def create_dataset_dirs():
  try :
    os.makedirs(os.path.join(nnUNet_raw_data_base, 'nnUNet_raw_data'))
    os.makedirs(nnUNet_preprocessed)
    os.makedirs(RESULTS_FOLDER)
  except FileExistsError:
    pass

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("-network", default="2d", required=False)
    parser.add_argument("-network_trainer", default="nnUNetTrainer", required=False)
    parser.add_argument("-task", help="can be task name or task id", default="Task005_Prostate", required=False)
    parser.add_argument("-fold", help='0, 1, ..., 5 or \'all\'', default='all', required=False) # should be fixed
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    parser.add_argument("--disable_validation_inference", required=False, action="store_true",
                        help="If set nnU-Net will not run inference on the validation set. This is useful if you are "
                             "only interested in the test set results and want to save some disk space and time.")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    args = parser.parse_args()

    task = args.task
    fold = 'all' #args.fold
    print("fold : ", fold)
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"
    
    class Custom_nnUNetTrainer(trainer_class):
        def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
          super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16)
          self.max_num_epochs = 1

    trainer = Custom_nnUNetTrainer(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                # new training without pretraine weights, do nothing
                pass

            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

        trainer.network.eval()

        if args.disable_validation_inference:
            print("Validation inference was disabled. Not running inference on validation set.")
        else:
            # predict validation
            trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                            run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                            overwrite=args.val_disable_overwrite)

        if network == '3d_lowres' and not args.disable_next_stage_pred:
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
  # prepare data
  create_dataset_dirs()

  Command('gdown', 'https://drive.google.com/uc?id=1vvgcavq_Za42T5YUVQZ2U6wgg4idq6Wy&confirm=t').run()
  Command('tar', '-xf', "Task05_Prostate.tar", "-C", f"{nnUNet_raw_data_base}/nnUNet_raw_data/").run()
  Command('nnUNet_convert_decathlon_task', '-i', f'{nnUNet_raw_data_base}/nnUNet_raw_data/Task05_Prostate').run()
  Command('nnUNet_plan_and_preprocess', '-t', '5', '--verify_dataset_integrity').run()

  run_training()