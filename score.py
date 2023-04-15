# TODO : remove of input and output directories if request fails by utilizing hooks
# TODO : load model from the init function in the future

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

import os
import shutil
import uuid
from time import time
import shutil 
import base64

from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, default_cascade_trainer, default_trainer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

inputs_root = "/var/azureml-app/avaj1870/inputs"
outputs_root = "/var/azureml-app/avaj1870/outputs"
task_name_cfg = "Task005_Prostate" #default_plans_identifier

def make_prediction(input_folder, output_folder):
    task_name = task_name_cfg
    network_dir = "nnunet_outputs/nnUNet"
    part_id = 0 # used when parallelization
    num_parts = 1 # used when parallelization
    folds = None # automatically detected if none
    save_npz = False # used when ensembling with other models
    lowres_segmentations = None
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    disable_tta = False
    step_size = 0.5
    overwrite_existing = False
    mode = "normal"
    all_in_gpu = None
    model = "2d"
    trainer_class_name = default_trainer
    cascade_trainer_class_name = default_cascade_trainer

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)


    model_folder_name = os.path.join(network_dir, model, task_name, trainer + "__" +
                                default_plans_identifier)
    print("using model stored in ", model_folder_name)
    assert os.path.isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    st = time()
    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=False,
                        step_size=step_size, checkpoint_name="model_best")
    end = time()
    print("prediction_time :", end - st)


def init():
    global model
    print("/ : ", os.listdir("/"))
    print("/var/azureml-app/avaj1870 : ", os.listdir("/var/azureml-app/avaj1870"))
    print("/var/azureml-app/avaj1870/src : ", os.listdir("/var/azureml-app/avaj1870/src"))
    # TODO: model will be loaded from here in the future
    # create input and output folder for inferece


# send files as a list in post request body
# json format data
@rawhttp
def run(request):
    # create unique input/output dirs for request 
    # (uniqueness needed to prevent possible race conditons)
    print("/ : ", os.listdir("/"))
    print("/var/azureml-app/avaj1870 : ", os.listdir("/var/azureml-app/avaj1870"))
    if not os.path.exists(inputs_root):
        os.mkdir(inputs_root)
    if not os.path.exists(outputs_root):
        os.mkdir(outputs_root)

    random_unique_id = str(uuid.uuid1())
    print("[INFO] : random unique id for request = ", random_unique_id)
    inputs_dir = os.path.join(inputs_root, random_unique_id)
    outputs_dir = os.path.join(outputs_root, random_unique_id)

    if request.method == 'POST':
        # save request data as a file here
        files = request.files

        os.mkdir(inputs_dir)
        os.mkdir(outputs_dir)
        
        # save files to inputs dir
        for f in files:
                # Copy the BytesIO stream to the output file
                files[f].save(os.path.join(inputs_dir, f))

        print("[INFO] Inputs dir content : ", os.listdir(inputs_dir))

        # make prediction  
        make_prediction(inputs_dir, outputs_dir)

        print("[INFO] Outputs dir content : ", os.listdir(outputs_dir))

        results_encoded = {}
        for fname in os.listdir(outputs_dir):
            with open(os.path.join(outputs_dir, fname), "rb") as f: 
                results_encoded[fname] = base64.b64encode(f.read()).decode("utf-8")

        # clean inputs and outputs directories
        shutil.rmtree(inputs_dir)
        #shutil.rmtree(outputs_dir)

        # return results
        print(results_encoded)
        return {'result_files': results_encoded}
    else:
        return AMLResponse("Bad request, use POST", 500)
