from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

from io import BytesIO
import os
import shutil
import uuid
from time import time
import shutil 
import base64

from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, default_cascade_trainer, default_trainer

inputs_root = "inputs"
outputs_root = "outputs"
task_name = "Task005_Prostate" #default_plans_identifier

def make_prediction(input_folder, output_folder):
    network_dir = "network_dir"
    input_folder = "inputs"
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
    # TODO: model will be loaded from here in the future
    # create input and output folder for inferece
    os.mkdir(inputs_dir)
    os.mkdir(outputs_dir)

# send files as a list in post request body
# json format data
@rawhttp
def run(request):
    print("run executed with raw data : ", request.get_data(False))
  
    if request.method == 'POST':
        # save request data as a file here
        reqBody = request.get_data(False)

        # create unique input/output dirs for request 
        # (uniqueness needed to prevent possible race conditons)
        random_unique_id = uuid.uuid1()
        inputs_dir = os.path.join(inputs_root, random_unique_id)
        outputs_dir = os.path.join(outputs_root, random_unique_id)

        os.mkdir(inputs_dir)
        os.mkdir(outputs_dir)
        
        # save files to inputs dir
        for f in reqBody:
            # filename will be determined according to nnunet repo
            with open(os.path.join(inputs_dir, "dummy.txt, ""wb")) as outfile:
                # Copy the BytesIO stream to the output file
                outfile.write(BytesIO(bytes).getbuffer())

        # make prediction  
        make_prediction(inputs_dir, outputs_dir)

        # encode output files to base64 format
        with open(os.path.join(outputs_dir, "plans.pkl"), "rb") as f:
            plans_encoded = base64.b64encode(f.read())

        results_encoded = []
        for fname in os.listdir(outputs_dir):
            if fname != "plans.pkl":
                with open(os.path.join(outputs_dir, fname), "rb") as f:
                    results_encoded.append(base64.b64encode(f.read()))

        # clean inputs and outputs directories
        shutil.rmtree(inputs_dir)
        shutil.rmtree(outputs_dir)

        # return results
        return {'plans': plans_encoded, 'results':results_encoded}
    else:
        return AMLResponse("Bad request, use POST", 500)