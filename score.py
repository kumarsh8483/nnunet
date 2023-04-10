from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

from io import BytesIO
import os

import nnunet

def init():
    global model
    # load the model here
    # this part will be determined w.r.t nnunet repo
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'latest_model.pth')
    model = joblib.load(model_path)
    # create input and output folder for inferece
    os.mkdir("outputs")
    os.mkdir("inputs")

# send files as a list in post request body
# json format data
@rawhttp
def run(request):
    print("run executed with raw data : ", request)
    if request.method == 'POST':
        # save request data as a file here
        reqBody = request.get_data(False)
        
        # save files to inputs dir
        for bytes in reqBody:
            # filename will be determined w.r.t to nnunet repo
            with open(filename, "wb") as outfile:
                # Copy the BytesIO stream to the output file
                outfile.write(BytesIO(bytes).getbuffer())

        # this part will be determined w.r.t nnunet repo prediction
        # make prediction  
        results = model.predict(imgprepped)

        return {'results':results}
    else:
        return AMLResponse("Bad request, use POST", 500)