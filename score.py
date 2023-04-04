def init():
    global model
    # load the model here
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'latest_model.pth')
    model = joblib.load(model_path)

def run(raw_data):
    # convert raw data base64 string into file format as expected from nnunet
    # run prediction on converted input file format
    # return the predictions as json