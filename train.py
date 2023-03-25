# training python script : train.py
import os
import subprocess

nnUNet_raw_data_base=os.getenv('nnUNet_raw')
nnUNet_preprocessed=os.getenv('nnUNet_preprocessed')
RESULTS_FOLDER=os.getenv('nnUNet_results')

class Command():
    counter = 0

    def init(self, *cmd):
      self.cmd = cmd

    def run(self):
      result = subprocess.run(self.cmd, stdout=subprocess.PIPE)
      print(f"Command {self.counter} execution : ", end="")
      print(result.stdout.decode('ascii'))

def train():
  # parameters :
  dataset_id = "005"
  train_type = "2d"
  fold = "0"

  Command('nnUNetv2_train', dataset_id, train_type, fold).run()


def create_dataset_dirs():
  try :
    os.makedirs(os.path.join(nnUNet_raw_data_base, 'nnUNet_raw_data'))
    os.makedirs(nnUNet_preprocessed)
    os.makedirs(RESULTS_FOLDER)
  except FileExistsError:
    pass

if __name__ == "__main__":
  # prepare data
  create_dataset_dirs()

  Command('gdown', 'https://drive.google.com/uc?id=1vvgcavq_Za42T5YUVQZ2U6wgg4idq6Wy&confirm=t').run()
  Command('tar', '-xf', "Task05_Prostate.tar", "-C", f"{nnUNet_raw_data_base}/nnUNet_raw_data/").run()
  Command('nnUNetv2_convert_MSD_dataset', '-i', f'{nnUNet_raw_data_base}/nnUNet_raw_data/Task05_Prostate').run()
  Command('nnUNetv2_plan_and_preprocess', '-d', '5', '--verify_dataset_integrity').run()

  train()
