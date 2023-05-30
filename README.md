# nnunet-azureml-training-pipeline

### Steps to test on local with current dockerfiles :

1. First, we need to docker image. So, run below command first time you use any dockefile. And, whenever you change the dockerfile:
    
        docker build -f dockerfile -t myimagename .

    This command builds docker environment image using the file 'dockerfile' & name builded image as myimagename:latest, by looking into files at current directory (. on the end makes docker to use current directory as build context).

2. Then, find the image name myimagename in the images tab of docker desktop.
   
   ![Alt text](assets/run-local-cpu.png?raw=true "Title")

3. Go to additional settings and enter your train.py file path, in host path, container path is the same.

    ![Alt text](assets/volume-mounting-local-cpu.png?raw=true "Title")

4. Then, click run and it will direct you to containers tab, which will print running container output.


Notes :

* Whenever you change dockerfile, run step1 command.
  
* After building once in step 1, you should start with step 2. Because train.py is moutned volume. Changes to train.py will be reflected when you rerun the image. No need to rebuild.
   
        
    
