# Training pytorch models on Cloud TPU Pods with Cloud Storage

Disclaimer: this code is provided for example purposes only.

This repo contains supporting files for training with the following configuration:

## Cloud TPU Pod

TPU Pods are broken up into several slices. Each slice needs to be paired with a VM worker (asynchronous training). A v2-32 TPU Pod has 4 slices (32/8=4).
When training happens each vm will get a dedicated slice.

The XLA compiler performs code transformations, including tiling a matrix multiply into smaller blocks, to efficiently execute computations on the matrix unit (MXU).
The structure of the MXU hardware, a 128x128 systolic array, and the design of TPU's memory subsystem, which prefers dimensions that are multiples of 8, are used by the XLA compiler for tiling efficiency.

## Managed instance group

Every TPU will get a node from the instance group, so the instance group will have 4 nodes. The image is the deep learning pytorch 1.7 image.

## Dataset

~135GB of 2012 ImageNet data stored in cloud storage in a ImageFolder-compatible layout.

```
imagenet/train/nXXXXXX/file.jpg
imagenet/validate/nXXXXXX/file.jpg
```

> #### TODO: add more about dataprep here.


Getting the data in-place is out of scope for now. For a quick overview:

1. Get the 2012 training archive and unpack into files that look like n1231231.tar. Each of those neeeds to ne unpacked into it's own folder: n123123123/file.JPEG.
2. Get the 2012 validation archive and bounding boxes from ImageNet.
3. Use the bounding box xml file and to classify files from the validation archive and then put them into a folder structure like above.

# Running the model

Create a cloud project if necessary, or re-use an existing on. Either way identify the project id because you will use it in later steps.
From console: Create the initial required monitoring workspace. This will take a minute and is required for terraform dashboard creation to succeed. Ensure that you replace "your-project-id"
with the cloud project.

```
echo "http://console.cloud.google.com/monitoring?project=your-project-id"
```

From cloudshell:

```
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

echo project=\"$PROJECT_ID\" >> terraform.tfvars
echo 'zone="us-central1-a"'  >> terraform.tfvars # choose zone to deploy

export BUCKET=your-bucket # eg for gs://foo/bar you would set BUCKET=foo
echo BUCKET=\"$BUCKET\" >> terraform.tfvars

./make-package.sh # this will place a driver script in cloud storage

terraform init
terraform apply
```
*If you recieve an error stating "Cannot assign requested address...", execute the following commands in cloudshell and run `terraform apply` again:* until https://github.com/hashicorp/terraform-provider-google/issues/6782 is resolved.

```
echo "74.125.142.95 tpu.googleapis.com" | sudo tee -a /etc/hosts
echo "74.125.195.95 www.googleapis.com" | sudo tee -a /etc/hosts
```


Within ~1 minute, the following resources should be available (assuming default config):
* 4 training VM instances (e.g., `tpupod-XXXX`) with PyTorch XLA base image
* 1 profiler VM instance (e.g., `torch-profile-vm-XXX`) with Tensorflow base image and dependencies for the TPU profiler
* 1 Cloud TPU node (e.g., `mytpu`)

**SSH into any training VM, either through UI or using `gcloud` in cloudshell:**

```
gcloud beta compute ssh --zone=$ZONE "YOUR_VM_NAME" --project=$PROJECT_ID
```

https://console.cloud.google.com/compute/instances?project=$PROJECT_ID&duration=PT1H&instancessize=50

**In SSH terminal, set required and optional variables/parameters**

(optional) Change ``LOGDIR`` from default value:

```
export LOGDIR="${LOGDIR:-gs://YOUR_BUCKET/log-$(date '+%Y%m%d%H%M%S')}"
```

(optional) Set training parameters

```
export NUM_EPOCHS=10
```
(required) Set `BUCKET` and `IMAGE_DIR`:
```
export BUCKET=your-bucket
export IMAGE_DIR=your dir # example IMAGE_DIR=gs://bucket/imagenet
```
**Execute bash script to kick-off distributed training job:**

```
bash /tmp/thepackage/go.sh
```
*Note: the first log will specify the GCS logdir location. Save this for later steps (or navigate to your GCS buckets)*

This will use `torch_xla.distributed.xla_dist` to:

1. Identify all instances in your instance group.
2. Identify grpc endpoints for all slices of your TPU.
3. Kick off a distributed training job on all nodes.

## Gathering metrics

Terraform creates a dashboard named "Pytorch Training" which displays TPU and worker utilization, memory, and I/O bandwidth.
To view in the GCP console: **Monitoring -> Dashboards -> Pytorch Training**

Additionally, use the [Cloud TPU Profiler TensorBoard Plug-in](https://cloud.google.com/tpu/docs/cloud-tpu-tools) for a more detailed view of TPU performance

* Wait until you see output logs indicating your model is training:

```
Training Device=xla:0/X Epoch=1 Step=X Loss=X Rate=X GlobalRate=X Time=21:19:06
```

* SSH into the profiler VM (e.g., `torch-profile-vm-XXX`) 

```
gcloud beta compute ssh --zone=$ZONE "YOUR_VM_NAME" --project=$PROJECT_ID
```

* Specify your `TPU_NAME` and the `LOGDIR` used for storing training outputs

```
export STORAGE_BUCKET=gs://$YOUR_BUCKET
export LOGDIR=${STORAGE_BUCKET}/log-XXXXX # log directory created earlier
export TPU_NAME=mytpu
```
* Run the following command
```
capture_tpu_profile --tpu=${TPU_NAME} --logdir=${LOGDIR} --duration_ms=10000
```
By default, `capture_tpu_profile` captures a 2-second trace. You can set the trace duration with the `--duration_ms` command-line option

When you use the `capture_tpu_profile` script to capture a profile, several files are saved to the GCS bucket you specify. The **.trace** file contins up to one million trace events that can be viewed in the trace viewer. The **.traceable** file contains up to 2 GB of trace events that can be viewed in the streaming trace viewer

## Viewing profiling data

At this time, Tensorboard has trouble reading the folder structure created by `capture_tpu_profile` on PyTorch models. To overcome this, run the following code snippet in **cloudshell** to restructure the folder hierarchy: 
* Replace `gs://$BUCKET/log-XXXXX` with the logdir specified in the ``capture_tpu_profile`` command 
* Replace `PROFILER_DATE_TIME` with the folder created by the profiler (check the logdir in GCS for a folder similar to **2020_11_13_23_13_01**)


```
for line in $(gsutil ls -R gs://$BUCKET/log-XXXXX/PROFILER_DATE_TIME ); do echo "gsutil mv '$line' '$(echo $line |  sed -e 's|\(PROFILER_DATE_TIME\)\(.*\)|plugins/profile/\1\2|g')'" ; done > out

bash out
```
* Wait until the script is finished moving all files
* Open a new cloudshell command prompt and SSH into the profiler VM using this command:

```
gcloud beta compute ssh $PROFILER_VM_NAME \
  --zone=$ZONE \
  --ssh-flag=-L6006:localhost:6006 \
  --project=$PROJECT_ID
```
* Specify the `LOGDIR` and launch Tensorboard

```
export STORAGE_BUCKET=gs://$YOUR_BUCKET
export LOGDIR=${STORAGE_BUCKET}/log-XXXXX # log directory created earlier

tensorboard --logdir=${LOGDIR}
```

* Click on the printed link to view Tensorboard 
* If you do not see the "Profile" tab in the top ribbon, click the update/refresh button (top right of dashboard) 

If it still doesn't appear, check the hierarchy of your logdir. It should be similar to this:

    .
    ├── BUCKET/            
    │   ├── LOGDIR/
    │   │   ├── event files
    │   │   └── plugins/
    │   │   │   ├── profile/
    │   │   │   │   ├── PROFILER_DATE_TIME/
    │   │   │   │   │   ├── profiler files
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   └── ...    
    │   └── ...                
    └── ...


See [here](https://cloud.google.com/tpu/docs/cloud-tpu-tools#profile_tab) for more information on how to interpret the "Profile" tab

> #### **TODO:**
* Update go.sh to include code restructuring folder hierarchy produced by ``capture_tpu_profile``
* Update model code to incorporate other Tensorboard capabilities, e.g., scalars, image, embedding projector, graph, etc.
