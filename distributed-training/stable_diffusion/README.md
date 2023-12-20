# Intel® Cloud Optimization Modules for AWS: Stable Diffusion Distributed Training
The Intel Cloud Optimization Modules (ICOMs) are open-source codebases with codified Intel AI software  optimizations and instructions built specifically for each Cloud Service Provider (CSP).  The ICOMs are built with production AI developers in mind, leveraging popular AI frameworks within the context of cloud services.

## Introduction

A Stable Diffusion Generative Text to Image Model leverages a diffusion process to convert textual descriptions into coherent image representations, establishing a robust foundation for multimodal learning tasks. Fine-tuning this model enables users to tailor its generation capabilities towards specific domains or datasets, thereby improving the quality and relevance of the produced imagery. 

As the fine-tuning process can be computationally intensive, especially with burgeoning datasets, distributing the training across multiple Intel 4th Gen CPUs equipped with Advanced Matrix Extension (AMX) and BF16 mixed precision training support through Intel's extension for PyTorch can significantly accelerate the fine-tuning task.

This Intel Cloud Optimization Module is focused on providing instructions for executing this fine-tuning workload on the AWS cloud using Intel 4th Generation Processors (Codednamed: Sapphire Rapids) and software optimizations offered through Hugging Face's Accelerate library. 

## 1. AWS Prerequisites
Before proceeding, ensure you have an AWS account and the necessary permissions to launch EC2 instances, create Amazon Machine Images (AMIs), create security groups, and create S3 storage buckets.

We used three [*m7i.4xlarge* EC2 instances](https://aws.amazon.com/ec2/instance-types/m6i/) with Ubuntu 22.04 and 250 GB of storage each.

In order to get started, you must first launch an EC2 instance and open it up in a command prompt. You can do so from the AWS console with the instructions that are found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-instance-wizard.html).

If you are using a 4th Geneneration Xeon CPU, you can verify that you have the AMX instruction set by running:

```bash
lscpu | grep amx
```

and you should see the following flags:

```
amx_bf16 amx_tile amx_int8
```

These flags indicate that the AMX instructions are available on your system, which are essential for leveraging mixed precision training and using `bfloat16`. Please keep in mind that, for now, the AMX instruction set is only supported by 4th Gen. Xeon CPUs.

## 2. Clone the Repo and Install Dependencies

Install make 
```bash
sudo apt install make
```

Clone the repository 

```bash
git clone https://github.com/intel/intel-cloud-optimizations-aws.git
``` 

navigate to the setup folder 

```bash
cd intel-cloud-optimizations-aws/distributed-training/stable_diffusion/
```

install miniconda by running

```bash
make install-miniconda
```

source bash profile to activate base conda env

```bash
source ~/.bashrc
```

and run make instructions below. We've created this instruction to make it as easy as possible to setup the environment in the master node. 

```bash
make setup-stable-diffusion-icom-master
```

This instruction will setup all of the dependencies in the EC2 Instance. These include:
- setting up a conda environment
- installing pytorch, intel extension for pytorch, oneccl bindings, transformers, accelerate, and the diffusers library 

Now we want to navigate to the diffusers folder and make a few changes to the textual_inversion.py script

it is located in: examples/textual_inversion/textual_inversion.py

We recommend using a text editor for this part, if you want to practice making the changes yourself, However we have already done this for you and you can just use the textual_inversion_icom.py file that is part of the ICOM. 

If doing it by hand you will add:

```python    
import intel_extension_for_pytorch as ipex
unet = ipex.optimize(unet, dtype=weight_dtype)
vae = ipex.optimize(vae, dtype=weight_dtype)
```

after: 

```python
unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=weight_dtype)
```

to test on a single node we can run the single-node-test make target, again make sure you are in the **setup/** directory when running this:

```bash
make single-node-test
```

The single node only runs 5 steps and should save the learned_embeds.safetensors in the textual_inversion_output folder when fine-tuning is completed. 

Once we successfully test on a single node, we can begin setting up the rest of the infrastructure for the distributed training. 

## 3. Setting up Amazon Machine Image (AMI) for Distributed Training

Before creating the AMI we want to run the following in the master node: 

If not already installed, install net-tools

```bash
sudo apt install net-tools
```

Get correct hydra Iface by running `ifconfig -a | grep enp` - this will give us the correct value to set for the **I_MPI_HYDRA_IFACE** environment variable.

Run the following to establish the required environment variables. 

```bash
export I_MPI_HYDRA_IFACE=<replace with value from previous step>
oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export CCL_ATL_TRANSPORT=ofi
export CCL_WORKER_COUNT=1

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="/home/ubuntu/stable_diffusion_aws_xeon/dicoo"
```

Also delete the textual_inversion_output folder, created during our test run, by running: 

```bash
rm -rf textual_inversion_output
```

**Adding SSH Access Key to Instance**: There a few different ways to accomplish this. If the following approach does not accomodate your workspace, feel free to use another method. (This is part of setting up passwordless ssh)
1. Locate your AWS EC2 instance key. This is the key that you specified when launching the instance. It should have a ".pem" extension
2. Open the file in a text editor
3. Copy the output of the command, which is your key.
4. SSH into the remote host where you want to add your key.
5. Once logged in, navigate to the ~/.ssh directory on the remote host if it exists. If the directory does not exist, you can create it:
```mkdir -p ~/.ssh```
6. Within the ~/.ssh directory on the remote host, create a file named id_rsa:
```nano ~/.ssh/id_rsa```
7. Paste the previously copied public key into this file, making sure it's on a new line.
8. Save the changes and exit the text editor.
9. Adjust permissions to file `chmod 600 ~/.ssh/id_rsa`


**Create an AMI**: Start by creating an Amazon Machine Image (AMI) from the existing instance where you have successfully run the fine-tuning on a single system. This AMI will capture the entire setup, including the dependencies, configurations, codebase, and dataset. To create an AMI, refer to [Create a Linux AMI from an instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/creating-an-ami-ebs.html#:~:text=EBS%20volumes.-,Create%20a%20Linux%20AMI%20from%20an%20instance,-You%20can%20create).


**Security Group**: While waiting for the AMI creation, let's continue by creating a security group that enables communication among the member nodes. This security group should be configured to allow inbound and outbound traffic on the necessary ports for effective communication between the master node and the worker nodes. The easiest place to do this is from the AWS console. (This is part of setting up passwordless ssh)

In the security group configuration, ensure that you have allowed *all* traffic originating from the security group itself. This setting allows seamless communication between the instances within the security group.

By setting up the security group in this manner, you ensure that all necessary traffic can flow between the master node and the worker nodes during distributed training.

![image](https://github.com/intel/intel-cloud-optimizations-aws/assets/57263404/f33d5d93-1822-4d4a-9bdb-8b2d67c224c2)


## 4. Setting up Workers Instances

If you don't already have the awscli installed, follow in the instructions on this site to install it: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

**Launch new instances**: Use the created AMI to launch new instances, specifying the desired number of instances based on the number of systems you want to use for distributed training. This ensures that all the instances have the same environment and setup. To initiate new EC2 instances, there are two options available: using the AWS console, or AWS CLI. If you have AWS CLI configured, you can launch instances by executing the following command:

    ```bash
    aws ec2 run-instances --image-id ami-xxxxxxxxx --count 2 --instance-type m7i.4xlarge --key-name <name of preferred ec2 key> --security-group-ids sg-xxxxxxxxx --subnet-id subnet-xxxxxxxxx
    ```
    
After completing the above the public key to all nodes, verify that you can connect using the key pair. Run the following from the node you have been working from (main node).

```ssh <username>@<private-ip-address>```

If you can successfully log in without entering a password, it means passwordless SSH is set up correctly.

Passwordless SSH between the master node and all worker nodes ensures smooth communication and coordination during distributed training. If you encounter any difficulties, additional information can be found here.

Next, to continue setting up the cluster, you will need to edit the SSH configuration file located at `~/.ssh/config` on the master node (if it doesn't exit you can create one). The configuration file should look like this:

In the host section you will put the Private IPv4 addresses. You can get these for each node from the AWS console OR by running `hostname -i` when SSH'd into the corresponding node.

```plaintext
Host 10.*.*.*
   StrictHostKeyChecking no

Host node1
    HostName 10.0.xxx.xxx
    User ubuntu

Host node2
    HostName 10.0.xxx.xxx
    User ubuntu
```

The `StrictHostKeyChecking no` line disables strict host key checking, allowing the master node to SSH into the worker nodes without prompting for verification.

With these settings, you can check your passwordless SSH by executing `ssh node1` or `ssh node2` to connect to any node without any additional prompts.

Additionally, on the master node, you will create a host file (`~/hosts`) that includes the names of all the nodes you want to include in the training process, as defined in the SSH configuration above. Use `localhost` for the master node itself as you will launch the training script from the master node. You can stick this file in your working directory. The `hosts` file will look like this:

```plaintext
localhost
node1
node2
```

This setup will allow you to seamlessly connect to any node in the cluster for distributed training.

## 4. Preparing for Distributed Training

Now that we have this running in a single system, let's try to run it on multiple systems. To prepare for distributed training and ensure a consistent setup across all systems, follow these steps:

1. **Configure Accelerate**: Activate your conda environment. If you did not change its name in the make file, it should be called "diffuser_icom" e.g. `conda activate diffuser_icom`

We need to prepare a new "accelerate" config for multi-CPU setup. But before setting up the multi-CPU environment, ensure you have the IP address of your machine handy. To obtain it, run the following command:

```bash
hostname -i
```

With the IP address ready, execute the following command to generate the new accelerate config for the multi-CPU setup:
```bash
accelerate config
```

When configuring the multi-CPU setup using `accelerate config`, you will be prompted with several questions. To select the appropriate answers based on your environment. Here's a step-by-step guide on how to proceed:

First, select `This machine` as we are not using Amazon SageMaker. 

```bash
In which compute environment are you running?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  This machine
    AWS (Amazon SageMaker)
```

Choose `multi-CPU` as the type of machine for our setup.

```bash
Which type of machine are you using?
Please select a choice using the arrow or number keys, and selecting with enter
    No distributed training       
 ➔  multi-CPU
    multi-XPU
    multi-GPU
    multi-NPU
    TPU
```

Next, you can enter the number of instances you will be using. For example, here we have 3 (including the master node). 

```bash
How many different machines will you use (use more than 1 for multi-node training)? [1]: 3
```

Concerning the rank, since we are initially running this from the master node, enter `0`. For each machine, you will need to change the rank accordingly.

```bash
What is the rank of this machine?
Please select a choice using the arrow or number keys, and selecting with enter
 ➔  0
    1
    2
```

Next, you will need to provide the private IP address of the machine where you are running the `accelerate launch` command (main node - likely the same node you have been working on this whole time), that we found earlier with `hostname -i`.

```bash
What is the IP address of the machine that will host the main process?   
```

Next, you can enter the port number to be used to communication. Commonly used port is 29500, but you can choose any available port. 

```bash
What is the port you will use to communicate with the main process?   
```

You will be prompted with a few more questions. Provide the required information as per your setup.

The prompt of
```bash
How many CPU(s) should be used for distributed training?
```
is actually about CPU sockets. Generally, each machine will have only 1 CPU socket. However, in the case of bare metal instances, you may have 2 CPU sockets per instance. Enter the appropriate number of sockets based on your instance configuration.

After completing the configuration, you will be ready to launch the multi-CPU fine-tuning process. The final output should look something like:

```bash
------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-CPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 3
------------------------------------------------------------------------------------------------------------------------------------------
What is the rank of this machine?
0
What is the IP address of the machine that will host the main process? xxx.xxx.xxx.xxx
What is the port you will use to communicate with the main process? 29500
Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]: nossh no
What rendezvous backend will you use? ('static', 'c10d', ...): static
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU? [yes/NO]:yes
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
How many CPU(s) should be used for distributed training? [1]:1
------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
bf16
```

You now should have generated a new config file named `multi_config.yaml` in the .cache folder. You now have 2 options:
Option 1: Repeat the accelerate configure process on each node by sshing into the worker nodes and running `accelerate config`
Option 2: Copy the contents of the hugging face accelerate cache to the other nodes. You would need to change the machine rank in the yaml file to correspond to the rank of each machine in the distributed system.

## 5. Fine-Tuning on Multiple CPUs

Finally, it's time to run the fine-tuning process on multi-CPU setup. Make sure you are connected to your main machine (rank 0) and in the "./stable_diffusion/" directory. Run the following command be used to launch distributed training:
```bash
mpirun -f ./hosts -n 3 -ppn 1 accelerate launch textual_inversion_icom.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --train_data_dir="./dicoo/" --learnable_property="object"   --placeholder_token="<dicoo>" --initializer_token="toy" --resolution=512  --train_batch_size=1  --seed=7  --gradient_accumulation_steps=1 --max_train_steps=30 --learning_rate=2.0e-03 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=3 --output_dir=./textual_inversion_output --mixed_precision bf16 --save_as_full_pipeline
```

Some notes on the arguments for `mpirun` to consider:
- `-n`: This parameter represents the number of CPUs or nodes. In our case, we specified `-n 3` to run on 3 nodes. Typically, it is set to the number of nodes you are using. However, in the case of bare metal instances with 2 CPU sockets per board, you would use `2n` to account for the 2 sockets.
- `-ppn`: The "process per node" parameter determines how many training jobs you want to start on each node. We only want 1 instance of each training to be run on each node, so we set this to `-ppn 1`. 
- `--pretrained_model_name_or_path`: Path to pretrained model or model identifier from huggingface.co/models.
- `--learnable_property`: Choose between 'object' and 'style'
- `--placeholder_token`: A token to use as a placeholder for the concept.
- `--initializer_token`: A token to use as initializer word.
- `--resolution`: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
- `--train_batch_size`: Batch size (per device) for the training dataloader.
- `--seed`: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
- `--gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.
- `--max_train_steps`: Total number of training steps to perform.  If provided, overrides num_train_epochs.
- `--learning_rate`: Initial learning rate (after the potential warmup period) to use.
- `--lr_scheduler`: The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", constant" "constant_with_warmup"]
- `--lr_warmup_steps`: Number of steps for the warmup in the lr scheduler.
- `--output_dir`: The output directory where the model predictions and checkpoints will be written.
- `--mixed_precision`: Whether to use mixed precision.
- `--save_as_full_pipeline`: Save the complete stable diffusion pipeline.

## 6. Comments on Distributed Training and Benefits

- Faster Training: As demonstrated in the output, distributed training significantly reduces the training time for large datasets. It allows parallel processing across multiple nodes, which accelerates the training process and enables efficient utilization of computing resources.

- Scalability: With distributed training, the model training process can easily scale to handle massive datasets, complex architectures, and larger batch sizes. This scalability is crucial for handling real-world, high-dimensional data.

- Model Generalization: Distributed training enables access to diverse data samples from different nodes, leading to improved model generalization. This, in turn, enhances the model's ability to perform well on unseen data.

Overall, distributed training is an indispensable technique that empowers data scientists, researchers, and organizations to efficiently tackle complex machine learning tasks and achieve superior results.


## 7. Cleaning up AWS Resources

Ensure that you properly remove and clean up all the resources created during the course of following this module. To delete EC2 instances and a security group using the AWS CLI, you can use the following commands:

1. Delete EC2 instances:
```bash
aws ec2 terminate-instances --instance-ids <instance_id1> <instance_id2> ... <instance_idN>
```
Replace `<instance_id1>`, `<instance_id2>`, ..., `<instance_idN>` with the actual instance IDs you want to terminate. You can specify multiple instance IDs separated by spaces.

2. Delete Security Group:
```bash
aws ec2 delete-security-group --group-id <security_group_id>
```
Replace `<security_group_id>` with the ID of the security group you want to delete.

Please be cautious when using these commands, as they will permanently delete the specified EC2 instances and security group. Double-check the instance IDs and security group ID to avoid accidental deletions.

## 8. Follow Up

1. [Register for Office Hours here](https://software.seek.intel.com/SupportFromIntelExperts-Reg) for help on your ICOM implementation.

2. Learn more about all of our [Intel Cloud Optimization Modules here](https://www.intel.com/content/www/us/en/developer/topic-technology/cloud-optimization.html).

3. Come chat with us on our [Intel DevHub Discord](https://discord.gg/rv2Gp55UJQ) server to keep interacting with fellow developers.

4. Stay connected with us on social media:

-  Eduardo Alvarez | Senior AI Solutions Engineer | [LinkedIn](https://www.linkedin.com/in/eduandalv/)

