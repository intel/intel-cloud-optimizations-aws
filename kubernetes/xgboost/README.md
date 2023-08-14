# Intel® Cloud Optimization Modules for AWS*: XGBoost* on Kubernetes*

This module can be used to build and deploy AI applications on the AWS cloud. Specifically, we will focus on one of the first Intel Cloud Optimization Modules, which serves as a template with codified Intel accelerations covering various AI workloads. We will also introduce the AWS services that we will use in the process, including Amazon Elastic Kubernetes Service (EKS), Amazon Elastic Container Registry (ECR), Amazon Elastic Compute Cloud (EC2), and Elastic Load Balancer (ELB).


## Solution Architecture 

The architecture uses Docker for application containerization and Elastic Container (ECR) Storage on AWS. The application image is then deployed on a cluster managed by Elastic Kubernetes Service (EKS). Our clusters are made up of EC2 instances. We use S3 for storing data and model objects, which are retrieved during various steps of our ML pipeline. The client interacts with our infrastructure through our Elastic Load Balancer, which gets provisioned by our Kubernetes service.

![Solution_Architecture_Diagram](https://user-images.githubusercontent.com/57263404/226037559-6eb6c83b-be59-4290-86ce-96e41356f174.png)
  
## Preparing your Environment

Install the AWS CLI — The AWS CLI (Command Line Interface) tool is a command-line tool for managing various Amazon Web Services (AWS) resources and services.
  
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install
```
  
Install eksctl — eksctl is a command-line tool for creating, managing, and operating Kubernetes clusters on EKS.
  
 ```
 curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
eksctl version
 ```
  
Install aws-iam-configurator — AWS IAM Authenticator is a command-line tool that enables users to authenticate with their Kubernetes clusters on EKS using their AWS IAM credentials.
  
```
curl -Lo aws-iam-authenticator https://github.com/kubernetes-sigs/aws-iam-authenticator/releases/download/v0.5.9/aws-iam-authenticator_0.5.9_linux_amd64
chmod +x ./aws-iam-authenticator
mkdir -p $HOME/bin && cp ./aws-iam-authenticator $HOME/bin/aws-iam-authenticator && export PATH=$PATH:$HOME/bin
echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc
aws-iam-authenticator help
```
  
Install kubectl — Kubectl is a command-line tool for interacting with Kubernetes clusters. It allows users to deploy, inspect, and manage applications and services running on a Kubernetes cluster and perform various administrative tasks such as scaling, updating, and deleting resources.

```
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```
  
## Our Loan Default Prediction Application
  
The application we will be deploying is based on the Loan Default Risk Prediction AI Reference Kit.

We refactored the code from this reference solution to be more modular in support of our three main APIs:

- Data processing — This endpoint preprocess data and stores it in a data lake or another structured format. This codebase also handles the expansion of the dataset for benchmarking purposes.
- Model Training — This endpoint trains an XGBoost Classifier and converts it to an inference-optimized daal4py format.
- Inference — This endpoint receives a payload with raw data and returns the loan default classification of each sample.
  
The directory tree below outlines the codebase’s various scripts, assets, and configuration files. The majority of the ML application code is in the app/ folder. This folder contains loan_default and utils packages — the loan_default package contains the server-side python modules that support our three main APIs. The server.py script contains the FastAPI endpoint configurations, payload data models, and commands to start a uvicorn server.
  
```
├───app/
|   ├───loan_default/
|   |   ├───__init__.py
|   |   ├───data.py
|   |   ├───model.py
|   |   └───predict.py
|   ├───utils/
|   |   ├───__init__.py
|   |   ├───base_model.py
|   |   ├───logger.py
|   |   └───storage.py  
|   ├───logs/
|   ├───server.py
|   └───requirements.txt    
├───assets/
|   └───cheatsheet.png
├───kubernetes/
|   ├───cluster.yaml
|   ├───deployment.yaml
|   ├───service.yaml
|   └───serviceaccount.yaml
|
├─README.md
├─Dockerfile
├─SECURITY.md
```
  
## Configuring and Launching Elastic Kubernetes Service Clusters

Elastic Kubernetes Service is a fully managed service that makes it easy to deploy, manage, and scale containerized applications using Kubernetes on Amazon Web Services (AWS). It eliminates the need to install, operate, and scale Kubernetes clusters on your own infrastructure.

To launch our EKS cluster, we must first create our cluster configuration file — cluster.yaml
  
```
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: "eks-cluster-loanDefault"
  version: "1.23"
  region: "us-east-1"

managedNodeGroups:
- name: "eks-cluster-loanDefault-mng"
  desiredCapacity: 2
  minSize: 2
  maxSize: 6
  volumeSize: 100
  instanceType: "m6i.xlarge"
```

- We can configure the name and region of our cluster deployment, as well as the version of EKS that we want to run, in our “metadata” section. Most importantly, we can configure basic requirements for we compute resources in the “managedNodeGroups” section:

- desiredCapacity - the number of nodes to scale to when your stack is created. In this tutorial, we will set this to 3.
- minSize - the mininum number of nodes at any given time.
- maxSize - the maximum number od nodes at any given time. 
- volumeSize - the size of the storage volume that is provisioned per node. 
- instanceType - the instance type for your nodes. This tutorial uses an m5.large instance, a 3rd Generation Xeon (2vCPU and 8GiB). Once openly available, we recommend trying out the r7iz instance family to take advantage of the Intel Advanced Matrix Extension (AMX) — a dedicated accelerator for deep learning workloads inside of Intel 4th Generation Xeon CPUs.

We execute `eksctl create cluster -f cluster.yaml` to create the Cloud Formation stack and provision all relevant resources. With the current configurations, this process should take 10 to 15 minutes.

You should run a quick test to ensure your cluster has been provisioned properly. Run `eksctl get cluster` to get the name of your available cluster(s), and `eksclt get nodegroup --cluster <cluster name>` to check on your cluster’s node group.
  

## Setting up all of the Kubernetes Application Resources

Let’s dig into launching your Kubernetes application. This process entails creating a namespace, a deployment manifest, and a Kubernetes service. All of these files are available in the tutorial’s codebase.

### Before moving on to this part of the tutorial, please:
- [Create a docker image using the Dockerfile in the application codebase and push it to the Elastic Container Registry on AWS](https://eduand-alvarez.medium.com/creating-an-ecr-registry-and-pushing-a-docker-image-93e372e74ff7)
- [Create and configure your kubernetes service account to grant your application proper access to S3 resources](https://eduand-alvarez.medium.com/how-to-assign-aws-service-permissions-to-kubernetes-resources-cb1e0257ca22)

### Once you've completed the tasks above, you can continue with the tutorial. 
  
A Kubernetes namespace is a virtual cluster that divides and isolates resources within a physical cluster. Let’s create a namespace called “loan-default-app” 

```
kubectl create namespace loan-default-app
```

### Kubernetes Deployment

Now, let’s configure our Kubernetes deployment manifest. A Kubernetes deployment is a Kubernetes resource that allows you to declaratively manage a set of replica pods for a given application, ensuring that the desired number of replicas are running and available at all times while enabling features such as scaling, rolling updates, and rollbacks. It also provides an abstraction layer over the pods, allowing you to define your application’s desired state without worrying about the underlying infrastructure.
  
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: "eks-loan-default-app"
  namespace: "loan-default-app"
  labels:
    app: "loan-default"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: "loan-default"
  template:
    metadata:
      labels:
        app: "loan-default"
    spec:
     serviceAccountName: "loan-default-service-account"
     topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: "loan-default"
     containers:
       - name: "model-image"
         image: # add your image URI here
         ports:
           - containerPort: 80
         imagePullPolicy: "Always"
         resources:
          limits:
            cpu: 500m
          requests:
           cpu: 250m
```

The Kubernetes deployment manifest (deployment.yaml) above defines the following:

- kind: Deployment — The type of Kubernetes resource
- name: “eks-loan-default-app” — The name of our deployment
- namespace: “loan-default-app” — The namespace that this deployment should be assigned to
- app: “loan-default” — The name we assign our application
- replicas: 3 — the number of desired copies of a pod that should be created and maintained at all times.
- serviceAccountName: “loan-default-service-account” — make sure this matches the service account you created earlier.
- topologySpreadConstraints: — helps define how pods should be distributed across your cluster. The current configuration will maintain an equal distribution of pods - across available nodes.
- containers: name/image — where you provide the URI for your application container image and assign the image a name.
- resources: establish the limits for CPU utilization

Run `kubectl apply -f deployment.yaml` to create your Kubernetes deployment.


### Kubernetes Autoscaler

Now we will configure the pod autoscaler kubernetes service. A Kubernetes pod autoscaler is a feature that automatically adjusts the number of running pods based on changes in resource usage and workload demand. This allows Kubernetes to scale up or down resources as needed, which helps optimize performance and minimize costs. Autoscaling is especially helpful for machine learning applications, where resource needs can vary greatly depending on the data being processed or the training model being used. By dynamically allocating resources as needed, autoscaling can ensure that machine learning applications can run efficiently and effectively, without wasting resources or incurring unnecessary costs.

```
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: loan-app-pod-autoscaler
  namespace: loan-default-app
  labels:
    app: "loan-default"
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: "eks-loan-default-app"
  maxReplicas: 6
  minReplicas: 1
  targetCPUUtilizationPercentage: 50 # 50% of CPU utilization
```

The Kubernetes Horizontal Pod Autoscaler manifest (pod-autoscaler.yaml) above defines the following:

- kind: HorizontalPodAutoscaler — The type of Kubernetes resource
- name: “loan-app-pod-autoscaler” — The name of our autoscaler
- namespace: “loan-default-app” — The namespace that this deployment should be assigned to
- app: “loan-default” — The name we assign our application
- maxReplicas: 6 - max number of replica pods to deploy
- minReplicas: 1 - min number of replica pods to deploy
- targetCPUUtilizationPercentage: 50% - threshold of CPU utilization for launching new pods

Run `kubectl apply -f pod-autoscaler.yaml` to create your Pod Autoscaling service. 

### Kubernetes Service
  
Now let’s configure our Kubernetes service. A Kubernetes service is an abstraction layer that provides a stable IP address and DNS name for a set of pods running the same application, enabling clients to access the application without needing to know the specific IP addresses of individual pods. It also provides a way to load-balance traffic between multiple replicas of the application and can be used to define ingress rules for external access.

```
apiVersion: v1
kind: Service
metadata:
  name: "loan-default-service"
  namespace: "loan-default-app"

spec:
  ports:
  - port: 8080
    targetPort: 5000
  selector:
    app: "loan-default"
  type: "LoadBalancer"
```

The Kubernetes service manifest (service.yaml) above defines the following:

- kind: Service — the type of Kubernetes resource.
- name: “loan-default-service” — The name of our deployment.
- namespace: “loan-default-app” — The namespace that this Service should be assigned to.
- port: 8080 — The port where the service will listen to.
- targetPort: 5000 — The port the service will communicate with on the pods.
- app: “loan-default” — The name we assigned to our application
- type: “LoadBalancer” — The type of service we selected.

Run `kubectl apply -f service.yaml` to create your Kubernetes service.

This will automatically launch an Elastic Load Balancer — a cloud service that distributes incoming network traffic across multiple targets, such as EC2 instances, containers, and IP addresses, to improve application availability and fault tolerance. We can use the ELB’s public DNS to make requests to our API endpoints from anywhere in the world.
  
## Here are a few tips before moving on:

- Run `kubectl get all -n loan-default-app` to get a full overview of the Kubernetes resources you have provisioned. You should see your pods, services, and replica groups.
- Run `kubectl -n loan-default-app describe pod <pod-id>` to get a detailed description of your pod.
- If you need to diagnose a specific pod’s behavior, you can start a bash shell inside your pod by running `kubectl exec -it <pod-id> -n loan-default-app -- bash` — type exit and hit enter to exit the shell.

## Testing our Loan Default Prediction Kubernetes Application

Now that all of our infrastructure is in place, we can set up the data component of our application and test our endpoints.

We will begin by downloading our dataset from Kaggle (https://www.kaggle.com/datasets/laotse/credit-risk-dataset). The dataset used for this demo is a set of 32581 simulated loans. It has 11 features, including customer and loan characteristics and one label, which is the outcome of the loan. Once we have the .csv file in our working directory, we can create an S3 bucket and upload are Kaggle dataset.

```
# create S3 Bucket for data
aws s3api create-bucket --bucket loan-default --region us-east-1

# upload dataset
aws s3api put-object --bucket loan-default --key data/credit_risk_dataset.csv --body <local path to data
```

### Making HTTP Requests to our API Endpoints
                                                                                                  
We will be using Curl to make HTTP requests to our server. Curl allows you to send HTTP requests by providing a command-line interface where you can specify the URL, request method, headers, and data. It then handles the low-level details of establishing a connection, sending the request, and receiving the response, making it easy to automate HTTP interactions.

We will start by sending a request to our data processing endpoint. This will create test/train files and save our preprocessing pipeline as a .sav file to S3. The body of the requests requires the following parameters:

- bucket: name of S3 bucket
- key: path where your raw data is saved in S3
- size: total samples you want to process
- backend: options include “local” or “s3” — the codebase supports running the entire app locally for debugging purposes. When using the “s3” backend, the “local_path” and “target_path” parameters can be set to “None”.
                                                                                                  
```
curl -X POST <loadbalancerdns>:8080/data -H 'Content-Type: application/json' -d '{"bucket":"loan-default","backend":"s3","key":"data/credit_risk_dataset.csv","target_path":"None","local_path":"None","size":400000}'
```
                                                                                                  
Now we are ready to train our XGBoost Classifier model. We will make a request to our /train endpoint, which trains our model, converts it to daal4py format, and saves it to S3. The body of the requests requires the following parameters:

- bucket: name of S3 bucket
- data_key: folder path that contains processed data created by our data processing API
- model_key: folder where we want to store our trained model
- model_name: the name that we want to give our trained model
- backend: options include “local” or “s3” — the codebase supports running the entire app locally for debugging purposes. When using the “s3” backend, the “local_model_path” and “local_data_path” parameters can be set to “None”     
                                                                                             
```
curl -X POST <loadbalancerdns>:8080/train -H 'Content-Type: application/json' -d '{"bucket":"loan-default","backend":"s3","local_model_path":"None","data_key":"data","model_key":"model","model_name":"model.joblib","local_data_path":"None"}'
```

Now that we have a trained daal4py optimized XGBoost Classifier, we can make inference requests to our API. The /predict endpoint will return a binary classification of True for high default likelihood and False for low default likelihood. The response also includes the probability generated by the classifier. In the codebase, we have set anything above a 50% probability to be labeled as a high default likelihood. This can be adjusted to return more discretized labels like low, medium, and high default likelihood. The body of the requests requires the following parameters:
                                                                                                 
- bucket: name of S3 bucket
- model_name: the name of the trained model is S3
- data_key: folder path that contains .sav processing pipeline file (should be the same as your processed data folder)
- model_key: folder where your trained model was saved in S3
- sample: your model inputs as a list of dictionaries
- backend: options include “local” or “s3” — the codebase supports running the entire app locally for debugging purposes. When using the “s3” backend, the “local_model_path” and “preprocessor_path” parameters can be set to “None”.

```
curl -X POST <loadbalancerdns>:8080/predict -H 'Content-Type: application/json' -d '{"backend":"s3","model_name":"model.joblib","data_key":"data","bucket":"loan-default","model_key":"model","sample":[{"person_age":22,"person_income":59000,"person_home_ownership":"RENT","person_emp_length":123,"loan_intent":"PERSONAL","loan_grade":"D","loan_amnt":35000,"loan_int_rate":16.02,"loan_percent_income":0.59,"cb_person_default_on_file":"Y","cb_person_cred_hist_length":3},{"person_age":22,"person_income":59000,"person_home_ownership":"RENT","person_emp_length":123,"loan_intent":"PERSONAL","loan_grade":"D","loan_amnt":35000,"loan_int_rate":55.02,"loan_percent_income":0.59,"cb_person_default_on_file":"Y","cb_person_cred_hist_length":0}],"local_model_path":"None","preprocessor_path":"None"}'
```

## Tracking Node CPU Utilization
If you are interested in tracking pod behavior associated with the HorizontalPodAutoscaler. You will need to follow the following steps to provision a Metrics tracking service. AWS EKS does not activate this service by default. 

1. Deploy the metrics server by running
```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

2. Verify that it is running 
```
kubectl get deployment metrics-server -n kube-system
```

3. Watch node utilization log to track CPU utilization and number of replicas. 
```
kubectl get hpa loan-app-pod-autoscaler -n loan-default-app --watch
```

<p align="center">
  <img src="https://github.com/intel/kubernetes-intel-aws-high-availability-training/blob/main/images/hpa-log.png" alt="HPA LOG" width="1000"/>
</p>


## Summary and Discussion
In this tutorial, we have demonstrated how to build a Kubernetes application on the AWS cloud based on a high-availability solution architecture. We have highlighted the use of Intel Xeon processors and AI Kit components to improve performance while enabling scale with Kubernetes.

We encourage readers to watch for upcoming workshops and future Intel Cloud Optimization Modules (ICOMs), as leveraging the Intel optimizations in these modules can qualify their applications for an “Accelerated by Intel” badge.

Our goal with ICOMs is to help developers enhance the performance and scalability of their applications with intel software and hardware. With the increasing demand for high-performance cloud applications, it is crucial for developers to stay informed and utilize the latest technologies and tools available to them.

*Most of the instructions above were sourced from this Optimization Module's associated Medium article (linked Below)*

### Associated Medium Article
- [How to Build Distributed ML Applications on the AWS Cloud with Kubernetes and oneAPI](https://medium.com/towards-data-science/how-to-build-distributed-ml-applications-on-the-aws-cloud-with-kubernetes-and-oneapi-81535012d136)
