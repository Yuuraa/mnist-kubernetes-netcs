# Scale ML job with Kubernetes on on-premise kubernetes cluster

Original Work: 
https://github.com/hongkunyoo/how-to-scale-your-ml-job-with-k8s.git

## How to scale your ML job with Kubernetes

## 워크샵 순서
2. Provisioning K8S (핸즈온)
3. [How to scale your ML job (핸즈온)](#3-how-to-scale-your-ml-job-with-k8s)
    - Run a Basic Job
    - Save a model file to model storage
    - Exception handling
    - Training with hyper-parameters
    - Run multiple jobs
    - Using GPUs
    - Hello workflow
    - DAG workflow
    - Building ML Pipeline
    - Launch Jupyter notebook
    - Kubeflow tutorials



## 2. Provisioning K8S

Production 환경에서 제대로 클러스터를 구축한다면 private k8s 구축 및 도메인 네임 설정 & Ingress 설정을 해야하지만 본 워크샵에서는 생략하도록 하겠습니다.



#### 설치 목록

##### kubespray
Ansible을 기반으로, 쿠버네티스 클러스터를 설치해줍니다

##### kubectl
[kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)은 쿠버네티스 마스터와 대화할 수 있는 CLI툴입니다.

##### helm
[helm](https://helm.sh/)이란 쿠버네티스 package manager입니다. 해당 툴을 이용하여 필요한 모듈들을 쿠버네티스에 설치할 수 있습니다. apt, yum, pip 툴들과 비슷한 녀석이라고 생각하시면 됩니다.
오늘 helm을 통해서 Distributed ML job에 필요한 패키지들을 설치해볼 예정입니다.

##### helm chart
helm chart는 helm을 통해 설치하는 패키지 레포지토리를 말합니다. 오늘은 다음 chart들을 설치해볼 예정입니다.
- argo workflow: Data pipeline & ML workflow를 실행 시켜줄 wf engine입니다.
- nfs-client-provisioner: NAS 서버(EFS)와 연결 시켜주는 Storage Provisioner입니다.
- minio: NAS 서버를 웹으로 통해 볼 수 있게 minio UI를 사용합니다.
- cluster-autoscaler: 요청한 자원 만큼 서버 리소스를 늘려주는 k8s autoscaler입니다.
- metrics-server: 서버의 리소스 사용량을 확인하는 패키지입니다. (kubectl top node)


#### Setup



## 3. How to scale your ML job with k8s

### [1. Run a basic job](hands-on/01-run-job)
몸풀기! 간단한 `train.py` 코드를 이용하여 도커 이미지를 만들고 Job을 이용하여 학습을 시켜보겠습니다.

### [2. Save a model file to model storage](hands-on/02-save-model)
기계학습을 통해 얻어진 모델을 한곳에서 관리하고 싶을 때는 어떻게할 할수 있을까요?
매번 S3로 모델을 업로드하는 것이 귀찮으신가요?
NFS storage 타입 PVC를 이용하여 filesystem에 저장하는 것 만으로 모델을 한곳에 모아서 관리할 수 있게 구성해 봅시다.

### [3. Exception handling](hands-on/03-exception)
간혹 한개의 문제가 되는 학습 job 때문에 서버 전체에 장애가 발생하는 경우가 있습니다.
쿠버네티스를 이용한다면 문제가 되는 job 하나만을 종료되게끔 만들 수 있습니다.
인위적으로 Out of Memory 상황을 발생 시켜 쿠버네티스가 어떻게 handling하는지 확인해 보도록 하겠습니다.

### [4. Training with hyper-parameters](hands-on/04-train-hp)
여러가지 종류의 하이퍼파라미터들을 실험해 보고 싶을때는 어떻게 하면 좋을까요?
단순히 프로세스 파라미터 전달 방법 외에 다른 방법이 있을까요?
`ConfigMap`을 이용하여 파일 기반의 모델 파라미터를 전달해 봅시다.

### [5. Run multiple jobs](hands-on/05-run-multi)
복수의 기계학습 job을 동시에 실행 시켜봅니다. 다음과 같은 것을 확인해볼 예정입니다.
- 스케줄링
- Job 진행 상황
- 모니터링
- 에러처리
- Autoscaling

### [6. Using GPUs](hands-on/06-using-gpu/)
쿠버네티스에서 GPU 자원을 사용하는 방법에 대해서 알아보도록 하겠습니다.
특히나 GPU 자원은 비용이 비싸기 때문에 서버의 개수가 0개부터 시작하여 autoscaling이 되도록 설정해보겠습니다.

### [7. Hello workflow](hands-on/07-hello-wf/)
간단하게 Argo workflow에 대해서 알아보도록 하겠습니다.
Argo workflow란 쿠버네티스 job끼리 서로 dependency를 가실 수 있게 만들어주는 프레임워크입니다.
오늘 저희는 argo workflow를 이용하여 Data Pipeline을 만들어 볼 예정입니다.

### [8. DAG workflow](hands-on/08-wf-dag/)
Argo workflow를 이용하여 DAG (Directed acyclic graph)를 만드는 법을 살펴보겠습니다.
조금 복잡할 수도 있어서 따로 구분하여 hands-on을 준비하였습니다.

### [9. Building ML Pipeline](hands-on/09-ml-pipeline/)
Argo workflow를 이용하여 최종적으로 Data Pipeline을 만들어 보도록 하겠습니다.
S3에서 데이터를 가져와서 병렬로 분산하여 기계학습을 실행하여 NAS storage에 학습된 모델을 저장하고 최종적으로 slack으로 알람이 가게끔 만들어 보겠습니다.

### [10. Launch Jupyter notebook](hands-on/10-jupyter/)
JupyterHub를 이용하여 쿠버네티스 상에서 분석할 수 있는 환경을 구축해 보겠습니다.

### [11. Kubeflow tutorials](hands-on/11-kubeflow/) (Advanced - GCP only)
Kubernetes + tensorflow 조합으로 탄생한 kubeflow에 대해서 간단히 알아보고  
codelab 튜토리얼에 대해서 소개해 드립니다.

