# 2. Save a model file to model storage

NFS storage PVC를 하나 생성합니다. 이름은 `model-storage`라고 정하겠습니다.
앞으로 해당 PVC에 모델 파일들을 저장할 예정입니다.

```bash
# Create model storage PVC
cat <<EOF | kubectl create -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: nfs-storage
EOF

kubectl get pvc
```

PVC 생성후 minio를 통해 생성된 PVC를 확인해 보겠습니다.
```bash
kubectl get svc -nkube-system  # --> external IP 확인
```

모델을 특정 위치에 저장하도록 `train.py`를 수정해 보겠습니다.

![](02-pvc.png)

#### 실행
```bash
docker build . -t !IMAGE:02
docker push !IMAGE:02
vim job.yaml # image 수정
kubectl apply -f job.yaml
```

### 확인사항
`model-storage` PVC에 원하는 모델파일이 생성이 되었는지 minio를 통하여 확인


### Do it more

인위적으로 서로 다른 노드에 job을 실행시켜서 저장한 모델 파일이 어디서든 보이는지 확인해 보겠습니다.
