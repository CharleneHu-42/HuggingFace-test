# Run Guide

## 1. Build dockerfile: 

These two dockerfiles are internal and external respectively.external uses the offical torch, ipex, ccl version (1.12)
```bash
for internal: docker build -f df_internal --build-arg imageVersion=2022_ww26 -t bert_qa:internal .

for external: docker build -f df_external --build-arg http_proxy=http://proxy-chain.intel.com:911 --build-arg https_proxy=http://proxy-chain.intel.com:911 --build-arg no_proxy=*.intel.com -t bert_qa:external .
```
notes:

The imageVersion is the version of base image from internal release of torch,ipex and ccl, it may be changed.

## 2. Pull the prebuild docker image from docker hub

```bash
for internal: docker pull appliedmlwf/hf:bert_qa_internal_2022ww26

for external: docker pull appliedmlwf/hf:bert_qa_external_2022ww31
```

## 3. Docker containers deployment in single node:

notes:
```bash
help: ./host_qa_test.sh -h

./host_qa_test.sh -c 0 -i {image_id} #2DDP in 1 container(master)
./host_qa_test.sh -c 1 -i {image_id} #4DPP in 2 containers(master and slave0) in single node

```
where `{image_id}` could be appliedmlwf/hf:bert_qa_external_2022ww31 or appliedmlwf/hf:bert_qa_internal_2022ww26 based on your needs

the output is /tmp/debug_squad, you could change it in host_qa_test.sh

please stop and remove existing containers named master and slave0 before starting the test

```bash
docker stop master
docker rm master
docker stop slave0
docker rm slave0
```
multiple dockers utilize "bridge" network mode here, hg-bridge is created in the host_qa_test.sh and you could remove it by "docker network rm hg-bridge"

## 4. Docker containers deployment in multiple nodes:

two scripts are provided. master_qa_node.sh should be launched in master node and 
slave_qa_node.sh should be launched in slave node. ***containers in slave node must be created before master***

**In slave node:**

```bash
help: ./slave_qa_node.sh -h

./slave_qa_node.sh -i {image_id} -n ov_net1 -s 0 #slave0
./slave_qa_node.sh -i {image_id} -n ov_net1 -s 1 #slave1
```

**In master node:**
```bash
help: ./master_qa_node.sh -h

echo "master" > /tmp/hostfile
echo "slave0" >> /tmp/hostfile
echo "slave1" >> /tmp/hostfile

./master_qa_node.sh -i {image_id} -n ov_net1 -p 6 -n ov_net1 #total 6DPP, 2DDP per container instance
```

where `{image_id}` could be appliedmlwf/hf:bert_qa_external_2022ww31 or appliedmlwf/hf:bert_qa_internal_2022ww26 based on your needs

the output is /tmp/debug_squad, you could change it in master_qa_node.sh

please stop and remove existing containers named master and slave0, slave1 before starting the test

```bash
docker stop master
docker rm master
docker stop slave0
docker rm slave0
docker stop slave1
docker rm slave1
```
multiple dockers in multiple nodes utilize "overlay" network mode here. Following is the step to create it.
for ex. we have two nodes(10.165.9.49, 10.165.9.48)

**install consul and create consul service**

in 10.165.9.49
```bash
docker pull progrium/consul
docker run -d -p 8500:8500 -h consul --name consul progrium/consul -server -bootstrap
```
you could open 10.165.9.49:8500 in browser to check the consul status

**configure the docker listening port and consul addr**

in 10.165.9.49
```bash
vi /etc/docker/daemon.json
{
  "hosts":["tcp://0.0.0.0:2376","unix:///var/run/docker.sock"], #listening in port 2376
  "cluster-store": "consul://10.165.9.49:8500", 
  "cluster-advertise": "10.165.9.49:2376"
}
```
in 10.165.9.48

```bash
vi /etc/docker/daemon.json
{
  "hosts":["tcp://0.0.0.0:2376","unix:///var/run/docker.sock"],
  "cluster-store": "consul://10.165.9.49:8500",
  "cluster-advertise": "10.165.9.48:2376"
}
```
change  
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock  
to  
ExecStart=/usr/bin/dockerd --containerd=/run/containerd/containerd.sock

**restart docker service**

```bash
systemctl restart docker
```
you could see the nodes(10.165.9.48 and 10.165.9.49) shown in the 10.165.9.49:8500

**create the global overlay network**
```bash
docker network create -d overlay ov_net1

docker network ls # to check its scope
```
and ov_net1 could be used when creating container in section 4