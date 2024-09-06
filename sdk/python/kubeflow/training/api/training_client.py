# Copyright 2023 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import multiprocessing
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from kubeflow.storage_initializer.constants import (
    VOLUME_PATH_DATASET,
    VOLUME_PATH_MODEL,
)
from kubeflow.training import models
from kubeflow.training.api_client import ApiClient
from kubeflow.training.constants import constants
from kubeflow.training.utils import utils
from kubernetes import client, config, watch

logger = logging.getLogger(__name__)

status_logger = utils.StatusLogger(
    header="{:<30.30} {:<20.20} {}".format("NAME", "STATE", "TIME"),
    column_format="{:<30.30} {:<20.20} {}",
)


class TrainingClient(object):
    def __init__(
        self,
        config_file: Optional[str] = None,
        context: Optional[str] = None,
        client_configuration: Optional[client.Configuration] = None,
        namespace: str = utils.get_default_target_namespace(),
        job_kind: str = constants.PYTORCHJOB_KIND,
    ):
        """
        TrainingClient constructor. Configure logging in your application
        as follows to see detailed information from the TrainingClient APIs:

        .. code-block:: python

            import logging
            logging.basicConfig()
            log = logging.getLogger("kubeflow.training.api.training_client")
            log.setLevel(logging.DEBUG)

        :param str config_file: Path to the kube-config file. Defaults to ``~/.kube/config``.
        :param str context: Set the active context. Defaults to ``current_context`` from the kube-config.
        :param client_configuration: Client configuration for cluster authentication.
            You must provide valid configuration with a Bearer token or with a username and password.
            An example can be found here:
            https://github.com/kubernetes-client/python/blob/67f9c7a97081b4526470cad53576bc3b71fa6fcc/examples/remote_cluster.py#L31
        :param str namespace: Target Kubernetes namespace. By default, it takes the namespace
            from `/var/run/secrets/kubernetes.io/serviceaccount/namespace` or is set as `default`.
            The namespace can be overridden during method invocations.
        :param str job_kind: Target Training Job kind (e.g., ``TFJob``, ``PyTorchJob``, ``MPIJob``).
            Job kind can be overridden during method invocations. The default Job kind is ``PyTorchJob``.

        :raises ValueError: If the Job kind is invalid.
        """


        # If client configuration is not set, use kube-config to access Kubernetes APIs.
        if client_configuration is None:
            # Load kube-config or in-cluster config.
            if config_file or not utils.is_running_in_k8s():
                config.load_kube_config(config_file=config_file, context=context)
            else:
                config.load_incluster_config()

        k8s_client = client.ApiClient(client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)
        self.api_client = ApiClient()

        self.namespace = namespace
        if job_kind not in constants.JOB_PARAMETERS:
            raise ValueError(
                f"Job kind must be one of these: {list(constants.JOB_PARAMETERS.keys())}"
            )
        self.job_kind = job_kind

    def train(
        self,
        name: str,
        namespace: Optional[str] = None,
        num_workers: int = 1,
        num_procs_per_worker: int = 1,
        resources_per_worker: Union[dict, client.V1ResourceRequirements, None] = None,
        model_provider_parameters=None,
        dataset_provider_parameters=None,
        trainer_parameters=None,
        storage_config: Dict[str, Optional[Union[str, List[str]]]] = {
            "size": constants.PVC_DEFAULT_SIZE,
            "storage_class": None,
            "access_modes": constants.PVC_DEFAULT_ACCESS_MODES,
        },
    ):
        """
        High-level API to fine-tune LLMs with distributed PyTorchJob. Follow this guide
        for more information about this feature: TODO (andreyvelich): Add link.

        It uses the pre-created Storage Initializer to download a pre-trained model and dataset, and
        Trainer to fine-tune LLM. Your cluster should support PVC with `ReadOnlyMany` access mode
        to distribute data across PyTorchJob workers.

        It uses the `torchrun` CLI to fine-tune models in distributed mode with multiple PyTorchJob
        workers. Follow this guide to learn more about `torchrun` CLI:
        https://pytorch.org/docs/stable/elastic/run.html.

        This feature is in alpha stage and the Kubeflow community is looking for feedback.
        Please use the `#kubeflow-training` Slack channel or the Kubeflow Training Operator GitHub
        for your questions or suggestions.

        :param str name: Name of the PyTorchJob.
        :param str namespace: Namespace for the PyTorchJob. By default, the namespace is taken from
            the `TrainingClient` object.
        :param int num_workers: Number of PyTorchJob workers.
        :param int num_procs_per_worker: Number of processes per PyTorchJob worker for `torchrun` CLI.
            Use this parameter to run more than one GPU per PyTorchJob worker.
        :param resources_per_worker: Resources each PyTorchJob worker container should have. You can either specify a
            `kubernetes.client.V1ResourceRequirements` object or a dictionary with one or more of the following keys:
            `cpu`, `memory`, or `gpu`. For example:

            .. code-block:: yaml

                {
                    "cpu": "1",
                    "memory": "2Gi",
                    "gpu": "1",
                }

            `gpu` specifies a resource request with a key of `nvidia.com/gpu` (for NVIDIA GPUs). If you need a different type
            of GPU, use a `V1ResourceRequirement` instance. This parameter is optional and defaults to None.
        :param model_provider_parameters: Parameters for the model provider in the Storage Initializer.
            For example, HuggingFace model name and Transformer type for that model, like:
            `AutoModelForSequenceClassification`. This argument must be of the type
            `kubeflow.storage_initializer.hugging_face.HuggingFaceModelParams`.
        :param dataset_provider_parameters: Parameters for the dataset provider in the Storage Initializer.
            For example, the name of the HuggingFace dataset or AWS S3 configuration. This argument must be of the type
            `kubeflow.storage_initializer.hugging_face.HuggingFaceDatasetParams` or
            `kubeflow.storage_initializer.s3.S3DatasetParams`.
        :param trainer_parameters: Parameters for the LLM Trainer that will fine-tune the pre-trained model
            with the provided dataset. For example, LoRA config for parameter-efficient fine-tuning
            and HuggingFace training arguments like optimizer or number of training epochs.
            This argument must be of the type
            `kubeflow.storage_initializer.HuggingFaceTrainerParams`.
        :param storage_config: Configuration for the Storage Initializer PVC to download the pre-trained model
            and dataset. You can configure the PVC size and storage class name in this argument.

        """
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "Train API dependencies not installed. "
                + "Run: pip install -U 'kubeflow-training[huggingface]' "
            )

        # fmt: off

        from kubeflow.storage_initializer.hugging_face import (
            HuggingFaceDatasetParams,
            HuggingFaceModelParams,
        )
        from kubeflow.storage_initializer.s3 import S3DatasetParams

        # fmt: on

        print(
            "Thank you for using `train` API for LLMs fine-tuning. This feature is in alpha stage "
            "Kubeflow community is looking for your feedback. Please share your experience "
            "via #kubeflow-training Slack channel or Kubeflow Training Operator GitHub."
        )

        if (
            not name
            or not model_provider_parameters
            or not dataset_provider_parameters
            or not trainer_parameters
        ):
            raise ValueError("One of the required parameters is None")

        namespace = namespace or self.namespace

        # TODO (andreyvelich): PVC Creation should be part of Training Operator Controller.
        # Ref issue: https://github.com/kubeflow/training-operator/issues/1971
        try:
            self.core_api.create_namespaced_persistent_volume_claim(
                namespace=namespace,
                body=utils.get_pvc_spec(
                    pvc_name=name,
                    namespace=namespace,
                    storage_config=storage_config,
                ),
            )
        except Exception as e:
            pvc_list = self.core_api.list_namespaced_persistent_volume_claim(namespace)
            # Check if the PVC with the specified name exists
            for pvc in pvc_list.items:
                if pvc.metadata.name == name:
                    print(f"PVC '{name}' already exists in namespace " f"{namespace}.")
                    break
            else:
                raise RuntimeError(f"failed to create PVC. Error: {e}")

        if isinstance(model_provider_parameters, HuggingFaceModelParams):
            mp = "hf"
        else:
            raise ValueError(
                f"Invalid model provider parameters {model_provider_parameters}"
            )

        if isinstance(dataset_provider_parameters, S3DatasetParams):
            dp = "s3"
        elif isinstance(dataset_provider_parameters, HuggingFaceDatasetParams):
            dp = "hf"
        else:
            raise ValueError(
                f"Invalid dataset provider parameters {dataset_provider_parameters}"
            )

        # create init container spec
        init_container_spec = utils.get_container_spec(
            name=constants.STORAGE_INITIALIZER,
            base_image=constants.STORAGE_INITIALIZER_IMAGE,
            args=[
                "--model_provider",
                mp,
                "--model_provider_parameters",
                json.dumps(model_provider_parameters.__dict__, cls=utils.SetEncoder),
                "--dataset_provider",
                dp,
                "--dataset_provider_parameters",
                json.dumps(dataset_provider_parameters.__dict__),
            ],
            volume_mounts=[constants.STORAGE_INITIALIZER_VOLUME_MOUNT],
        )

        # create app container spec
        container_spec = utils.get_container_spec(
            name=constants.JOB_PARAMETERS[constants.PYTORCHJOB_KIND]["container"],
            base_image=constants.TRAINER_TRANSFORMER_IMAGE,
            args=[
                "--model_uri",
                model_provider_parameters.model_uri,
                "--transformer_type",
                model_provider_parameters.transformer_type.__name__,
                "--num_labels",
                str(model_provider_parameters.num_labels),
                "--model_dir",
                VOLUME_PATH_MODEL,
                "--dataset_dir",
                VOLUME_PATH_DATASET,
                "--lora_config",
                json.dumps(
                    trainer_parameters.lora_config.__dict__, cls=utils.SetEncoder
                ),
                "--training_parameters",
                json.dumps(trainer_parameters.training_parameters.to_dict()),
            ],
            volume_mounts=[constants.STORAGE_INITIALIZER_VOLUME_MOUNT],
            resources=resources_per_worker,
        )

        storage_initializer_volume = models.V1Volume(
            name=constants.STORAGE_INITIALIZER,
            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                claim_name=name
            ),
        )

        # create worker pod spec
        worker_pod_template_spec = utils.get_pod_template_spec(
            containers=[container_spec],
            volumes=[storage_initializer_volume],
        )

        # create master pod spec
        master_pod_template_spec = utils.get_pod_template_spec(
            containers=[container_spec],
            init_containers=[init_container_spec],
            volumes=[storage_initializer_volume],
        )

        job = utils.get_pytorchjob_template(
            name=name,
            namespace=namespace,
            master_pod_template_spec=master_pod_template_spec,
            worker_pod_template_spec=worker_pod_template_spec,
            num_workers=num_workers,
            num_procs_per_worker=num_procs_per_worker,
        )

        self.create_job(job, namespace=namespace)

    def create_job(
        self,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        base_image: Optional[str] = None,
        train_func: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        num_workers: Optional[int] = None,
        resources_per_worker: Union[dict, models.V1ResourceRequirements, None] = None,
        num_chief_replicas: Optional[int] = None,
        num_ps_replicas: Optional[int] = None,
        packages_to_install: Optional[List[str]] = None,
        pip_index_url: str = constants.DEFAULT_PIP_INDEX_URL,
    ):
        """
        Create the Training Job.

        The job can be created using one of the following options:

        - Define a custom resource object in the `job` parameter (e.g. `TFJob` or `PyTorchJob`).
        - Define a training function in the `train_func` parameter and specify the number of workers.
        - Define a Docker image in the `base_image` parameter and specify the number of workers.

        :param job: Job object. Must be one of the following types: `KubeflowOrgV1TFJob`,
            `KubeflowOrgV1PyTorchJob`, etc.
        :param str name: Name of the job. This must be set if the `job` parameter is omitted.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g. `TFJob` or `PyTorchJob`). This must be set if
            the `job` parameter is omitted. By default, the job kind is taken from the
            `TrainingClient` object.
        :param str base_image: Docker image that the job uses to train the model on each training replica.
            If the `train_func` parameter is set, this image is used to execute the training function.
            The `constants` module contains some base images, with the default image being
            `docker.io/pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`.
        :param Callable train_func: Function that the job uses to train the model on each training replica.
            This function must be callable. Optionally, this function can take a single `dict` argument
            to define input parameters. If `train_func` is set, the base image must support the `bash`
            CLI to execute the training script.
        :param dict parameters: Dictionary of input parameters that the training function might receive.
        :param int num_workers: Number of worker replicas for the job.
        :param resources_per_worker: Resources for each worker container. You can either specify a
            `kubernetes.client.V1ResourceRequirements` object or a dictionary that includes one or more of the
            following keys: `cpu`, `memory`, or `gpu`. For example:

            .. code-block:: yaml

                {
                    "cpu": "1",
                    "memory": "2Gi",
                    "gpu": "1",
                }

            `gpu` specifies a resource request with the key `nvidia.com/gpu` (for NVIDIA GPUs). For other types of GPUs, use a
            `V1ResourceRequirements` instance. This parameter is optional and defaults to `None`.
        :param int num_chief_replicas: Number of Chief replicas for the `TFJob`. The number of Chief replicas
            cannot exceed 1.
        :param int num_ps_replicas: Number of Parameter Server replicas for the `TFJob`.
        :param list packages_to_install: List of Python packages to install in addition to those in the base image
            if the `train_func` parameter is set. These packages are installed before executing the training function.
        :param str pip_index_url: PyPI URL from which to install Python packages.

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If job creation times out.
        :raises RuntimeError: If the job fails to be created.

        """

        # When Job is set, only namespace arg is allowed.
        if job is not None:
            for key, value in locals().items():
                if (
                    key not in ["self", "job", "namespace", "pip_index_url"]
                    and value is not None
                ):
                    raise ValueError(
                        "If `job` is set only `namespace` argument is allowed. "
                        f"Argument `{key}` must be None."
                    )

        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind
        if job is not None:
            job_kind = str(job.kind)

        if job_kind not in constants.JOB_PARAMETERS:
            raise ValueError(
                f"Job kind must be one of these: {constants.JOB_PARAMETERS.keys()}"
            )

        # If Training function or base image is set, configure Job template.
        if job is None and (train_func is not None or base_image is not None):
            # Job name must be set to configure Job template.
            if name is None:
                raise ValueError(
                    "Job name must be set to configure Job from function or image"
                )

            # Assign the default base image.
            # TODO (andreyvelich): Add base image for other Job kinds.
            if base_image is None:
                base_image = constants.JOB_PARAMETERS[job_kind]["base_image"]

            # Get Training Container template.
            container_spec = utils.get_container_spec(
                name=constants.JOB_PARAMETERS[job_kind]["container"],
                base_image=base_image,
                train_func=train_func,
                train_func_parameters=parameters,
                packages_to_install=packages_to_install,
                pip_index_url=pip_index_url,
                resources=resources_per_worker,
            )

            # Get Pod template spec using the above container.
            pod_template_spec = utils.get_pod_template_spec(
                containers=[container_spec],
            )

            # Configure template for different Jobs.
            # TODO (andreyvelich): Add support for other kinds (e.g. MPIJob).
            if job_kind == constants.TFJOB_KIND:
                job = utils.get_tfjob_template(
                    name=name,
                    namespace=namespace,
                    pod_template_spec=pod_template_spec,
                    num_workers=num_workers,
                    num_chief_replicas=num_chief_replicas,
                    num_ps_replicas=num_ps_replicas,
                )
            elif job_kind == constants.PYTORCHJOB_KIND and num_workers:
                job = utils.get_pytorchjob_template(
                    name=name,
                    namespace=namespace,
                    worker_pod_template_spec=pod_template_spec,
                    num_workers=num_workers,
                )
            else:
                raise ValueError(
                    f"Job kind {job_kind} can't be created using function or image. "
                    + "Number of Workers must be set."
                )

        # Verify Job object type.
        if not isinstance(
            job,
            getattr(models, constants.JOB_PARAMETERS[job_kind]["model"]),
        ):
            raise ValueError(
                f"Job must be one of these types: {constants.JOB_MODELS}, but Job is: {type(job)}"
            )

        # Create the Training Job.
        try:
            self.custom_api.create_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                namespace,
                constants.JOB_PARAMETERS[job_kind]["plural"],
                job,
            )
        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to create {job_kind}: {namespace}/{job.metadata.name}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to create {job_kind}: {namespace}/{job.metadata.name}"
            )

        logger.debug(f"{job_kind} {namespace}/{job.metadata.name} has been created")

    def get_job(
        self,
        name: str,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> constants.JOB_MODELS_TYPE:
        """
        Get the Training Job.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g. `TFJob` or `PyTorchJob`). By default, the job kind is
            taken from the `TrainingClient` object.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: Job object, such as `KubeflowOrgV1PyTorchJob`.
        :rtype: object

        :raises TimeoutError: If the request times out when trying to get the job.
        :raises RuntimeError: If the job retrieval fails.

"""
        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        if job_kind not in constants.JOB_PARAMETERS:
            raise ValueError(
                f"Job kind must be one of these: {constants.JOB_PARAMETERS.keys()}"
            )

        try:
            thread = self.custom_api.get_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                namespace,
                constants.JOB_PARAMETERS[job_kind]["plural"],
                name,
                async_req=True,
            )
            response = utils.FakeResponse(thread.get(timeout))
            job = self.api_client.deserialize(
                response, constants.JOB_PARAMETERS[job_kind]["model"]
            )

        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Timeout to get {job_kind}: {namespace}/{name}")
        except Exception:
            raise RuntimeError(f"Failed to get {job_kind}: {namespace}/{name}")

        return job

    def list_jobs(
        self,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> List[constants.JOB_MODELS_TYPE]:
        """
        List all Training Jobs of a specific kind in a namespace.

        :param str namespace: Namespace to list the jobs. By default, the namespace is taken from the
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g. `TFJob` or `PyTorchJob`). By default, the job kind is
            taken from the `TrainingClient` object.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: List of job objects. For example, a list of `KubeflowOrgV1PyTorchJob` objects. Returns
            an empty list if no jobs are found.
        :rtype: list[object]

        :raises TimeoutError: If the request times out while trying to list jobs.
        :raises RuntimeError: If listing jobs fails.

        """


        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        if job_kind not in constants.JOB_PARAMETERS:
            raise ValueError(
                f"Job kind must be one of these: {constants.JOB_PARAMETERS.keys()}"
            )

        result = []
        try:
            thread = self.custom_api.list_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                namespace,
                constants.JOB_PARAMETERS[job_kind]["plural"],
                async_req=True,
            )
            response = thread.get(timeout)
            result = [
                self.api_client.deserialize(
                    utils.FakeResponse(item),
                    constants.JOB_PARAMETERS[job_kind]["model"],
                )
                for item in response.get("items")
            ]
        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Timeout to list {job_kind}s in namespace: {namespace}")
        except Exception:
            raise RuntimeError(f"Failed to list {job_kind}s in namespace: {namespace}")

        return result

    def get_job_conditions(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> List[models.V1JobCondition]:
        """
        Get the Training Job conditions. A Training Job is in a specific condition when `status=True` 
        for the appropriate condition `type`. For example, a Training Job is considered Succeeded when 
        `status=True` and `type=Succeeded`.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param object job: Job object to get the conditions. The object must be one of these types: 
            `KubeflowOrgV1TFJob`, `KubeflowOrgV1PyTorchJob`, etc. If omitted, it retrieves the job using 
            the provided name and kind.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: List of job conditions with the last transition time, last update time, message, 
            reason, type, and status. Returns an empty list if the job has no conditions yet.
        :rtype: list[V1JobCondition]

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If getting the job fails.
        """


        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        if job_kind not in constants.JOB_PARAMETERS:
            raise ValueError(
                f"Job kind must be one of these: {constants.JOB_PARAMETERS.keys()}"
            )

        if job is not None and not isinstance(
            job, getattr(models, constants.JOB_PARAMETERS[job_kind]["model"])
        ):
            raise ValueError(f"Job must be one of these types: {constants.JOB_MODELS}")

        # If Job is not set, get the Training Job.
        if job is None:
            # Job name must be set when Job object is not set.
            if name is None:
                raise ValueError(
                    "Job name must be set to configure Job from function or image"
                )

            job = self.get_job(
                name=name,
                namespace=namespace,
                job_kind=job_kind,
                timeout=timeout,
            )
        if job.status and job.status.conditions and len(job.status.conditions) > 0:
            return job.status.conditions
        return []

    def is_job_created(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Check if the Training Job is Created.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param object job: Job object used to check conditions. The object must be one of the following 
            types: `KubeflowOrgV1TFJob`, `KubeflowOrgV1PyTorchJob`, etc. If omitted, it retrieves the 
            job by the given name and kind.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: True if the job is created, else False.
        :rtype: bool

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If retrieving the job fails.
        """

        return utils.has_condition(
            self.get_job_conditions(name, namespace, job_kind, job, timeout),
            constants.JOB_CONDITION_CREATED,
        )

    def is_job_running(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Check if the Training Job is Running.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param object job: Job object used to check conditions. The object must be one of the following 
            types: `KubeflowOrgV1TFJob`, `KubeflowOrgV1PyTorchJob`, etc. If omitted, it retrieves the 
            job by the given name and kind.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: True if the job is running, else False.
        :rtype: bool

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If retrieving the job fails.
        """

        return utils.has_condition(
            self.get_job_conditions(name, namespace, job_kind, job, timeout),
            constants.JOB_CONDITION_RUNNING,
        )

    def is_job_restarting(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Check if the Training Job is Restarting.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param object job: Job object used to check conditions. The object must be one of the following 
            types: `KubeflowOrgV1TFJob`, `KubeflowOrgV1PyTorchJob`, etc. If omitted, it retrieves the 
            job by the given name and kind.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: True if the job is restarting, else False.
        :rtype: bool

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If retrieving the job fails.
        """

        return utils.has_condition(
            self.get_job_conditions(name, namespace, job_kind, job, timeout),
            constants.JOB_CONDITION_RESTARTING,
        )

    def is_job_succeeded(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Check if the Training Job has Succeeded.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param object job: Job object used to get the conditions. The object must be one of the following 
            types: `KubeflowOrgV1TFJob`, `KubeflowOrgV1PyTorchJob`, etc. If omitted, it retrieves the 
            job by the given name and kind.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: True if the job has succeeded, else False.
        :rtype: bool

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If retrieving the job fails.
        """

        return utils.has_condition(
            self.get_job_conditions(name, namespace, job_kind, job, timeout),
            constants.JOB_CONDITION_SUCCEEDED,
        )

    def is_job_failed(
        self,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        job: Optional[constants.JOB_MODELS_TYPE] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> bool:
        """
        Check if the Training Job has Failed.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param object job: Job object used to get the conditions. The object must be one of the following 
            types: `KubeflowOrgV1TFJob`, `KubeflowOrgV1PyTorchJob`, etc. If omitted, it retrieves the 
            job by the given name and kind.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: True if the job has failed, else False.
        :rtype: bool

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If retrieving the job fails.
        """


        return utils.has_condition(
            self.get_job_conditions(name, namespace, job_kind, job, timeout),
            constants.JOB_CONDITION_FAILED,
        )

    def wait_for_job_conditions(
        self,
        name: str,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        expected_conditions: Set = {constants.JOB_CONDITION_SUCCEEDED},
        wait_timeout: int = 600,
        polling_interval: int = 15,
        callback: Optional[Callable] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> constants.JOB_MODELS_TYPE:
        """
        Wait until the Training Job reaches any of the specified conditions.
        By default, it waits for the `Succeeded` condition.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind 
            is taken from the `TrainingClient` object.
        :param set expected_conditions: Set of expected conditions. Must be a subset of 
            `{"Created", "Running", "Restarting", "Succeeded", "Failed"}`.
        :param int wait_timeout: Number of seconds to wait until the job reaches one of the 
            expected conditions.
        :param int polling_interval: The polling interval in seconds to check the job status.
        :param callable callback: Callback function invoked after each status poll. This function 
            takes a single argument, which is the current job object.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: The job object.
        :rtype: object
            For example: `KubeflowOrgV1PyTorchJob`.

        :raises ValueError: If input parameters are invalid.
        :raises TimeoutError: If the request times out while trying to get the job.
        :raises RuntimeError: If retrieving the job fails, or if the job reaches the `Failed` condition 
            and `Failed` is not in the `expected_conditions` set.
        """

        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        if not expected_conditions.issubset(constants.JOB_CONDITIONS):
            raise ValueError(
                f"Expected conditions: {expected_conditions} must be subset of \
                    {constants.JOB_CONDITIONS}"
            )
        for _ in range(round(wait_timeout / polling_interval)):
            # We should get Job only once per cycle and check the statuses.
            job = self.get_job(
                name=name,
                namespace=namespace,
                job_kind=job_kind,
                timeout=timeout,
            )

            # Get Job conditions.
            conditions = self.get_job_conditions(job=job, timeout=timeout)
            if len(conditions) > 0:
                status_logger(
                    name,
                    conditions[-1].type,
                    conditions[-1].last_transition_time,
                )

            # Execute callback function is it is set.
            if callback:
                callback(job)

            # Raise an exception if Job is Failed and Failed is not the expected condition.
            if (
                constants.JOB_CONDITION_FAILED not in expected_conditions
                and utils.has_condition(conditions, constants.JOB_CONDITION_FAILED)
            ):
                raise RuntimeError(
                    f"{job_kind} {namespace}/{name} is Failed. "
                    f"{job_kind} conditions: {job.status.conditions}"
                )

            # Return Job when it reaches expected condition.
            for expected_condition in expected_conditions:
                if utils.has_condition(conditions, expected_condition):
                    return job

            time.sleep(polling_interval)

        raise TimeoutError(
            f"Timeout waiting for {job_kind}: {namespace}/{name} to reach expected conditions: \
                {expected_conditions}"
        )

    def get_job_pods(
        self,
        name: str,
        namespace: Optional[str] = None,
        is_master: bool = False,
        replica_type: Optional[str] = None,
        replica_index: Optional[int] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> List[models.V1Pod]:
        """
        Get pods for the Training Job.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param bool is_master: Whether to get pods only with the label 
            `training.kubeflow.org/job-role: master`.
        :param str replica_type: Type of the job replica.
            For `TFJob`: One of `Chief`, `PS`, or `worker`.
            For `PyTorchJob`: One of `master` or `worker`.
            For `XGBoostJob`: One of `master` or `worker`.
            For `MPIJob`: One of `launcher` or `worker`.
            For `PaddleJob`: One of `master` or `worker`.
        :param int replica_index: Index for the job replica.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: List of job pods.
        :rtype: list[V1Pod]

        :raises ValueError: If the job replica type is invalid.
        :raises TimeoutError: If the request times out while trying to get job pods.
        :raises RuntimeError: If retrieving job pods fails.
        """


        namespace = namespace or self.namespace

        if (
            replica_type is not None
            and replica_type not in constants.TFJOB_REPLICA_TYPES
            and replica_type not in constants.PYTORCHJOB_REPLICA_TYPES
            and replica_type not in constants.XGBOOSTJOB_REPLICA_TYPES
            and replica_type not in constants.MPIJOB_REPLICA_TYPES
            and replica_type not in constants.PADDLEJOB_REPLICA_TYPES
        ):
            raise ValueError(
                f"TFJob replica type must be one of {constants.TFJOB_REPLICA_TYPES}\n"
                f"PyTorchJob replica type must be one of {constants.PYTORCHJOB_REPLICA_TYPES}\n"
                f"XGBoostJob replica type must be one of {constants.XGBOOSTJOB_REPLICA_TYPES}\n"
                f"MPIJob replica type must be one of {constants.MPIJOB_REPLICA_TYPES}\n"
                f"PaddleJob replica type must be one of {constants.PADDLEJOB_REPLICA_TYPES}"
            )

        label_selector = f"{constants.JOB_NAME_LABEL}={name}"

        # Add Job role label if that is required.
        if is_master:
            label_selector += f",{constants.JOB_ROLE_LABEL}={constants.JOB_ROLE_MASTER}"

        # Add Replica type label if that is required.
        if replica_type:
            label_selector += (
                f",{constants.REPLICA_TYPE_LABEL}={str.lower(replica_type)}"
            )

        # Add Replica index label if that is required.
        if replica_index is not None:
            label_selector += f",{constants.REPLICA_INDEX_LABEL}={replica_index}"

        # Return list of Training Job pods.
        try:
            thread = self.core_api.list_namespaced_pod(
                namespace,
                label_selector=label_selector,
                async_req=True,
            )
            return thread.get(timeout).items
        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Timeout to list pods for Job: {namespace}/{name}")
        except Exception:
            raise RuntimeError(f"Failed to list pods for Job: {namespace}/{name}")

    def get_job_pod_names(
        self,
        name: str,
        namespace: Optional[str] = None,
        is_master: bool = False,
        replica_type: Optional[str] = None,
        replica_index: Optional[int] = None,
        timeout: int = constants.DEFAULT_TIMEOUT,
    ) -> List[str]:
        """
        Get pod names for the Training Job.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the 
            `TrainingClient` object.
        :param bool is_master: Whether to get pods only with the label 
            `training.kubeflow.org/job-role: master`.
        :param str replica_type: Type of the job replica.
            For `TFJob`: One of `Chief`, `PS`, or `worker`.
            For `PyTorchJob`: One of `master` or `worker`.
            For `XGBoostJob`: One of `master` or `worker`.
            For `MPIJob`: One of `launcher` or `worker`.
            For `PaddleJob`: One of `master` or `worker`.
        :param int replica_index: Index for the job replica.
        :param int timeout: Kubernetes API server timeout in seconds to execute the request.

        :returns: List of job pod names.
        :rtype: list[str]

        :raises ValueError: If the job replica type is invalid.
        :raises TimeoutError: If the request times out while trying to get job pods.
        :raises RuntimeError: If retrieving job pods fails.
        """


        namespace = namespace or self.namespace

        pods = self.get_job_pods(
            name=name,
            namespace=namespace,
            is_master=is_master,
            replica_type=replica_type,
            replica_index=replica_index,
            timeout=timeout,
        )
        pod_names = []
        for pod in pods:
            pod_names.append(pod.metadata.name)
        return pod_names
    
    def get_job_logs(
        self,
        name: str,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        is_master: bool = True,
        replica_type: Optional[str] = None,
        replica_index: Optional[int] = None,
        follow: bool = False,
        timeout: int = constants.DEFAULT_TIMEOUT,
        verbose: bool = False,
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Get the logs for every Training Job pod. By default, it returns logs from
        the `master` pod. Logs are returned in this format: ``{ "pod-name": "Log data" }``.

        :param str name: Name for the Job.
        :param str namespace: Namespace for the Job. By default, the namespace is taken from
            the `TrainingClient` object.
        :param str job_kind: Kind for the Job (e.g., `TFJob` or `PyTorchJob`). By default, the Job kind
            is taken from the `TrainingClient` object.
        :param bool is_master: Whether to get logs for the pod with the label
            `training.kubeflow.org/job-role: master`.
        :param str replica_type: Optional. Type of the Job replica:
            - For `TFJob`: one of `chief`, `ps`, or `worker`.
            - For `PyTorchJob`: one of `master` or `worker`.
            - For `XGBoostJob`: one of `master` or `worker`.
            - For `MPIJob`: one of `launcher` or `worker`.
            - For `PaddleJob`: one of `master` or `worker`.
        :param int replica_index: Optional. Index for the Job replica.
        :param str container: Pod container to get the logs.
        :param bool follow: Whether to follow the log stream of the pod and print logs to StdOut.
        :param int timeout: Optional. Kubernetes API server timeout in seconds to execute the request.
        :param bool verbose: Whether to get Kubernetes events for Job and corresponding pods.
            If you need to get events from all `PyTorchJob`'s Pods, set `is_master` to `False`.

        :returns: A dictionary where the keys are pod names and the values are the corresponding logs.
            If `verbose` is `True`, it also returns a dictionary where the keys are object kinds and names, and the
            values are lists of the corresponding Kubernetes events with their timestamps. For example:

            .. code-block:: json

                {
                    "PyTorchJob train-mnist": [
                        "2024-01-05 22:58:20 Created pod: train-mnist-worker-0"
                    ],
                    "Pod train-mnist-worker-0": [
                        "2024-01-05 22:58:20 Created container init-pytorch"
                    ]
                }

        :raises ValueError: If the Job replica type is invalid.
        :raises TimeoutError: If the request times out while getting the Job or Job's pods.
        :raises RuntimeError: If retrieving the Job or Job's pods fails.
        """
        
        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        pods = self.get_job_pods(
            name=name,
            namespace=namespace,
            is_master=is_master,
            replica_type=replica_type,
            replica_index=replica_index,
            timeout=timeout,
        )

        logs_dict = {}
        events_dict = {}
        if pods and follow:
            log_streams = []
            for pod in pods:
                if (
                    pod.status is not None
                    and pod.status.phase != constants.POD_PHASE_PENDING
                ):
                    log_streams.append(
                        watch.Watch().stream(
                            self.core_api.read_namespaced_pod_log,
                            name=pod.metadata.name,
                            namespace=namespace,
                            container=constants.JOB_PARAMETERS[job_kind]["container"],
                        )
                    )
            finished = [False for _ in log_streams]

            # Create thread and queue per stream, for non-blocking iteration
            log_queue_pool = utils.get_log_queue_pool(log_streams)

            # Iterate over every watching pods' log queue
            while True:
                for index, log_queue in enumerate(log_queue_pool):
                    if all(finished):
                        break
                    if finished[index]:
                        continue
                    # Grouping every 50 log lines of the same pod
                    for _ in range(50):
                        try:
                            logline = log_queue.get(timeout=1)
                            if logline is None:
                                finished[index] = True
                                break

                            # Print logs to StdOut
                            print(f"[Pod {pods[index].metadata.name}]: {logline}")
                            # Add logs to the results dict.
                            if pods[index].metadata.name not in logs_dict:
                                logs_dict[pods[index].metadata.name] = logline
                            else:
                                logs_dict[pods[index].metadata.name] += logline
                        except queue.Empty:
                            break
                if all(finished):
                    break
        elif pods:
            for pod in pods:
                if (
                    pod.status is not None
                    and pod.status.phase != constants.POD_PHASE_PENDING
                ):
                    try:
                        pod_logs = self.core_api.read_namespaced_pod_log(
                            name=pod.metadata.name,
                            namespace=namespace,
                            container=constants.JOB_PARAMETERS[job_kind]["container"],
                        )
                        logs_dict[pod.metadata.name] = pod_logs
                    except Exception:
                        raise RuntimeError(
                            f"Failed to read logs for pod {namespace}/{pod.metadata.name}"
                        )
        # If verbose is set, return Kubernetes events for Job and pods.
        if verbose:
            job = self.get_job(name=name, namespace=namespace)
            events = self.core_api.list_namespaced_event(namespace=namespace)

            # Get events for the Job and Job's pods.
            for event in events.items:
                utils.add_event_to_dict(
                    events_dict=events_dict,
                    event=event,
                    object_kind=job_kind,
                    object_name=name,
                    object_creation_timestamp=job.metadata.creation_timestamp,
                )
                if pods:
                    for pod in pods:
                        utils.add_event_to_dict(
                            events_dict=events_dict,
                            event=event,
                            object_kind=constants.POD_KIND,
                            object_name=pod.metadata.name,
                            object_creation_timestamp=pod.metadata.creation_timestamp,
                        )

        return logs_dict, events_dict

    def update_job(
        self,
        job: constants.JOB_MODELS_TYPE,
        name: str,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
    ):
        """
        Update the Training Job using the patch Kubernetes API.

        :param object job: Job object to be updated. For example, an object with type
            `KubeflowOrgV1TFJob` or `KubeflowOrgV1PyTorchJob`.
        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind
            is taken from the `TrainingClient` object.

        :raises TimeoutError: If the request times out while updating the job.
        :raises RuntimeError: If updating the job fails.
        """


        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        if job_kind not in constants.JOB_PARAMETERS:
            raise ValueError(
                f"Job kind must be one of these: {constants.JOB_PARAMETERS.keys()}"
            )

        try:
            self.custom_api.patch_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                namespace,
                constants.JOB_PARAMETERS[job_kind]["plural"],
                name,
                job,
            )
        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Timeout to update {job_kind}: {namespace}/{name}")
        except Exception:
            raise RuntimeError(f"Failed to update {job_kind}: {namespace}/{name}")

        logger.debug(f"{job_kind} {namespace}/{name} has been updated")

    def delete_job(
        self,
        name: str,
        namespace: Optional[str] = None,
        job_kind: Optional[str] = None,
        delete_options: Optional[models.V1DeleteOptions] = None,
    ):
        """
        Delete the Training Job.

        :param str name: Name of the job.
        :param str namespace: Namespace for the job. By default, the namespace is taken from the
            `TrainingClient` object.
        :param str job_kind: Kind of the job (e.g., `TFJob` or `PyTorchJob`). By default, the job kind
            is taken from the `TrainingClient` object.
        :param V1DeleteOptions delete_options: Optional. `V1DeleteOptions` to set while deleting
            the job, such as grace period seconds.

        :raises TimeoutError: If the request times out while deleting the job.
        :raises RuntimeError: If deleting the job fails.
        """


        namespace = namespace or self.namespace
        job_kind = job_kind or self.job_kind

        try:
            self.custom_api.delete_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                namespace,
                constants.JOB_PARAMETERS[job_kind]["plural"],
                name=name,
                body=delete_options,
            )
        except multiprocessing.TimeoutError:
            raise TimeoutError(f"Timeout to delete {job_kind}: {namespace}/{name}")
        except Exception:
            raise RuntimeError(f"Failed to delete {job_kind}: {namespace}/{name}")

        logger.debug(f"{job_kind} {namespace}/{name} has been deleted")