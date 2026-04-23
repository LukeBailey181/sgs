from typing import Optional
from sgs.utils import get_submitit_executor
import torch
import multiprocessing as mp

from sgs.utils.server import TaskServer
from sgs.models.query_local import QueryWorkerToServerTask


class QueryServer(TaskServer):
    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        if self._tasks.qsize() == 0 and len(self._in_progress) == 0:
            raise ValueError(
                "There is no work to be done and you are trying to launch jobs."
            )

        resources_config = self.worker_resources_config
        if not resources_config.submitit:
            return

        executor = get_submitit_executor(resources_config)

        if num_workers is None:
            num_workers = resources_config.num_jobs

        assert self.model_config is not None, "Model config must be set for QueryServer"
        for _ in range(num_workers):
            worker = QueryWorkerToServerTask(
                self.server_address, monitor=self.in_depth_monitoring
            )
            job = executor.submit(
                worker,
                model_config=self.model_config,
                buffer_size=self.model_config.gen_batch_size,
            )
            self.workers.append(job)

    def launch_master_worker(self) -> None:
        """
        Launches as many workers on master machine as there are GPUs available
        """

        assert self.model_config is not None, "Model config must be set for QueryServer"

        num_gpus = torch.cuda.device_count()

        if num_gpus <= 0:
            print("No GPUs detected on master; skipping local query workers.")
            return

        ctx = mp.get_context("spawn")
        for i in range(num_gpus):
            worker = QueryWorkerToServerTask(
                self.server_address, monitor=self.in_depth_monitoring
            )
            job = ctx.Process(
                target=worker.__call__,
                args=(
                    self.model_config,
                    self.model_config.gen_batch_size,
                    i,
                ),
                name=f"query_worker_{i}",
            )

            job.start()

            self.master_worker_processes.append(job)  # type: ignore


class QueryServerSubprocesses(TaskServer):
    # This is a query server that launches workers as subprocesses

    def launch_workers(self, num_workers: Optional[int] = None) -> None:
        if self._tasks.qsize() == 0 and len(self._in_progress) == 0:
            raise ValueError(
                "There is no work to be done and you are trying to launch jobs."
            )

        # Will have to think about what is the right thing to do here

        # resources_config = self.worker_resources_config

        # executor = get_submitit_executor(resources_config)

    def kill_workers(self) -> None:
        pass
