
import asyncio
from json import dumps
import logging
import sys
from typing import Any
import warnings

import dask
from distributed.deploy.ssh import Worker, Scheduler, old_cluster_kwargs
from distributed.deploy.spec import SpecCluster

logger = logging.getLogger(__name__)

class UnreliableWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def start(self):
        try:
            await asyncio.wait_for(self._start(), timeout=30)
        except Exception as ex:
            raise

    async def _start(self):
        try:
            import asyncssh  # import now to avoid adding to module startup time
        except ImportError:
            raise ImportError(
                "Dask's SSHCluster requires the `asyncssh` package to be installed. "
                "Please install it using pip or conda."
            )

        self.connection = await asyncssh.connect(self.address, **self.connect_options)

        result = await self.connection.run("uname")
        if result.exit_status == 0:
            set_env = 'env DASK_INTERNAL_INHERIT_CONFIG="{}"'.format(
                dask.config.serialize(dask.config.global_config)
            )
        else:
            result = await self.connection.run("cmd /c ver")
            if result.exit_status == 0:
                set_env = "set DASK_INTERNAL_INHERIT_CONFIG={} &&".format(
                    dask.config.serialize(dask.config.global_config)
                )
            else:
                raise Exception(
                    "Worker failed to set DASK_INTERNAL_INHERIT_CONFIG variable "
                )

        if not self.remote_python:
            self.remote_python = sys.executable

        cmd = " ".join(
            [
                set_env,
                self.remote_python,
                "-m",
                "distributed.cli.dask_spec",
                self.scheduler,
                "--spec",
                "'%s'"
                % dumps(
                    {
                        i: {
                            "cls": self.worker_class,
                            "opts": {
                                **self.kwargs,
                            },
                        }
                        for i in range(self.n_workers)
                    }
                ),
            ]
        )

        self.proc = await self.connection.create_process(cmd)

        # We watch stderr in order to get the address, then we return
        started_workers = 0
        while started_workers < self.n_workers:
            line = await self.proc.stderr.readline()
            if not line.strip():
                stdout = await self.proc.stdout.read()
                raise Exception(f"Worker at {self.address}, with username: {self.connect_options['username']} failed to start: {stdout}")
            logger.info(line.strip())
            if "worker at" in line:
                started_workers += 1
        logger.debug("%s", line)
        await super().start()

def UnreliableSSHCluster(
    hosts: list[str] | None = None,
    connect_options: dict | list[dict] | None = None,
    worker_options: dict | None = None,
    scheduler_options: dict | None = None,
    worker_module: str = "deprecated",
    worker_class: str = "distributed.Nanny",
    remote_python: str | list[str] | None = None,
    **kwargs: Any,
) -> SpecCluster:
    
    connect_options = connect_options or {}
    worker_options = worker_options or {}
    scheduler_options = scheduler_options or {}

    if worker_module != "deprecated":
        raise ValueError(
            "worker_module has been deprecated in favor of worker_class. "
            "Please specify a Python class rather than a CLI module."
        )

    if set(kwargs) & old_cluster_kwargs:
        from distributed.deploy.old_ssh import SSHCluster as OldSSHCluster

        warnings.warn(
            "Note that the SSHCluster API has been replaced.  "
            "We're routing you to the older implementation.  "
            "This will be removed in the future"
        )
        kwargs.setdefault("worker_addrs", hosts)
        return OldSSHCluster(**kwargs)  # type: ignore

    if not hosts:
        raise ValueError(
            f"`hosts` must be a non empty list, value {repr(hosts)!r} found."
        )
    if isinstance(connect_options, list) and len(connect_options) != len(hosts):
        raise RuntimeError(
            "When specifying a list of connect_options you must provide a "
            "dictionary for each address."
        )

    if isinstance(remote_python, list) and len(remote_python) != len(hosts):
        raise RuntimeError(
            "When specifying a list of remote_python you must provide a "
            "path for each address."
        )

    scheduler = {
        "cls": Scheduler,
        "options": {
            "address": hosts[0],
            "connect_options": connect_options
            if isinstance(connect_options, dict)
            else connect_options[0],
            "kwargs": scheduler_options,
            "remote_python": remote_python[0]
            if isinstance(remote_python, list)
            else remote_python,
        },
    }
    workers = {
        i: {
            "cls": UnreliableWorker,
            "options": {
                "address": host,
                "connect_options": connect_options
                if isinstance(connect_options, dict)
                else connect_options[i + 1],
                "kwargs": worker_options,
                "worker_class": worker_class,
                "remote_python": remote_python[i + 1]
                if isinstance(remote_python, list)
                else remote_python,
            },
        }
        for i, host in enumerate(hosts[1:])
    }
    return SpecCluster(workers, scheduler, name="SSHCluster", **kwargs)