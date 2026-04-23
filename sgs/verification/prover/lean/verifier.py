from typing import Tuple, List
from func_timeout import FunctionTimedOut, func_set_timeout


import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import subprocess
import multiprocessing as mp
import psutil
from pprint import pprint

from dotenv import load_dotenv

from sgs.verification.prover.lean.ast_parser import lean4_parser
from sgs.verification.prover.workers import ProcessScheduler
from sgs.verification.prover.utils import AttrDict


load_dotenv()

if "ELAN_HOME" in os.environ:
    HOME_DIR = os.path.expanduser(os.environ["ELAN_HOME"])
else:
    HOME_DIR = os.path.expanduser("~")
DEFAULT_LAKE_PATH = f"{HOME_DIR}/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "./mathlib4"
DEFAULT_TIMEOUT = 200
VERBOSE = False

SGS_REPL_PATH = os.environ.get("SGS_REPL_PATH")
SGS_MATHLIB_PATH = os.environ.get("SGS_MATHLIB_PATH")

if SGS_REPL_PATH is None:
    raise RuntimeError(
        "SGS_REPL_PATH environment variable must be set to the path of the "
        "Lean REPL binary (e.g. <repl>/.lake/build/bin/repl)."
    )
if SGS_MATHLIB_PATH is None:
    raise RuntimeError(
        "SGS_MATHLIB_PATH environment variable must be set to the path of the "
        "mathlib4 directory used by the REPL."
    )

# Resolve to absolute paths so they survive the subprocess `cwd=` switch below.
SGS_REPL_PATH = os.path.abspath(SGS_REPL_PATH)
SGS_MATHLIB_PATH = os.path.abspath(SGS_MATHLIB_PATH)


# -------------------------------------------------------------------
# Code adapted for verifier from STP
# -------------------------------------------------------------------

def split_snippet(code: str) -> Tuple[str, str]:
    """
    From https://github.com/project-numina/kimina-lean-server/blob/main/server/split.py

    Splits a code snippet into a header (imports) and body.

    - Header: all lines at the top that are 'import ...' or blank before the first non-import line.
      If any import starts with 'import Mathlib', include a single 'import Mathlib' at the top of the header.
      Other imports follow in their original order, without duplicates.
    - Body: the rest of the code starting from the first non-import/non-blank line.
    """
    lines = code.splitlines()

    # Separate header from body
    i = 0
    while i < len(lines) and (
        lines[i].strip() == "" or lines[i].strip().startswith("import ") or lines[i].strip().startswith("set_option ") or lines[i].strip().startswith("open ")
    ):
        i += 1
    header_lines = [x.strip() for x in lines[:i]]
    body = "\n".join(lines[i:])

    # Process imports in header
    import_lines = [line for line in header_lines if line.startswith("import ") or line.startswith("set_option ") or line.startswith("open ")]
    imports: list[str] = []
    seen: set[str] = set()
    has_mathlib = False
    for line in import_lines:
        if line.startswith("import Mathlib"):
            has_mathlib = True
        else:
            if line not in seen:
                seen.add(line)
                imports.append(line)

    # Build final header
    result_header: list[str] = []
    if has_mathlib:
        result_header.append("import Mathlib")
    result_header.extend(imports)

    header = "\n".join(result_header) + "\n\n"
    return header, body

@func_set_timeout(DEFAULT_TIMEOUT, allowOverride=True)
def terminate_repl(proc):
    if proc is None:
        return
    
    try:
        # Create a psutil Process instance for the main process
        parent = psutil.Process(proc.pid)
        
        # Retrieve all child processes recursively
        children = parent.children(recursive=True)
        
        # Terminate all child processes
        for child in children:
            child.terminate()
        
        # Terminate the main process
        parent.terminate()
        
        # Wait for all processes to terminate gracefully
        gone, alive = psutil.wait_procs([parent] + children, timeout=5)
        
        # Force kill any processes that are still alive after the timeout
        for p in alive:
            p.kill()
            
    except psutil.NoSuchProcess:
        # The process may have already terminated
        pass
    except Exception as e:
        # Optionally log the exception if needed
        # print(f"Error in terminating processes: {e}")
        pass


def get_result_from_repl(repl_result, code, start_time):
    result = {
        "sorries" : repl_result.get('sorries', []), 
        "tactics" : repl_result.get('tactics', []),
        "errors" : [m for m in repl_result.get('messages', []) if m['severity'] == 'error'],
        "warnings" : [m for m in repl_result.get('messages', []) if m['severity'] == 'warning'],
        "infos" : [m for m in repl_result.get('messages', []) if m['severity'] == 'info'],
        "verified_code" : code,
    }
    result['pass'] = not result['errors']
    result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
    #if result['complete']:
    #    ast_results = lean4_parser(code, repl_result['ast']) if 'ast' in repl_result and repl_result['ast'] else {}
    #    result['invokes'] = extract_invokes(ast_results)
    #    if __DEBUG__:
    #        result['ast'] = ast_results
    result['verify_time'] = time.time() - start_time
    return result

def read_from_repl(proc):
    ret = ''
    while True:
        line = proc.stdout.readline()
        if len(line.strip()) == 0:
            break
        ret += line
    return ret

@func_set_timeout(DEFAULT_TIMEOUT, allowOverride=True)
def query_repl(proc, message_str):
    proc.stdin.write(message_str)
    proc.stdin.flush()
    return read_from_repl(proc)

@func_set_timeout(DEFAULT_TIMEOUT + 10, allowOverride=True)
def _start_repl_process(lake_path, lean_workspace, header):
    proc = subprocess.Popen([lake_path, "exe", 'repl'], 
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,  # Capture stderr
                                    text=True, 
                                    cwd=lean_workspace,)
    cmd = json.dumps({"cmd": header, "allTactics": False, "ast": False, "tactics": False, "premises": False}, ensure_ascii=False) + '\r\n\r\n'
    query_repl(proc, cmd)
    return proc

def start_repl_process(lake_path, lean_workspace, header = None):
    # Retry if the process is not started
    for i in range(5):
        try:
            return _start_repl_process(lake_path, lean_workspace, header)
        except Exception as e:
            print(f"Error in starting Lean4 process: {e}")
            time.sleep(i + 1)
            continue
    raise Exception("Failed to start Lean4 process")

def verify_lean4_file_multiple(
    codes: List[str], 
    lake_path=DEFAULT_LAKE_PATH, 
    lean_workspace=DEFAULT_LEAN_WORKSPACE, 
    last_env=None, 
    verbose=False, 
    allTactics=False, 
    ast=False, 
    premises=False, 
    tactics=False
):

    command = dict(allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    
    results = []

    split_codes = [split_snippet(x) for x in codes]
    proofs = [x[1] for x in split_codes]
    headers = [x[0] for x in split_codes]

    try:
        proc = None
        last_header = None
        for code, header in zip(proofs, headers):
            if proc is None or header != last_header:
                terminate_repl(proc)
                proc = start_repl_process(lake_path, lean_workspace, header)
                last_header = header
            
            message_str = json.dumps(command | {'cmd': code, 'env': 0}, ensure_ascii=False) + '\r\n\r\n'
            try:
                start_time = time.time()
                output = query_repl(proc, message_str)
                repl_result = json.loads(output)
                result = get_result_from_repl(repl_result, code, start_time)
                results.append(result)
            except (Exception, FunctionTimedOut) as e:
                results.append({"system_messages": str(e), 'complete': False})
                result = {
                    "pass": False,
                    "complete": False,
                    "system_errors": traceback.format_exc(),
                    "system_messages": str(e),
                    "outputs":  None,
                    "outputs_stderr": None,
                    "message_str": message_str,
                    "lake_path": lake_path,
                    "lean_workspace": lean_workspace,
                    "verify_time": time.time() - start_time,
                }
                results.append(result)
                terminate_repl(proc)
                proc = None

        terminate_repl(proc)
    except (Exception, FunctionTimedOut) as e:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": str(e),
            "outputs":  None,
            "outputs_stderr": None,
            "message_str": message_str,
            "lake_path": lake_path,
            "lean_workspace": lean_workspace,
            "verify_time": time.time() - start_time,
        }

        #results += [{"system_messages": str(e)}] * (len(codes) - len(results))
        results.append(result)

    assert len(results) == len(codes), f"Results length mismatch: {len(results)} != {len(codes)}"
    return results


class Lean4ServerProcessMultiple(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get("timeout", 300)
        self.memory_limit = extra_args.get("memory_limit", -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)

    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000**3), self.memory_limit * (1000**3)),
            )
        while True:
            inputs = self.task_queue.get()
            if inputs is None:  # Terminate when receiving None
                break

            codes: List[str] = []
            request_ids: List[str] = []
            for _, request_id, task in inputs:
                assert isinstance(task, list), f"Task is not a list: {task}"
                codes.append(task)
                request_ids.append(request_id)

            results = verify_lean4_file_multiple(
                codes=codes,
            )

            assert len(results) == len(codes), f"Results length mismatch: {len(results)} != {len(codes)}"
            #result = verify_lean4_file(**task)
            #if len(result["system_messages"]) > 0:
            #    retry_start_time = time.time()
            #    while (
            #        "lean::exception: failed to create thread"
            #        in result["system_messages"]
            #        or "std::bad_alloc: std::bad_alloc" in result["system_messages"]
            #        or "Cannot allocate memory" in result["system_messages"]
            #    ) and time.time() - retry_start_time < self.timeout:
            #        time.sleep(0.1)
            #        result = verify_lean4_file(**task)
            with self.lock:
                for result, request_id in zip(results, request_ids):
                    self.request_statuses[request_id] = result
                self.last_output_time.value = time.time()
                self.complete_count.value += len(results)


class Lean4ServerSchedulerMultiple(ProcessScheduler):
    def __init__(
        self, 
        max_concurrent_requests=64, 
        timeout=300, 
        memory_limit=-1, 
        name="verifier",
        batch_size=8,
    ):
        super().__init__(batch_size=8, name=name)

        self.processes = [
            Lean4ServerProcessMultiple(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                ),
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f"Complete launching {len(self.processes)} LeanServerProcesses")

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()

    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(
                ["killall", "repl", f"--older-than={int(self.timeout) + 10}s"],
                capture_output=True,
            )

    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f"All {len(self.processes)} LeanServerProcesses stopped")



# ---------------------------------------------------------------
# Verifier code from deepseek prover
# --------------------------------------------------------------

def verify_lean4_file_with_memory_limit(
    proof, memory_limit, timeout, lean_workspace=DEFAULT_LEAN_WORKSPACE
):
    if memory_limit > 0:
        # Set memory limit in bytes (convert GB to bytes)
        memory_limit_bytes = memory_limit * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        try:
            return verify_lean4_file(
                proof, timeout=timeout, lean_workspace=lean_workspace, ast=False
            )
        except resource.error:
            return {"pass": False, "complete": False, "error": "Used too much memory"}
    else:
        return verify_lean4_file(proof, timeout=timeout, lean_workspace=lean_workspace)


def verify_lean4_file(
    code,
    lake_path=DEFAULT_LAKE_PATH,
    lean_workspace=DEFAULT_LEAN_WORKSPACE,
    last_env=None,
    verbose=False,
    timeout=300,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
):

    # check if "DEFAULT_LEAN_WORKSPACE" environ exists
    if "DEFAULT_LEAN_WORKSPACE" in os.environ:
        lean_workspace = os.environ["DEFAULT_LEAN_WORKSPACE"]
    else:
        lean_workspace = lean_workspace

    if "DEFAULT_LAKE_PATH" in os.environ:
        lake_path = os.environ["DEFAULT_LAKE_PATH"]
    else:
        lake_path = lake_path

    if VERBOSE:
        print(f"lean_workspace: {lean_workspace}")
        print(f"lake_path: {lake_path}")

    command = dict(
        cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises
    )
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ""
    outputs = None
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout,
            )


        result = json.loads(outputs.stdout)
        ast_results = (
            lean4_parser(code, result["ast"])
            if "ast" in result and result["ast"]
            else {}
        )
        result = {
            "sorries": result.get("sorries", []),
            "tactics": result.get("tactics", []),
            "errors": [
                m for m in result.get("messages", []) if m["severity"] == "error"
            ],
            "warnings": [
                m for m in result.get("messages", []) if m["severity"] == "warning"
            ],
            "infos": [m for m in result.get("messages", []) if m["severity"] == "info"],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
            "outputs": outputs.stdout,
            "outputs_stderr": outputs.stderr,
            "message_str": message_str,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in warning["data"]
                or "failed" in warning["data"]
                for warning in result["warnings"]
            )
        )
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages,
            "outputs": outputs.stdout if outputs is not None else None,
            "outputs_stderr": outputs.stderr if outputs is not None else None,
            "message_str": message_str,
            "lake_path": lake_path,
            "lean_workspace": lean_workspace,
        }
    result["verify_time"] = time.time() - start_time
    return result


def verify_lean4_file_kimina(
    code,
    lake_path=DEFAULT_LAKE_PATH,
    lean_workspace=DEFAULT_LEAN_WORKSPACE,
    last_env=None,
    verbose=False,
    timeout=300,
    allTactics=False,
    ast=False,
    premises=False,
    tactics=False,
):

    if VERBOSE:
        print(f"lean_workspace: {lean_workspace}")
        print(f"lake_path: {lake_path}")

    command = dict(
        cmd=code, allTactics=allTactics, ast=ast, tactics=tactics, premises=premises
    )
    if last_env is not None:
        command.update(env=last_env)
    message_str = json.dumps(command, ensure_ascii=False)
    if verbose:
        print(message_str)
    start_time = time.time()
    system_messages = ""
    outputs = None
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run(
                [lake_path, "env", SGS_REPL_PATH],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=SGS_MATHLIB_PATH,
                timeout=timeout,
            )


        result = json.loads(outputs.stdout)
        ast_results = (
            lean4_parser(code, result["ast"])
            if "ast" in result and result["ast"]
            else {}
        )
        result = {
            "sorries": result.get("sorries", []),
            "tactics": result.get("tactics", []),
            "errors": [
                m for m in result.get("messages", []) if m["severity"] == "error"
            ],
            "warnings": [
                m for m in result.get("messages", []) if m["severity"] == "warning"
            ],
            "infos": [m for m in result.get("messages", []) if m["severity"] == "info"],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
            "outputs": outputs.stdout,
            "outputs_stderr": outputs.stderr,
            "message_str": message_str,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in warning["data"]
                or "failed" in warning["data"]
                for warning in result["warnings"]
            )
        )
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages,
            "outputs": outputs.stdout if outputs is not None else None,
            "outputs_stderr": outputs.stderr if outputs is not None else None,
            "message_str": message_str,
            "lake_path": lake_path,
            "lean_workspace": lean_workspace,
        }
    result["verify_time"] = time.time() - start_time
    return result




VALID_LEAN_VERSIONS = ("4.9", "4.15")


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get("timeout", 300)
        self.memory_limit = extra_args.get("memory_limit", -1)
        self.lean_version = extra_args.get("lean_version", "4.15")
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)

        assert self.lean_version in VALID_LEAN_VERSIONS, f"Invalid lean_version: {self.lean_version}. Must be one of {VALID_LEAN_VERSIONS}"

    def _verify(self, **task):
        if self.lean_version == "4.15":
            return verify_lean4_file_kimina(**task)
        elif self.lean_version == "4.9":
            return verify_lean4_file(**task)

    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000**3), self.memory_limit * (1000**3)),
            )
        while True:
            inputs = self.task_queue.get()
            if inputs is None:  # Terminate when receiving None
                break
            for _, request_id, task in inputs:
                if isinstance(task, str):
                    task = dict(code=task)
                if "timeout" not in task:
                    task["timeout"] = self.timeout
                result = self._verify(**task)
                if len(result["system_messages"]) > 0:
                    retry_start_time = time.time()
                    while (
                        "lean::exception: failed to create thread"
                        in result["system_messages"]
                        or "std::bad_alloc: std::bad_alloc" in result["system_messages"]
                        or "Cannot allocate memory" in result["system_messages"]
                    ) and time.time() - retry_start_time < self.timeout:
                        time.sleep(0.1)
                        result = verify_lean4_file(**task)
                with self.lock:
                    self.request_statuses[request_id] = result
                    self.last_output_time.value = time.time()
                    self.complete_count.value += 1




class Lean4ServerScheduler(ProcessScheduler):
    def __init__(
        self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name="verifier", lean_version="4.15"
    ):
        super().__init__(batch_size=1, name=name)

        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                    lean_version=lean_version,
                ),
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f"Complete launching {len(self.processes)} LeanServerProcesses")

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()

    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(
                ["killall", "repl", f"--older-than={int(self.timeout) + 10}s"],
                capture_output=True,
            )

    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f"All {len(self.processes)} LeanServerProcesses stopped")


if __name__ == "__main__":
    code = open("mathlib4/.lake/packages/REPL/test/aime_1983_p9.code.in").read()
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=1, timeout=300, memory_limit=10, name="verifier"
    )
    request_id_list = lean4_scheduler.submit_all_request(
        [dict(code=code, ast=True, tactics=True)]
    )
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    pprint(outputs_list)
