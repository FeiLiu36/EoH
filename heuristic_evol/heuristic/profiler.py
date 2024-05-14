from __future__ import annotations

import json
import os.path
from abc import ABC
from threading import Lock

import wandb
from torch.utils.tensorboard import SummaryWriter

from .code import Function


class ProfilerBase(ABC):

    def register_function(self, function: Function):
        raise NotImplementedError()

    def finish(self):
        pass


class TensorboardProfiler(ProfilerBase):
    _num_samples = 0

    def __init__(
            self,
            log_dir: str | None = None,
            *,
            initial_num_samples=0,
            resume_from_log_dir=False
    ):
        """
        Args:
            log_dir: folder path for tensorboard log files.
            resume_from_log_dir: read samples in log_dir and recover runs.
        """
        # args and keywords
        self.__class__._num_samples = initial_num_samples
        self._log_dir = log_dir
        self._samples_json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._samples_json_dir, exist_ok=True)

        # statistics
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = float('-inf')
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0

        # summary writer instance for Tensorboard
        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        # resume profiler
        if resume_from_log_dir:
            self._resume_profiler()

        # lock for multi-thread invoking self.register_function(...)
        self._register_function_lock = Lock()

    def register_function(self, function: Function):
        """Record an obtained function. This is a synchronized function.
        """
        try:
            self._register_function_lock.acquire()
            self.__class__._num_samples += 1
            self._record_and_verbose(function)
            self._write_tensorboard()
            self._write_json(function)
        finally:
            self._register_function_lock.release()

    def _resume_profiler(self):
        # read total sample nums from file
        log_path = self._samples_json_dir
        max_samples = 0
        for dir in os.listdir(log_path):
            order = int(dir.split('.')[0].split('_')[1])
            if order > max_samples:
                max_samples = order
        self.__class__._num_samples = max_samples

        # read max score from file
        max_score = float('-inf')
        for dir in os.listdir(log_path):
            json_file = os.path.join(log_path, dir)
            with open(json_file, 'r') as f:
                json_dict = json.load(f)
            if json_dict['score'] is not None and json_dict['score'] > max_score:
                max_score = json_dict['score']
        assert max_score != float('-inf')
        self._cur_best_program_score = max_score

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self.__class__._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self.__class__._num_samples
        )

    def _write_json(self, function: Function):
        if not self._log_dir:
            return

        sample_order = self.__class__._num_samples
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(function)
        score = function.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._samples_json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _record_and_verbose(self, function):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        # update best function
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self.__class__._num_samples

        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(self.__class__._num_samples)}')
        print(f'------------------------------------------------------')
        print(f'Current best score: {self._cur_best_program_score}')
        print(f'======================================================\n')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time


class WandBProfiler(ProfilerBase):
    _num_samples = 0

    def __init__(
            self,
            wandb_project_name: str,
            log_dir: str | None = None,
            *,
            initial_num_samples=0,
            resume_from_log_dir=False,
            **wandb_init_kwargs
    ):
        """
        Args:
            log_dir: folder path for tensorboard log files.
            resume_from_log_dir: read samples in log_dir and recover runs.
        """
        # args and keywords
        self.__class__._num_samples = initial_num_samples
        self._wandb_project_name = wandb_project_name
        self._log_dir = log_dir
        self._samples_json_dir = os.path.join(log_dir, 'samples')

        wandb.init(project=self._wandb_project_name, **wandb_init_kwargs)
        os.makedirs(self._samples_json_dir, exist_ok=True)

        # statistics
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = float('-inf')
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0

        # summary writer instance for Tensorboard
        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        # resume profiler
        if resume_from_log_dir:
            self._resume_profiler()

        # lock for multi-thread invoking self.register_function(...)
        self._register_function_lock = Lock()

    def register_function(self, function: Function):
        """Record an obtained function. This is a synchronized function.
        """
        try:
            self._register_function_lock.acquire()
            self.__class__._num_samples += 1
            self._record_and_verbose(function)
            self._write_wandb()
            self._write_json(function)
        finally:
            self._register_function_lock.release()

    def _resume_profiler(self):
        # read total sample nums from file
        log_path = self._samples_json_dir
        max_samples = 0
        for dir in os.listdir(log_path):
            order = int(dir.split('.')[0].split('_')[1])
            if order > max_samples:
                max_samples = order
        self.__class__._num_samples = max_samples

        # read max score from file
        max_score = float('-inf')
        for dir in os.listdir(log_path):
            json_file = os.path.join(log_path, dir)
            with open(json_file, 'r') as f:
                json_dict = json.load(f)
            if json_dict['score'] is not None and json_dict['score'] > max_score:
                max_score = json_dict['score']
        assert max_score != float('-inf')
        self._cur_best_program_score = max_score

    def _write_wandb(self):
        wandb.log(
            {
                'Best Score of Function': self._cur_best_program_score
            }
        )
        wandb.log(
            {
                'Valid Function Num': self._evaluate_success_program_num,
                'Invalid Function Num': self._evaluate_failed_program_num
            }
        )
        wandb.log(
            {
                'Total Sample Time': self._tot_sample_time,
                'Total Evaluate Time': self._tot_evaluate_time
            }
        )

    def _write_json(self, function: Function):
        if self._log_dir is None:
            return
        sample_order = self.__class__._num_samples
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(function)
        score = function.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score
        }
        path = os.path.join(self._samples_json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _record_and_verbose(self, function):
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        # update best function
        if score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = self.__class__._num_samples

        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(self.__class__._num_samples)}')
        print(f'------------------------------------------------------')
        print(f'Current best score: {self._cur_best_program_score}')
        print(f'======================================================\n')

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def finish(self):
        wandb.finish()
