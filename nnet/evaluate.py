# -*- coding: utf-8 -*-
"""
@author: PengChuan
"""

from multiprocessing import Process, Queue
import assess.core
import global_config

class Evaler:
    def __init__(self, dataset, runner, post_runner=None):
        self.generator = dataset
        self.runner = runner
        self.post = post_runner

    def run_eval(self, n_proc=-1):
        '''
        :param n_proc:多进程
        :return: 均值stoi, 均值pesq,均值ssnr
        '''
        if n_proc <= 0: # 进程
            n_proc = global_config.NUM_PROCESS

        n_workers = n_proc - 1
        q_score = Queue()
        if n_workers <= 0:
            for sample in self.generator.asyn_gen():
                sample.enhed = self.runner.run(sample)
                self.eval(sample)
                q_score.put(sample)
            q_score.put(None)
        else:
            q_task = Queue()
            for sample in self.generator.asyn_gen():
                sample.enhed = self.runner.run(sample)
                q_task.put(sample)
            for _ in range(n_workers):
                p_eval = Process(target=_eval, args=(self, q_task, q_score))
                p_eval.start()
                q_task.put(None)

        mean_stoi, mean_pesq, mean_ssnr = 0., 0., 0.
        n_finished, count = 0, 0
        while True:
            sample_score = q_score.get()
            if sample_score is None:
                n_finished += 1
                if n_finished >= n_workers:
                    break
                continue
            count += 1
            mean_stoi += sample_score.stoi
            mean_pesq += sample_score.pesq
            mean_ssnr += sample_score.ssnr
            if self.post is not None:
                self.post.post_run(sample_score)
        return mean_stoi / count, mean_pesq / count, mean_ssnr / count

    def eval(self, sample):
        samplerate = self.generator.conf['samplerate']
        windowsize = self.generator.conf['windowsize']
        self.generator.gen_raw(sample)
        sample.stoi = assess.core.calc_stoi(sample.clean_raw, sample.enhed_raw, samplerate)
        sample.pesq = assess.core.calc_pesq(sample.clean_raw, sample.enhed_raw, samplerate, False)
        sample.ssnr = assess.core.calc_ssnr(sample.clean_raw, sample.enhed_raw, windowsize)


def _eval(evaler, q_task, q_score):
    while True:
        # print("eval: %d" % q_task.qsize())
        sample = q_task.get()
        if sample is None:
            q_score.put(None)
            break
        else:
            evaler.eval(sample)
            q_score.put(sample)
