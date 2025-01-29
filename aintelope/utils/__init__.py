# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import time
import datetime

from progressbar import ProgressBar

if os.name == "nt":
    from semaphore_win_ctypes import (
        AcquireSemaphore,
        CreateSemaphore,
        OpenSemaphore,
        Semaphore,
    )
else:
    import posix_ipc

# this one is cross-platform
from filelock import FileLock


def wait_for_enter(message=None):
    if os.name == "nt":
        import msvcrt

        if message is not None:
            print(message)
        msvcrt.getch()  # Uses less CPU on Windows than input() function. This becomes perceptible when multiple console windows with Python are waiting for input. Note that the graph window will be frozen, but will still show graphs.
    else:
        if message is None:
            message = ""
        input(message)


# TODO: function to read CSV inside lock


def try_df_to_csv_write(df, filepath, **kwargs):
    while (
        True
    ):  # TODO: refactor this loop to a shared helper function. recording.py uses a same pattern
        try:
            with FileLock(
                str(filepath)
                + ".lock"  # filepath may be PosixPath, so need to convert to str
            ):  # NB! take the lock inside the loop, not outside, so that when we are waiting for user confirmation for retry, we do not block other processes during that wait
                df.to_csv(filepath, **kwargs)
            return
        except PermissionError:
            print(
                f"Cannot write to file {filepath} Is the file open by Excel or some other program?"
            )
            wait_for_enter("\nPress [enter] to retry.")


class RobustProgressBar(ProgressBar):
    def __init__(self, *args, initial_value=0, disable=False, granularity=1, **kwargs):
        self.disable = disable
        self.granularity = granularity
        self.prev_value = initial_value
        super(RobustProgressBar, self).__init__(
            *args, initial_value=initial_value, **kwargs
        )

    def __enter__(self):
        if not self.disable:
            try:
                super(RobustProgressBar, self).__enter__()
            except Exception:  # TODO: catch only console write related exceptions
                pass
        return self

    def __exit__(self, type, value, traceback):
        if not self.disable:
            try:
                super(RobustProgressBar, self).__exit__(type, value, traceback)
            except Exception:  # TODO: catch only console write related exceptions
                pass
        return

    def update(self, value=None, *args, force=False, **kwargs):
        if not self.disable:
            try:
                if force or (
                    value is not None and value - self.prev_value >= self.granularity
                ):  # avoid too frequent console updates which would slow down the computation
                    if value is not None:
                        self.prev_value = value
                    super(RobustProgressBar, self).update(
                        value, *args, force=force, **kwargs
                    )
            except Exception:  # TODO: catch only console write related exceptions
                pass
        return

    # def _blackHoleMethod(*args, **kwargs):
    #    return

    # def __getattr__(self, attr):
    #    if not self.disable:
    #        return super(RobustProgressBar, self).__getattr__(attr)
    #    else:
    #        return self._blackHoleMethod


# / class RobustProgressBar(ProgressBar):


def get_now_str():
    now_str = datetime.datetime.strftime(datetime.datetime.now(), "%m.%d %H:%M:%S")
    return now_str


# / def get_now_str():


# https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
class Timer(object):
    def __init__(self, name=None, quiet=False):
        self.name = name
        self.quiet = quiet

    def __enter__(self):
        if not self.quiet and self.name:
            print(get_now_str() + " : " + self.name + "...")

        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        elapsed = time.time() - self.tstart

        if not self.quiet:
            if self.name:
                print(
                    get_now_str() + " : " + self.name + " totaltime: {}".format(elapsed)
                )
            else:
                print(get_now_str() + " : " + "totaltime: {}".format(elapsed))
        # / if not quiet:


# / class Timer(object):


# There does not seem to be a cross platform semaphore class available, so lets create one by combining platform specific semaphores.
# Note that there is a cross-platform lock class available in filelock package. This would be equivalent to special case of Semaphore with max_count=1.
class Semaphore(object):
    def __init__(self, name, max_count, *args, disable=False, **kwargs):
        self.name = name
        self.max_count = max_count
        self.disable = disable
        self.win_semaphore = None
        self.win_acquired_semaphore = None
        self.posix_semaphore = None
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        if not self.disable:
            if os.name == "nt":
                self.win_semaphore = CreateSemaphore(
                    self.name, *self.args, maximum_count=self.max_count, **self.kwargs
                )
                self.win_semaphore.__enter__()
                self.win_acquired_semaphore = AcquireSemaphore(self.win_semaphore)
                self.win_acquired_semaphore.__enter__()
            else:
                self.posix_semaphore = posix_ipc.Semaphore(
                    self.name,
                    *self.args,
                    flags=posix_ipc.O_CREAT,
                    initial_value=self.max_count,
                    **self.kwargs,
                )
                self.posix_semaphore.__enter__()

        return self

    def __exit__(self, type, value, traceback):
        if not self.disable:
            if os.name == "nt":
                self.win_acquired_semaphore.__exit__(type, value, traceback)
                self.win_acquired_semaphore = None
                self.win_semaphore.__exit__(type, value, traceback)
                self.win_semaphore = None
            else:
                self.posix_semaphore.__exit__(type, value, traceback)
                self.posix_semaphore = None
        return


# / class Semaphore(object):
