import psutil
import os
class RayOutOfMemoryError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

    @staticmethod
    def get_message(used_gb, total_gb, threshold):
        pids = psutil.pids()
        proc_stats = []
        for pid in pids:
            proc = psutil.Process(pid)
            proc_stats.append((proc.memory_info().rss, pid, proc.cmdline()))
        proc_str = "PID\tMEM\tCOMMAND"
        for rss, pid, cmdline in sorted(proc_stats, reverse=True)[:100]:
            proc_str += "\n{}\t{}GB\t{}".format(
                pid, round(rss / 1e9, 2), " ".join(cmdline)[:100].strip())
        return ("More than {}% of the memory on ".format(int(
            100 * threshold)) + "node {} is used ({} / {} GB). ".format(
                os.uname()[1], round(used_gb, 2), round(total_gb, 2)) +
                "The top 5 memory consumers are:\n\n{}".format(proc_str) +
                "\n\nIn addition, ~{} GB of shared memory is ".format(
                    round(psutil.virtual_memory().shared / 1e9, 2)) +
                "currently being used by the Ray object store. You can set "
                "the object store size with the `object_store_memory` "
                "parameter when starting Ray, and the max Redis size with "
                "`redis_max_memory`.")

print(RayOutOfMemoryError.get_message(15, 15, 15))
