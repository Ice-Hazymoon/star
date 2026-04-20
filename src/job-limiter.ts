export class JobQueueFullError extends Error {
  constructor(message = "job queue is full") {
    super(message);
    this.name = "JobQueueFullError";
  }
}

export class JobQueueAbortedError extends Error {
  constructor(message = "job queue wait aborted") {
    super(message);
    this.name = "JobQueueAbortedError";
  }
}

type QueueEntry = {
  resolve: () => void;
  reject: (reason?: unknown) => void;
  cleanup: () => void;
};

export type JobLimiterStats = {
  activeJobs: number;
  queuedJobs: number;
};

export function createJobLimiter(concurrency: number, maxQueuedJobs: number) {
  let activeJobs = 0;
  const queuedEntries: QueueEntry[] = [];

  function stats(): JobLimiterStats {
    return {
      activeJobs,
      queuedJobs: queuedEntries.length,
    };
  }

  function dispatchNext() {
    while (activeJobs < concurrency && queuedEntries.length > 0) {
      const entry = queuedEntries.shift();
      if (!entry) {
        return;
      }
      activeJobs += 1;
      entry.cleanup();
      entry.resolve();
    }
  }

  async function acquire(signal?: AbortSignal) {
    if (signal?.aborted) {
      throw new JobQueueAbortedError();
    }

    if (activeJobs < concurrency) {
      activeJobs += 1;
      return;
    }

    if (queuedEntries.length >= maxQueuedJobs) {
      throw new JobQueueFullError();
    }

    await new Promise<void>((resolve, reject) => {
      let queueEntry: QueueEntry | null = null;

      const cleanup = () => {
        if (signal) {
          signal.removeEventListener("abort", onAbort);
        }
      };

      const onAbort = () => {
        if (queueEntry) {
          const index = queuedEntries.indexOf(queueEntry);
          if (index >= 0) {
            queuedEntries.splice(index, 1);
          }
        }
        cleanup();
        reject(new JobQueueAbortedError());
      };

      queueEntry = {
        resolve,
        reject,
        cleanup,
      };

      queuedEntries.push(queueEntry);
      if (signal) {
        signal.addEventListener("abort", onAbort, { once: true });
      }
    });
  }

  function release() {
    if (activeJobs <= 0) {
      return;
    }
    activeJobs -= 1;
    dispatchNext();
  }

  async function run<T>(job: () => Promise<T>, signal?: AbortSignal) {
    await acquire(signal);
    try {
      return await job();
    } finally {
      release();
    }
  }

  return {
    run,
    stats,
  };
}
