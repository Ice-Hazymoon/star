import { expect, test } from "bun:test";
import {
  createJobLimiter,
  JobQueueAbortedError,
  JobQueueFullError,
} from "./job-limiter";

function createDeferred() {
  let resolvePromise: (() => void) | null = null;
  const promise = new Promise<void>((resolve) => {
    resolvePromise = () => resolve();
  });
  return {
    promise,
    resolve() {
      if (resolvePromise) {
        resolvePromise();
      }
    },
  };
}

test("job limiter queues work until a slot is released", async () => {
  const limiter = createJobLimiter(1, 4);
  const firstGate = createDeferred();
  const events: string[] = [];

  const first = limiter.run(async () => {
    events.push("first-start");
    await firstGate.promise;
    events.push("first-end");
  });

  const second = limiter.run(async () => {
    events.push("second-start");
    events.push("second-end");
  });

  await Promise.resolve();
  expect(limiter.stats()).toEqual({ activeJobs: 1, queuedJobs: 1 });
  firstGate.resolve();

  await Promise.all([first, second]);
  expect(events).toEqual([
    "first-start",
    "first-end",
    "second-start",
    "second-end",
  ]);
  expect(limiter.stats()).toEqual({ activeJobs: 0, queuedJobs: 0 });
});

test("job limiter rejects when the queue is full", async () => {
  const limiter = createJobLimiter(1, 1);
  const firstGate = createDeferred();

  const first = limiter.run(async () => {
    await firstGate.promise;
  });

  const second = limiter.run(async () => undefined);
  await expect(limiter.run(async () => undefined)).rejects.toBeInstanceOf(JobQueueFullError);

  firstGate.resolve();
  await Promise.all([first, second]);
});

test("job limiter removes aborted jobs from the queue", async () => {
  const limiter = createJobLimiter(1, 2);
  const firstGate = createDeferred();

  const first = limiter.run(async () => {
    await firstGate.promise;
  });

  const abortController = new AbortController();
  const queuedJob = limiter.run(async () => undefined, abortController.signal);
  expect(limiter.stats()).toEqual({ activeJobs: 1, queuedJobs: 1 });

  abortController.abort();
  await expect(queuedJob).rejects.toBeInstanceOf(JobQueueAbortedError);
  expect(limiter.stats()).toEqual({ activeJobs: 1, queuedJobs: 0 });

  firstGate.resolve();
  await first;
});
