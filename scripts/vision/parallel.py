import argparse
import multiprocessing as mp
import os
import queue
import traceback


def gpu_list(value):
    try:
        ids = [int(x.strip()) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated GPU indices, e.g. 0,1,2') from exc
    if not ids or min(ids) < 0 or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError('GPU indices must be nonnegative and unique')
    return ids


def _entry(worker, args, jobs, gpu, events):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='4', HF_HUB_DISABLE_PROGRESS_BARS='1')
    try:
        worker(args, jobs, lambda row: events.put(('result', gpu, row)))
        events.put(('done', gpu, None))
    except BaseException:
        events.put(('error', gpu, traceback.format_exc()))


def prediction_progress(args, total, initial=0, gpu=None, position=0):
    from tqdm import tqdm
    from scripts.vision.consensus import VOTERS
    methods = list(getattr(args, 'progress_methods', None) or args.methods or VOTERS)
    if args.consensus and not getattr(args, 'progress_methods', None):
        methods.append('scantrust_v2')
    label = ','.join(methods)
    if gpu is not None:
        label = f'GPU {gpu} | {label}'
    return tqdm(total=total, initial=initial, desc=label, unit='q',
                position=position, dynamic_ncols=True, mininterval=1,
                disable=not getattr(args, 'progress', True),
                bar_format='{desc}: {n_fmt}/{total_fmt} ({percentage:.1f}%) '
                           '|{bar}| [{elapsed}<{remaining}, {rate_fmt}]')


def stream(worker, args, jobs, gpus):
    shards = [[] for _ in gpus]
    loads = [0] * len(gpus)
    for job in sorted(jobs, key=lambda j: -j.get('weight', 1)):
        owner = min(range(len(gpus)), key=loads.__getitem__)
        shards[owner].append(job)
        loads[owner] += job.get('weight', 1)
    ctx = mp.get_context('spawn')
    events = ctx.Queue(maxsize=32)
    workers = {}
    progress = {}
    done = set()
    try:
        for gpu, shard in zip(gpus, shards):
            if not shard:
                continue
            process = ctx.Process(target=_entry, args=(worker, args, shard, gpu, events))
            process.start()
            workers[gpu] = process
            if worker is prediction_worker:
                progress[gpu] = prediction_progress(args, sum(len(j['ids']) for j in shard),
                                                     gpu=gpu, position=len(progress))
        while len(done) < len(workers):
            try:
                kind, gpu, value = events.get(timeout=1)
            except queue.Empty:
                dead = [gpu for gpu, p in workers.items() if gpu not in done and p.exitcode is not None]
                if dead:
                    raise RuntimeError(f'GPU workers exited without completing: {dead}')
                continue
            if kind == 'error':
                raise RuntimeError(f'GPU {gpu} worker failed:\n{value}')
            if kind == 'done':
                done.add(gpu)
            else:
                if gpu in progress:
                    progress[gpu].update()
                yield value
    finally:
        for gpu in done:
            workers[gpu].join(timeout=5)
        for process in workers.values():
            if process.is_alive():
                process.terminate()
        for process in workers.values():
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        events.close()
        for bar in progress.values():
            bar.close()


def prediction_worker(args, jobs, emit):
    from scripts.vision.run import predict
    args.device = 'cuda:0'
    args.progress = False
    args.only_ids = {identity for job in jobs for identity in job['ids']}
    predict(args, sink=emit)


def open_images(job):
    from PIL import Image
    from scripts.vision.run import sha256
    images = []
    for path, digest in zip(job['images'], job['image_sha256s']):
        if sha256(path) != digest:
            raise ValueError(f'Image hash mismatch: {path}')
        with Image.open(path) as im:
            images.append(im.convert('RGB'))
    return images[0] if len(images) == 1 else images


def caption_worker(args, jobs, emit):
    import hashlib
    import time
    from scripts.vision.inference import Engine
    from scripts.vision.run import compact
    engine = Engine.captioner(str(args.sender), 'cuda:0')
    batch_size = getattr(args, 'batch_size', 1) or 1
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        images = [open_images(job) for job in chunk]
        started = time.monotonic()
        results = engine.describe_batch(images)
        seconds = (time.monotonic() - started) / len(chunk)
        for job, result in zip(chunk, results):
            result = compact(result)
            if not result['output'].strip():
                raise ValueError(f'Empty caption: {job["key"]}')
            emit((job['key'], {**result, 'image_sha256': job['key'],
                'image_sha256s': job['image_sha256s'], 'image_paths': job['images'],
                'caption_sha256': hashlib.sha256(result['output'].encode()).hexdigest(),
                'seconds': seconds}))


def baseline_results(engine, jobs, batch_size=1):
    from scripts.vision.run import compact, GEMMA_BASELINES, CAPTION_BASELINES, OCR_BASELINES
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    pending = {}

    def flush(method):
        rows = pending.pop(method)
        answers = engine.baseline_batch([item for _, item in rows], method)
        if len(answers) != len(rows):
            raise ValueError('Batch result count differs from input count')
        for (identity, _), result in zip(rows, answers):
            yield identity, method, compact(result)

    for job in jobs:
        image = open_images(job) if any(set(q['methods']) & set(GEMMA_BASELINES) for q in job['questions']) else None
        for question in job['questions']:
            for method in question['methods']:
                item = dict(image=image if method in GEMMA_BASELINES else None,
                    question=question['question'], caption=job.get('caption') if method in CAPTION_BASELINES else None,
                    **({'ocr': job['ocr']} if method in OCR_BASELINES else {}),
                    **({'reasoning_budget': job['reasoning_budget']} if method == 'gemma_cot' else {}))
                if batch_size > 1 and method in GEMMA_BASELINES:
                    pending.setdefault(method, []).append((question['id'], item))
                    if len(pending[method]) == batch_size:
                        yield from flush(method)
                else:
                    yield question['id'], method, compact(engine.baseline_answer(method=method, **item))
    for method in list(pending):
        yield from flush(method)


def baseline_worker(args, jobs, emit):
    from scripts.vision.inference import Engine
    methods = {m for job in jobs for q in job['questions'] for m in q['methods']}
    engine = Engine.baseliner(str(args.sender), str(args.receiver), methods, 'cuda:0')
    for record in baseline_results(engine, jobs, getattr(args, 'batch_size', 1)):
        emit(record)
