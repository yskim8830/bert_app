from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import multiprocessing as mp
  
WORK_LIST = [int(1e5), int(1e6), int(1e7), int(1e8)]

# 누적 합 계산 함수
def sum_generator(n):
    sum_val = sum(x for x in range(1, n+1))
    return sum_val


def main():
    worker = min(10, len(WORK_LIST))

    st = time.time()

    future_lst = []
    with ProcessPoolExecutor(max_workers=worker) as executor:
        for work in WORK_LIST:
            future = executor.submit(sum_generator, work)
            future_lst.append(future)
        
        # as_completed
        for future in as_completed(future_lst):
            # 가장 적게 걸리는 작업물부터 수행
            result = future.result()
            # 수행 완료된 작업물
            done = future.done()
            # 타임아웃을 벗어나 수행되지 않은 작업물
            cancelled = future.cancelled()
            print('Result: {}, Done: {}'.format(result, done))
            print('Cancelled: {}'.format(cancelled))



if __name__ == '__main__':
    print(mp.cpu_count())
    main()