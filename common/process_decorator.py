import logging
import threading
import time

from loguru import logger


class ProcessDecorator:
    @staticmethod
    def tag(action: str, raise_exception=True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                begin_time = time.time()
                current_thread = threading.current_thread()
                thread_name = current_thread.name
                try:
                    logger.info('线程-{}: 任务：{}，开始。'.format(thread_name, action))
                    result = func(*args, **kwargs)
                    logger.info('线程-{}: 任务：{}，结束。总耗时{:.2f}秒'.format(thread_name, action, (time.time() - begin_time)))
                    return result  # 调用成功，返回结果
                except Exception as e:
                    logger.info('线程-{}: 任务：{}，异常。总耗时{:.2f}秒。错误信息：{}'.format(thread_name, action, (time.time() - begin_time), {str(e)}))
                    logging.exception(e)
                    if raise_exception:
                        raise e

            return wrapper

        return decorator
