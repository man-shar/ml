from threading import current_thread
from colorama import Fore, Style
import logging

logging.basicConfig(level=logging.INFO)


log_colors = {
    "r": Fore.RED,
    "g": Fore.GREEN,
    "b": Fore.BLUE,
    "y": Fore.YELLOW,
}

log_styles = {
    "br": Style.BRIGHT,
    "d": Style.DIM,
}


def indent_level(n=1):
    return " " * 2 * n


def get_msg_type(logger: logging):
    return {
        "i": logger.info,
        "w": logger.warning,
        "e": logger.error,
    }


def log(msg="", color_code="", style_code="", level="i", indent=0, logger=logging):
    color = log_colors.get(color_code, "")
    style = log_styles.get(style_code, "")
    msg_type = get_msg_type(logger)

    msg_type[level](f"{indent_level(indent)}{color}{style}{msg}{Style.RESET_ALL}")


def get_thread_logger(
    logger_name="",
    force_main_logger=False,
    file_handler=True,
    filename=None,
    propagate=True,
    format="%(name)s - %(levelname)s - %(message)s",
):
    if force_main_logger:
        return log, {}

    # create a logger for this thread/worker
    nm = current_thread().name
    log_file = None

    if "ThreadPoolExecutor" in nm or "ProcessPoolExecutor" in nm:
        # ThreadPoolExecutor-%d_%d
        nm = nm.split("-")[1]
        nm = nm.split("_")
        nm = int(nm[0]) + int(nm[1])
        import logging

        # define a logger for planner executor agent with a filehandler
        logger = logging.getLogger(f"{logger_name}")
        logger.propagate = propagate
        logger.setLevel(logging.INFO)

        if file_handler:
            if not logger.hasHandlers():
                if filename:
                    log_file = filename
                else:
                    log_file = f"./logs/{logger_name}.log"

                fh = logging.FileHandler(log_file)
                fh.setLevel(logging.INFO)
                fh.setFormatter(logging.Formatter(format))
                logger.addHandler(fh)
            else:
                # get handler and store as log_file
                log_file = logger.handlers[0].baseFilename

        # redefine log function as a wrapper to the imported one just with logger=logger
        thread_logger = lambda *args, **kwargs: log(*args, **kwargs, logger=logger)
    else:
        thread_logger = log

    return thread_logger, {"log_file": log_file, "logger": logger}


def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    c = 0
    for i in range(0, len(l), group_size):
        yield c, l[i : i + group_size]
        c += 1
