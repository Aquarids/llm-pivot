import logging
import traceback

ENV_LOCAL = "local"
ENV_DEV = "dev"
ENV_PROD = "prod"

class Logger:

    def __init__(self, name, env, enabled=True, level=None):
        self.env = env
        self.enabled = enabled
        
        if not enabled:
            self.logger = logging.getLogger(name)
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            self.logger = self._init_logging(env, name)
            if level:
                self.set_level(getattr(logging, level.upper()))

    def _init_logging(self, env, name):
        if env in [ENV_LOCAL, ENV_DEV]:
            level = logging.DEBUG
            handlers = [logging.StreamHandler()]
        elif env == ENV_PROD:
            level = logging.INFO
            handlers = [
                logging.FileHandler("lpivot.log", encoding="utf-8"),
            ]

        logging.basicConfig(
            format="%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=level,
            handlers=handlers,
        )

        if env in [ENV_LOCAL, ENV_DEV]:
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpcore").setLevel(logging.WARNING)
        elif env == ENV_PROD:
            logging.getLogger("httpx").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)

        return logging.getLogger(name)

    def set_level(self, level):
        if self.enabled:
            self.logger.setLevel(level)

    def log_exception(self, e: Exception):
        if not self.enabled:
            return
        error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        self.logger.error(error_trace)

    def error(self, msg, *args, **kwargs):
        if not self.enabled:
            return
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if not self.enabled:
            return
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not self.enabled or not self.is_debug():
            return
        self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if not self.enabled or not self.is_debug():
            return
        self.logger.debug(msg, *args, **kwargs)

    def is_debug(self):
        return self.env in [ENV_LOCAL, ENV_DEV]
