"""确保 backend 启动时同时加载用户表和会话记忆表。"""

from . import user_models  # noqa: F401
from . import session_memory_models  # noqa: F401
