from .app import SemiAutoDFTWebApp, run_server

# 兼容旧命名
DFTWebUI = SemiAutoDFTWebApp
serve = run_server

__all__ = ["SemiAutoDFTWebApp", "run_server", "DFTWebUI", "serve"]
