"""
ScoringSys.py

Backwards-compatible entry point. The implementation now lives in the ``ssm``
package; this thin shim keeps the original command-line interface (and the
Flask backend's ``python ScoringSys.py ...`` subprocess call) working unchanged.

See:
    ssm.cli            — argument parsing + main()
    ssm.scoring.facade — ScoringFacade (the orchestrating Facade)
"""

from ssm.cli import main

if __name__ == "__main__":
    main()
